import os
import json
import pickle
import tqdm
import trimesh
import torch.nn

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision.io import write_video
from torchvision.utils import make_grid, save_image
from pytorch3d.renderer import BlendParams


class Tester:
    def __init__(self, model_manager, norm_dict,
                 train_load, test_load, out_dir, config):
        self._manager = model_manager
        self._device = model_manager.device
        self._norm_dict = norm_dict
        self._out_dir = out_dir
        self._config = config
        self._train_loader = train_load
        self._test_loader = test_load
        self.latent_stats = self.compute_latent_stats(train_load)

    def __call__(self):
        self.set_renderings_size(512)
        self.set_rendering_background_color([1, 1, 1])

        # Qualitative evaluations
        self.latent_swapping(next(iter(self._test_loader)).x)
        self.per_variable_range_experiments()
        self.random_generation_and_rendering(n_samples=16)
        self.random_generation_and_save(n_samples=16)

        # Quantitative evaluation
        recon_errors = self.reconstruction_errors(self._test_loader)
        train_set_diversity = self.compute_diversity_train_set()
        diversity = self.compute_diversity()
        regional_diversity = self.compute_regional_diversity()
        specificity = self.compute_specificity()
        metrics = {'recon_errors': recon_errors,
                   'train_set_diversity': train_set_diversity,
                   'diversity': diversity,
                   'regional_diversity': regional_diversity,
                   'specificity': specificity}

        outfile_path = os.path.join(self._out_dir, 'eval_metrics.json')
        with open(outfile_path, 'w') as outfile:
            json.dump(metrics, outfile)

    def set_renderings_size(self, size):
        self._manager.renderer.rasterizer.raster_settings.image_size = size

    def set_rendering_background_color(self, color=None):
        color = [1, 1, 1] if color is None else color
        blend_params = BlendParams(background_color=color)
        self._manager._default_shader.blend_params = blend_params
        self._manager._simple_shader.blend_params = blend_params

    def compute_latent_stats(self, data_loader):
        storage_path = os.path.join(self._out_dir, 'z_stats.pkl')
        try:
            with open(storage_path, 'rb') as file:
                z_stats = pickle.load(file)
        except FileNotFoundError:
            latents_list = []
            for data in tqdm.tqdm(data_loader):
                latents_list.append(self._manager.encode(
                    data.x.to(self._device)).detach().cpu())
            latents = torch.cat(latents_list, dim=0)
            z_means = torch.mean(latents, dim=0)
            z_stds = torch.std(latents, dim=0)
            z_mins, _ = torch.min(latents, dim=0)
            z_maxs, _ = torch.max(latents, dim=0)
            z_stats = {'means': z_means, 'stds': z_stds,
                       'mins': z_mins, 'maxs': z_maxs}

            with open(storage_path, 'wb') as file:
                pickle.dump(z_stats, file)
        return z_stats

    @staticmethod
    def string_to_color(rgba_string, swap_bw=True):
        rgba_string = rgba_string[1:-1]  # remove [ and ]
        rgb_values = rgba_string.split()[:-1]
        colors = [int(c) / 255 for c in rgb_values]
        if colors == [1., 1., 1.] and swap_bw:
            colors = [0., 0., 0.]
        return tuple(colors)

    def per_variable_range_experiments(self, z_range_multiplier=1):
        z_means = self.latent_stats['means']
        z_mins = self.latent_stats['mins'] * z_range_multiplier
        z_maxs = self.latent_stats['maxs'] * z_range_multiplier

        # Create video perturbing each latent variable from min to max.
        # Show generated mesh and error map next to each other
        # Frames are all concatenated along the same direction. A black frame is
        # added before start perturbing the next latent variable
        n_steps = 10
        all_frames, all_rendered_differences, max_distances = [], [], []
        for i in tqdm.tqdm(range(z_means.shape[0])):
            z = z_means.repeat(n_steps, 1)
            z[:, i] = torch.linspace(
                z_mins[i], z_maxs[i], n_steps).to(self._device)
            gen_verts = self._manager.generate(z.to(self._device)) * \
                self._norm_dict['std'].to(self._device) + \
                self._norm_dict['mean'].to(self._device)

            differences_from_first = self._manager.compute_vertex_errors(
                gen_verts, gen_verts[0].expand(gen_verts.shape[0], -1, -1))
            max_distances.append(differences_from_first[-1, ::])
            renderings = self._manager.render(gen_verts).cpu()
            differences_renderings = self._manager.render(
                gen_verts, differences_from_first, error_max_scale=5).cpu()
            all_rendered_differences.append(differences_renderings)
            frames = torch.cat([renderings, differences_renderings], dim=-1)
            all_frames.append(
                torch.cat([frames, torch.zeros_like(frames)[:2, ::]]))

        write_video(
            os.path.join(self._out_dir, 'latent_exploration.mp4'),
            torch.cat(all_frames, dim=0).permute(0, 2, 3, 1) * 255, fps=4)

        # Same video as before, but effects of perturbing each latent variables
        # are shown in the same frame. Only error maps are shown.
        grid_frames = []
        grid_nrows = 8
        if self._config['data']['swap_features']:
            z_size = self._config['model']['latent_size']
            grid_nrows = z_size // len(self._manager.latent_regions)

        stacked_frames = torch.stack(all_rendered_differences)
        for i in range(stacked_frames.shape[1]):
            grid_frames.append(
                make_grid(stacked_frames[:, i, ::], padding=10,
                          pad_value=1, nrow=grid_nrows))
        save_image(grid_frames[-1],
                   os.path.join(self._out_dir, 'latent_exploration_tiled.png'))
        write_video(
            os.path.join(self._out_dir, 'latent_exploration_tiled.mp4'),
            torch.stack(grid_frames, dim=0).permute(0, 2, 3, 1) * 255, fps=1)

        # Create a plot showing the effects of perturbing latent variables in
        # each region of the face
        df = pd.DataFrame(columns=['mean_dist', 'z_var', 'region'])
        df_row = 0
        for zi, vert_distances in enumerate(max_distances):
            for region, indices in self._manager.template.feat_and_cont.items():
                regional_distances = vert_distances[indices['feature']]
                mean_regional_distance = torch.mean(regional_distances)
                df.loc[df_row] = [mean_regional_distance.item(), zi, region]
                df_row += 1

        sns.set_theme(style="ticks")
        palette = {k: self.string_to_color(k) for k in
                   self._manager.template.feat_and_cont.keys()}
        grid = sns.FacetGrid(df, col="region", hue="region", palette=palette,
                             col_wrap=4, height=3)

        grid.map(plt.plot, "z_var", "mean_dist", marker="o")
        plt.savefig(os.path.join(self._out_dir, 'latent_exploration.svg'))

    def random_latent(self, n_samples, z_range_multiplier=1):
        z_means = self.latent_stats['means']
        z_mins = self.latent_stats['mins'] * z_range_multiplier
        z_maxs = self.latent_stats['maxs'] * z_range_multiplier

        uniform = torch.rand([n_samples, z_means.shape[0]],
                             device=z_means.device)
        return uniform * (z_maxs - z_mins) + z_mins

    def random_generation(self, n_samples=16, z_range_multiplier=1):
        z = self.random_latent(n_samples, z_range_multiplier)

        gen_verts = self._manager.generate(z.to(self._device)) * \
            self._norm_dict['std'].to(self._device) + \
            self._norm_dict['mean'].to(self._device)
        return gen_verts

    def random_generation_and_rendering(self, n_samples=16,
                                        z_range_multiplier=1):
        gen_verts = self.random_generation(n_samples, z_range_multiplier)
        renderings = self._manager.render(gen_verts).cpu()
        grid = make_grid(renderings, padding=10, pad_value=1)
        save_image(grid, os.path.join(self._out_dir, 'random_generation.png'))

    def random_generation_and_save(self, n_samples=16, z_range_multiplier=1):
        out_mesh_dir = os.path.join(self._out_dir, 'random_meshes')
        if not os.path.isdir(out_mesh_dir):
            os.mkdir(out_mesh_dir)

        gen_verts = self.random_generation(n_samples, z_range_multiplier)

        self.save_batch(gen_verts, out_mesh_dir)

    def save_batch(self, batch_verts, out_mesh_dir):
        for i in range(batch_verts.shape[0]):
            mesh = trimesh.Trimesh(
                batch_verts[i, ::].cpu().detach().numpy(),
                self._manager.template.face.t().cpu().numpy())
            mesh.export(os.path.join(out_mesh_dir, str(i) + '.ply'))

    def reconstruction_errors(self, data_loader):
        print('Compute reconstruction errors')
        data_errors = []
        for data in tqdm.tqdm(data_loader):
            if self._config['data']['swap_features']:
                data.x = data.x[self._manager._batch_diagonal_idx, ::]
            data = data.to(self._device)

            recon, _ = self._manager.forward(data)

            gt = data.x * self._norm_dict['std'].to(self._device) + \
                self._norm_dict['mean'].to(self._device)
            recon = recon * self._norm_dict['std'].to(self._device) + \
                self._norm_dict['mean'].to(self._device)
            errors = self._manager.compute_vertex_errors(recon, gt)
            data_errors.append(torch.mean(errors.detach(), dim=1))
        data_errors = torch.cat(data_errors, dim=0)
        return {'mean': torch.mean(data_errors).item(),
                'median': torch.median(data_errors).item(),
                'max': torch.max(data_errors).item()}

    def compute_diversity_train_set(self):
        print('Computing train set diversity')
        previous_verts_batch = None
        mean_distances = []
        for data in tqdm.tqdm(self._train_loader):
            if self._config['data']['swap_features']:
                x = data.x[self._manager._batch_diagonal_idx, ::]
            else:
                x = data.x

            current_verts_batch = x * self._norm_dict['std'].to(x.device) + \
                self._norm_dict['mean'].to(x.device)

            if previous_verts_batch is not None:
                verts_batch_distances = self._manager.compute_vertex_errors(
                    previous_verts_batch, current_verts_batch)
                mean_distances.append(torch.mean(verts_batch_distances, dim=1))
            previous_verts_batch = current_verts_batch
        return torch.mean(torch.cat(mean_distances, dim=0))

    def compute_diversity(self, n_samples=10000):
        print('Computing generative model diversity')
        samples_per_batch = 20
        mean_distances = []
        for _ in tqdm.tqdm(range(n_samples // samples_per_batch)):
            verts_batch_distances = self._manager.compute_vertex_errors(
                self.random_generation(samples_per_batch),
                self.random_generation(samples_per_batch))
            mean_distances.append(torch.mean(verts_batch_distances, dim=1))
        return torch.mean(torch.cat(mean_distances, dim=0))

    def compute_regional_diversity(self, n_samples=1000):
        print('Computing generative model regional diversity')
        samples_per_batch = 20
        r_diversity = {k: None for k in self._manager.latent_regions.keys()}

        for key, z_region in tqdm.tqdm(self._manager.latent_regions.items()):
            z_mean = self.latent_stats['means'].expand(samples_per_batch, -1)
            region_dist = []
            not_region_dist = []
            for _ in range(n_samples):
                z_rand_0 = self.random_latent(samples_per_batch)
                z_rand_1 = self.random_latent(samples_per_batch)
                z_0, z_1 = z_mean.clone(), z_mean.clone()
                z_0[z_region[0]:z_region[1]] = z_rand_0[z_region[0]:z_region[1]]
                z_1[z_region[0]:z_region[1]] = z_rand_1[z_region[0]:z_region[1]]

                x_0 = self._manager.generate(z_0.to(self._device)) * \
                    self._norm_dict['std'].to(self._device) + \
                    self._norm_dict['mean'].to(self._device)
                x_1 = self._manager.generate(z_1.to(self._device)) * \
                    self._norm_dict['std'].to(self._device) + \
                    self._norm_dict['mean'].to(self._device)

                verts_distances = self._manager.compute_vertex_errors(x_0, x_1)
                verts_region_idx = \
                    self._manager.template.feat_and_cont[key]['feature']
                mask_not_region = torch.ones(verts_distances.shape[-1],
                                             dtype=torch.bool,
                                             device=verts_distances.device)
                mask_not_region[verts_region_idx] = 0
                region_dist.append(
                    torch.mean(verts_distances[:, verts_region_idx], dim=1))
                not_region_dist.append(
                    torch.mean(verts_distances[:, mask_not_region], dim=1))

            r_diversity[key] = {
                'region': torch.mean(torch.cat(region_dist, dim=0)).item(),
                'outside': torch.mean(torch.cat(not_region_dist, dim=0)).item()}
        return r_diversity

    def compute_specificity(self, n_samples=100):
        print('Computing generative model specificity')
        min_distances = []
        for _ in tqdm.tqdm(range(n_samples)):
            mean_distances = []
            for data in self._train_loader:
                if self._config['data']['swap_features']:
                    x = data.x[self._manager._batch_diagonal_idx, ::]
                else:
                    x = data.x

                x = x.to(self._device)
                x = x * self._norm_dict['std'].to(self._device) + \
                    self._norm_dict['mean'].to(self._device)
                sample = self.random_generation(1) * \
                    self._norm_dict['std'].to(self._device) + \
                    self._norm_dict['mean'].to(self._device)
                v_dist = self._manager.compute_vertex_errors(
                    x, sample.expand(x.shape[0], -1, -1))
                mean_distances.append(torch.mean(v_dist, dim=1))
            min_distances.append(torch.min(torch.cat(mean_distances, dim=0)))
        return torch.mean(torch.stack(min_distances))

    def latent_swapping(self, v_batch=None):
        if v_batch is None:
            v_batch = self.random_generation(2)
        else:
            assert v_batch.shape[0] >= 2
            v_batch = v_batch.to(self._device)
            if self._config['data']['swap_features']:
                v_batch = v_batch[self._manager._batch_diagonal_idx, ::]
            v_batch = v_batch[:2, ::]

        z = self._manager.encode(v_batch)
        z_0, z_1 = z[0, ::], z[1, ::]

        swapped_verts = []
        for key, z_region in self._manager.latent_regions.items():
            z_swap = z_0.clone()
            z_swap[z_region[0]:z_region[1]] = z_1[z_region[0]:z_region[1]]
            swapped_verts.append(self._manager.generate(z_swap))

        all_verts = torch.cat([v_batch, torch.cat(swapped_verts, dim=0)], dim=0)
        all_verts = all_verts * self._norm_dict['std'].to(self._device) + \
            self._norm_dict['mean'].to(self._device)

        out_mesh_dir = os.path.join(self._out_dir, 'latent_swapping')
        if not os.path.isdir(out_mesh_dir):
            os.mkdir(out_mesh_dir)
        self.save_batch(all_verts, out_mesh_dir)

        source_dist = self._manager.compute_vertex_errors(
            all_verts, all_verts[0, ::].expand(all_verts.shape[0], -1, -1))
        target_dist = self._manager.compute_vertex_errors(
            all_verts, all_verts[1, ::].expand(all_verts.shape[0], -1, -1))

        renderings = self._manager.render(all_verts)
        renderings_source = self._manager.render(all_verts, source_dist, 5)
        renderings_target = self._manager.render(all_verts, target_dist, 5)
        grid = make_grid(torch.cat(
            [renderings, renderings_source, renderings_target], dim=-2),
            padding=10, pad_value=1, nrow=renderings.shape[0])
        save_image(grid, os.path.join(out_mesh_dir, 'latent_swapping.png'))


if __name__ == '__main__':
    import argparse
    import utils
    from data_generation_and_loading import get_data_loaders
    from model_manager import ModelManager

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='configurations/default.yaml',
                        help="Path to the configuration file.")
    parser.add_argument('--id', type=str, default='none',
                        help="ID of experiment")
    parser.add_argument('--output_path', type=str, default='.',
                        help="outputs path")
    opts = parser.parse_args()
    configurations = utils.get_config(opts.config)
    model_name = '033_dummy'  # opts.id

    output_directory = os.path.join(opts.output_path + "/outputs", model_name)
    checkpoint_dir = os.path.join(output_directory, 'checkpoints')

    if not torch.cuda.is_available():
        device = torch.device('cpu')
        print("GPU not available, running on CPU")
    else:
        device = torch.device('cuda')

    manager = ModelManager(configurations=configurations, device=device)
    manager.resume(checkpoint_dir)

    train_loader, _, test_loader, normalization_dict = \
        get_data_loaders(configurations, manager.template)

    tester = Tester(manager, normalization_dict, train_loader, test_loader,
                    output_directory, configurations)

    tester()
    # tester.set_renderings_size(512)
    # tester.set_rendering_background_color()
    # tester.latent_swapping(next(iter(test_loader)).x)
    # tester.per_variable_range_experiments()
    # tester.random_generation_and_rendering(n_samples=16)
    # tester.random_generation_and_save(n_samples=16)
    # print(tester.reconstruction_errors(test_loader))
    # print(tester.compute_specificity(train_loader, 100))
    # print(tester.compute_diversity_train_set())
    # print(tester.compute_diversity())
    # print(tester.compute_regional_diversity(10))
