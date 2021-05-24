import os
import json
import pickle
import tqdm
import trimesh
import torch.nn
import pytorch3d.loss

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision.io import write_video
from torchvision.utils import make_grid, save_image
from pytorch3d.renderer import BlendParams
from pytorch3d.loss.point_mesh_distance import point_face_distance
from pytorch3d.loss.chamfer import _handle_pointcloud_input
from pytorch3d.ops.knn import knn_points


class Tester:
    def __init__(self, model_manager, norm_dict,
                 train_load, test_load, out_dir, config):
        self._manager = model_manager
        self._device = model_manager.device
        self._norm_dict = norm_dict
        self._normalized_data = config['data']['normalize_data']
        self._out_dir = out_dir
        self._config = config
        self._train_loader = train_load
        self._test_loader = test_load
        self.latent_stats = self.compute_latent_stats(train_load)

        self.coma_landmarks = [
            1337, 1344, 1163, 878, 3632, 2496, 2428, 2291, 2747,
            3564, 1611, 2715, 3541, 1576, 3503, 3400, 3568, 1519,
            203, 183, 870, 900, 867, 3536]
        self.uhm_landmarks = [
            10754, 10826, 9123, 10667, 19674, 28739, 4831, 19585,
            8003, 22260, 12492, 27386, 1969, 31925, 31158, 20963,
            1255, 9881, 32055, 45778, 5355, 27515, 18482, 33691]

    def __call__(self):
        self.set_renderings_size(512)
        self.set_rendering_background_color([1, 1, 1])

        # Qualitative evaluations
        if self._config['data']['swap_features']:
            self.latent_swapping(next(iter(self._test_loader)).x)
        self.per_variable_range_experiments()
        self.random_generation_and_rendering(n_samples=16)
        self.random_generation_and_save(n_samples=16)

        # Quantitative evaluation
        recon_errors = self.reconstruction_errors(self._test_loader)
        train_set_diversity = self.compute_diversity_train_set()
        diversity = self.compute_diversity()
        specificity = self.compute_specificity()
        metrics = {'recon_errors': recon_errors,
                   'train_set_diversity': train_set_diversity,
                   'diversity': diversity,
                   'specificity': specificity}

        outfile_path = os.path.join(self._out_dir, 'eval_metrics.json')
        with open(outfile_path, 'w') as outfile:
            json.dump(metrics, outfile)

    def _unnormalize_verts(self, verts, dev=None):
        d = self._device if dev is None else dev
        return verts * self._norm_dict['std'].to(d) + \
            self._norm_dict['mean'].to(d)

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

            gen_verts = self._manager.generate(z.to(self._device))

            if self._normalized_data:
                gen_verts = self._unnormalize_verts(gen_verts)

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
        gen_verts = self._manager.generate(z.to(self._device))
        if self._normalized_data:
            gen_verts = self._unnormalize_verts(gen_verts)
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
            gt = data.x

            recon, _ = self._manager.forward(data)

            if self._normalized_data:
                gt = self._unnormalize_verts(gt)
                recon = self._unnormalize_verts(recon)

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

            current_verts_batch = x
            if self._normalized_data:
                current_verts_batch = self._unnormalize_verts(
                    current_verts_batch, x.device)

            if previous_verts_batch is not None:
                verts_batch_distances = self._manager.compute_vertex_errors(
                    previous_verts_batch, current_verts_batch)
                mean_distances.append(torch.mean(verts_batch_distances, dim=1))
            previous_verts_batch = current_verts_batch
        return torch.mean(torch.cat(mean_distances, dim=0)).item()

    def compute_diversity(self, n_samples=10000):
        print('Computing generative model diversity')
        samples_per_batch = 20
        mean_distances = []
        for _ in tqdm.tqdm(range(n_samples // samples_per_batch)):
            verts_batch_distances = self._manager.compute_vertex_errors(
                self.random_generation(samples_per_batch),
                self.random_generation(samples_per_batch))
            mean_distances.append(torch.mean(verts_batch_distances, dim=1))
        return torch.mean(torch.cat(mean_distances, dim=0)).item()

    def compute_specificity(self, n_samples=100):
        print('Computing generative model specificity')
        min_distances = []
        for _ in tqdm.tqdm(range(n_samples)):
            sample = self.random_generation(1)
            if self._normalized_data:
                sample = self._unnormalize_verts(sample)

            mean_distances = []
            for data in self._train_loader:
                if self._config['data']['swap_features']:
                    x = data.x[self._manager._batch_diagonal_idx, ::]
                else:
                    x = data.x

                if self._normalized_data:
                    x = self._unnormalize_verts(x.to(self._device))
                else:
                    x = x.to(self._device)

                v_dist = self._manager.compute_vertex_errors(
                    x, sample.expand(x.shape[0], -1, -1))
                mean_distances.append(torch.mean(v_dist, dim=1))
            min_distances.append(torch.min(torch.cat(mean_distances, dim=0)))
        return torch.mean(torch.stack(min_distances)).item()

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

        if self._normalized_data:
            all_verts = self._unnormalize_verts(all_verts)

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

    def fit_vertices(self, target_verts, lr=5e-3, iterations=250,
                     target_noise=0, target_landmarks=None):
        # Scale and position target_verts
        target_verts = target_verts.unsqueeze(0).to(self._device)
        if target_landmarks is None:
            target_landmarks = target_verts[:, self.coma_landmarks, :]
        target_landmarks = target_landmarks.to(self._device)

        if target_noise > 0:
            target_verts = target_verts + (torch.randn_like(target_verts) *
                                           target_noise /
                                           self._manager.to_mm_const)
            target_landmarks = target_landmarks + (
                torch.randn_like(target_landmarks) *
                target_noise / self._manager.to_mm_const)

        z = self.latent_stats['means'].clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr)
        gen_verts = None
        for i in range(iterations):
            optimizer.zero_grad()
            gen_verts = self._manager.generate_for_opt(z.to(self._device))
            if self._normalized_data:
                gen_verts = self._unnormalize_verts(gen_verts)

            if i < iterations // 3:
                er = self._manager._compute_mse_loss(
                    gen_verts[:, self.uhm_landmarks, :], target_landmarks)
            else:
                er, _ = pytorch3d.loss.chamfer_distance(gen_verts, target_verts)

            er.backward()
            optimizer.step()
        return gen_verts, target_verts.squeeze()

    def fit_coma_data(self, base_dir='meshes2fit',
                      noise=0, export_meshes=False):
        print(f"Fitting CoMA meshes with noise = {noise} mm")
        out_mesh_dir = os.path.join(self._out_dir, 'fitting')
        if not os.path.isdir(out_mesh_dir):
            os.mkdir(out_mesh_dir)

        names_and_scale = {}
        for dirpath, _, fnames in os.walk(base_dir):
            for f in fnames:
                if f.endswith('.ply'):
                    if f[:5] in ['03274', '03275', '00128', '03277']:
                        names_and_scale[f] = 9
                    else:
                        names_and_scale[f] = 8

        dataframes = []
        for m_id, scale in tqdm.tqdm(names_and_scale.items()):
            df_id = m_id.split('.')[0]
            subd = False
            mesh_path = os.path.join(base_dir, m_id)
            target_mesh = trimesh.load_mesh(mesh_path, 'ply', process=False)
            target_verts = torch.tensor(
                target_mesh.vertices, dtype=torch.float,
                requires_grad=False, device=self._device)

            # scale and translate to match template. Values manually computed
            target_verts *= scale
            target_verts[:, 1] += 0.15

            # If target mesh was subdivided use original target to retrieve its
            # landmarks
            target_landmarks = None
            if 'subd' in m_id:
                subd = True
                df_id = m_id.split('_')[0]
                base_path = os.path.join(base_dir, m_id.split('_')[0] + '.ply')
                base_mesh = trimesh.load_mesh(base_path, 'ply', process=False)
                base_verts = torch.tensor(
                    base_mesh.vertices, dtype=torch.float,
                    requires_grad=False, device=self._device)
                target_landmarks = base_verts[self.coma_landmarks, :]
                target_landmarks = target_landmarks.unsqueeze(0)
                target_landmarks *= scale
                target_landmarks[:, 1] += 0.15

            out_verts, target_verts = self.fit_vertices(
                target_verts, target_noise=noise,
                target_landmarks=target_landmarks)

            closest_p_errors = self._manager.to_mm_const * \
                self._dist_closest_point(out_verts, target_verts.unsqueeze(0))

            dataframes.append(pd.DataFrame(
                {'id': df_id, 'noise': noise, 'subdivided': subd,
                 'errors': closest_p_errors.squeeze().detach().cpu().numpy()}))

            if export_meshes:
                mesh_name = m_id.split('.')[0]
                out_mesh = trimesh.Trimesh(
                    out_verts[0, ::].cpu().detach().numpy(),
                    self._manager.template.face.t().cpu().numpy())
                out_mesh.export(os.path.join(
                    out_mesh_dir, mesh_name + f"_fit_{str(noise)}" + '.ply'))
                target_mesh.vertices = target_verts.detach().cpu().numpy()
                target_mesh.export(os.path.join(
                    out_mesh_dir, mesh_name + f"_t_{str(noise)}" + '.ply'))
        return pd.concat(dataframes)

    def fit_coma_data_different_noises(self, base_dir='meshes2fit'):
        noises = [0, 2, 4, 6, 8]
        dataframes = []
        for n in noises:
            dataframes.append(self.fit_coma_data(base_dir, n, True))
        df = pd.concat(dataframes)
        sns.set_theme(style="ticks")
        plt.figure()
        sns.lineplot(data=df, x='noise', y='errors', style='subdivided',
                     markers=True, dashes=False, ci='sd')
        plt.savefig(os.path.join(self._out_dir, 'coma_fitting.svg'))

        plt.figure()
        sns.boxplot(data=df, x='noise', y='errors', hue='subdivided',
                    showfliers=False)
        plt.savefig(os.path.join(self._out_dir, 'coma_fitting_box.svg'))

        plt.figure()
        sns.violinplot(data=df[df.errors < 3], x='noise', y='errors',
                       hue='subdivided', split=True)
        plt.savefig(os.path.join(self._out_dir, 'coma_fitting_violin.svg'))

    @staticmethod
    def _point_mesh_distance(points, verts, faces):
        points = points.squeeze()
        verts_packed = verts.to(points.device)
        faces_packed = torch.tensor(faces, device=points.device).t()
        first_idx = torch.tensor([0], device=points.device)

        tris = verts_packed[faces_packed]

        point_to_face = point_face_distance(points, first_idx, tris,
                                            first_idx, points.shape[0])
        return point_to_face / points.shape[0]

    @staticmethod
    def _dist_closest_point(x, y):
        # for each point on x return distance to closest point in y
        x, x_lengths, x_normals = _handle_pointcloud_input(x, None, None)
        y, y_lengths, y_normals = _handle_pointcloud_input(y, None, None)
        x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
        cham_x = x_nn.dists[..., 0]
        return cham_x


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
    model_name = opts.id

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

    # tester()
    tester.fit_coma_data_different_noises()
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
