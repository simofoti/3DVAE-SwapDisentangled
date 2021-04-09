import os
import argparse
import tqdm
import torch.nn

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision.io import write_video
from torchvision.utils import make_grid

import utils
from data_generation_and_loading import get_data_loaders
from model_manager import ModelManager

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configurations/default.yaml',
                    help="Path to the configuration file.")
parser.add_argument('--id', type=str, default='none', help="ID of experiment")
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
opts = parser.parse_args()
config = utils.get_config(opts.config)
model_name = opts.id

output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_dir = os.path.join(output_directory, 'checkpoints')

if not torch.cuda.is_available():
    device = torch.device('cpu')
    print("GPU not available, running on CPU")
else:
    device = torch.device('cuda')

manager = ModelManager(configurations=config, device=device)
manager.resume(checkpoint_dir)

train_loader, _, test_loader, normalization_dict = \
    get_data_loaders(config, manager.template)

test_visualization_batch = next(iter(test_loader))

# ae measure latent mean and std on train set
latents_list = []
for data in tqdm.tqdm(train_loader):
    latents_list.append(manager.encode(data.x.to(device)).detach().cpu())
latents = torch.cat(latents_list, dim=0)
z_means = torch.mean(latents, dim=0)
z_stds = torch.std(latents, dim=0)
z_mins, _ = torch.min(latents, dim=0)
z_maxs, _ = torch.max(latents, dim=0)

# print(z_means)
# print(z_mins)
# print(z_maxs)
# manager.show_mesh(manager.generate(z_means.to(device)), normalization_dict)
# manager.show_mesh(manager.generate(z_mins.to(device)), normalization_dict)
# manager.show_mesh(manager.generate(z_maxs.to(device)), normalization_dict)

n_steps = 10
all_frames, max_distances = [], []
for i in tqdm.tqdm(range(z_means.shape[0])):
    z = z_means.repeat(n_steps, 1)
    z[:, i] = torch.linspace(z_mins[i], z_maxs[i], n_steps).to(device)
    gen_verts = manager.generate(z.to(device)) * \
        normalization_dict['std'].to(device) + \
        normalization_dict['mean'].to(device)

    differences_from_first = manager.compute_vertex_errors(
        gen_verts, gen_verts[0].expand(gen_verts.shape[0], -1, -1))
    max_distances.append(differences_from_first[-1, ::])
    renderings = manager.render(gen_verts).cpu()
    differences_renderings = manager.render(gen_verts, differences_from_first,
                                            error_max_scale=5).cpu()
    frames = torch.cat([renderings, differences_renderings], dim=-1)
    all_frames.append(torch.cat([frames, torch.zeros_like(frames)[:2, ::]]))

write_video(os.path.join(output_directory, 'latent_exploration.mp4'),
            torch.cat(all_frames, dim=0).permute(0, 2, 3, 1) * 255, fps=4)

grid_frames = []
stacked_frames = torch.stack(all_frames)[:, :-2, ::]
for i in range(stacked_frames.shape[1]):
    grid_frames.append(
        make_grid(stacked_frames[:, i, ::], padding=10, pad_value=1))
write_video(os.path.join(output_directory, 'latent_exploration_tiled.mp4'),
            torch.stack(grid_frames, dim=0).permute(0, 2, 3, 1) * 255, fps=1)

df = pd.DataFrame(columns=['mean_dist', 'z_var', 'region'])
df_row = 0
for zi, vert_distances in enumerate(max_distances):
    for region, indices in manager.template.feat_and_cont.items():
        regional_distances = vert_distances[indices['feature']]
        mean_regional_distance = torch.mean(regional_distances)
        df.loc[df_row] = [mean_regional_distance.item(), zi, region]
        df_row += 1

sns.set_theme(style="ticks")


def string_to_color(rgba_string, swap_bw=True):
    rgba_string = rgba_string[1:-1]  # remove [ and ]
    rgb_values = rgba_string.split()[:-1]
    colors = [int(c) / 255 for c in rgb_values]
    if colors == [1., 1., 1.] and swap_bw:
        colors = [0., 0., 0.]
    return tuple(colors)


palette = {k: string_to_color(k) for k in manager.template.feat_and_cont.keys()}
grid = sns.FacetGrid(df, col="region", hue="region", palette=palette,
                     col_wrap=4, height=3)

grid.map(plt.plot, "z_var", "mean_dist", marker="o")
plt.savefig(os.path.join(output_directory, 'latent_exploration.png'))
plt.show()
