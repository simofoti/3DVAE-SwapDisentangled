import torch.nn

import utils
from mesh_simplification import MeshSimplifier
from model import AE


class ModelManager(torch.nn.Module):
    def __init__(self, configurations, device):
        super(ModelManager, self).__init__()
        self._model_params = configurations['model']
        self.device = device
        self.template = utils.load_template(configurations['template_path'])
        down_transforms, up_transforms = self._compute_transformations()
        spirals_indices = self._compute_spirals()
        self.net = AE(in_channels=3, out_channels=3, latent_channels=None,
                      spiral_indices=spirals_indices,
                      down_transform=down_transforms,
                      up_transform=up_transforms)

    def _compute_transformations(self):
        sampling_params = self._model_params['sampling']
        m = self.template

        r_weighted = False if sampling_params['type'] == 'basic' else True

        low_res_templates = []
        down_transforms = []
        up_transforms = []
        for sampling_factor in sampling_params['global_factors']:
            simplifier = MeshSimplifier(in_mesh=m, debug=True)
            m, down, up = simplifier(sampling_factor, r_weighted)
            low_res_templates.append(m)
            down_transforms.append(down)
            up_transforms.append(up)
        return down_transforms, up_transforms

    def _compute_spirals(self):
        pass

