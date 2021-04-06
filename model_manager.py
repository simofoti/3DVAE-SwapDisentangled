import os
import pickle
import torch.nn

import utils
from mesh_simplification import MeshSimplifier
from compute_spirals import preprocess_spiral
from model import AE


class ModelManager(torch.nn.Module):
    def __init__(self, configurations, device,
                 precomputed_storage_path='precomputed'):
        super(ModelManager, self).__init__()
        self._model_params = configurations['model']
        self._optimization_params = configurations['optimization']
        self._precomputed_storage_path = precomputed_storage_path
        self.device = device
        self.template = utils.load_template(
            configurations['data']['template_path'])
        low_res_templates, down_transforms, up_transforms = \
            self._precompute_transformations()
        meshes_all_resolutions = [self.template] + low_res_templates
        spirals_indices = self._precompute_spirals(meshes_all_resolutions)

        self._net = AE(in_channels=self._model_params['in_channels'],
                       out_channels=self._model_params['out_channels'],
                       latent_size=self._model_params['latent_size'],
                       spiral_indices=spirals_indices,
                       down_transform=down_transforms,
                       up_transform=up_transforms).to(device)

        self._optimizer = torch.optim.Adam(
            self._net.parameters(),
            lr=float(self._optimization_params['lr']),
            weight_decay=self._optimization_params['weight_decay'])

        self._losses = None

    @property
    def loss_keys(self):
        return ['reconstruction']

    def _precompute_transformations(self):
        storage_path = os.path.join(self._precomputed_storage_path,
                                    'transforms.pkl')
        try:
            with open(storage_path, 'rb') as file:
                low_res_templates, down_transforms, up_transforms = \
                    pickle.load(file)
        except FileNotFoundError:
            print("Computing Down- and Up- sampling transformations ")
            if not os.path.isdir(self._precomputed_storage_path):
                os.mkdir(self._precomputed_storage_path)

            sampling_params = self._model_params['sampling']
            m = self.template

            r_weighted = False if sampling_params['type'] == 'basic' else True

            low_res_templates = []
            down_transforms = []
            up_transforms = []
            for sampling_factor in sampling_params['sampling_factors']:
                simplifier = MeshSimplifier(in_mesh=m, debug=False)
                m, down, up = simplifier(sampling_factor, r_weighted)
                low_res_templates.append(m)
                down_transforms.append(down)
                up_transforms.append(up)

            with open(storage_path, 'wb') as file:
                pickle.dump(
                    [low_res_templates, down_transforms, up_transforms], file)

        down_transforms = [d.to(self.device) for d in down_transforms]
        up_transforms = [u.to(self.device) for u in up_transforms]
        return low_res_templates, down_transforms, up_transforms

    def _precompute_spirals(self, templates):
        storage_path = os.path.join(self._precomputed_storage_path,
                                    'spirals.pkl')
        try:
            with open(storage_path, 'rb') as file:
                spiral_indices_list = pickle.load(file)
        except FileNotFoundError:
            print("Computing Spirals")
            spirals_params = self._model_params['spirals']
            spiral_indices_list = []
            for i in range(len(templates) - 1):
                spiral_indices_list.append(
                    preprocess_spiral(templates[i].face.t().cpu().numpy(),
                                      spirals_params['length'][i],
                                      templates[i].pos.cpu().numpy(),
                                      spirals_params['dilation'][i]))
            with open(storage_path, 'wb') as file:
                pickle.dump(spiral_indices_list, file)
        spiral_indices_list = [s.to(self.device) for s in spiral_indices_list]
        return spiral_indices_list

    def forward(self, data):
        return self._net(data.x)

    def run_epoch(self, data_loader, device, train=True):
        if train:
            self._net.train()
        else:
            self._net.eval()

        self._reset_losses()
        it = 0
        for it, data in enumerate(data_loader):
            if train:
                losses = self._do_iteration(data, device, train=True)
            else:
                with torch.no_grad():
                    losses = self._do_iteration(data, device, train=False)
            self._add_losses(losses)
        self._divide_losses(it + 1)

    def _do_iteration(self, data, device='cpu', train=True):
        if train:
            self._optimizer.zero_grad()

        data = data.to(device)
        reconstructed = self.forward(data)
        loss_recon = self._compute_mse_loss(reconstructed, data.x)

        if train:
            loss_recon.backward()
            self._optimizer.step()
        return {'reconstruction': loss_recon.item()}

    @staticmethod
    def _compute_l1_loss(prediction, gt, reduction='mean'):
        return torch.nn.L1Loss(reduction=reduction)(prediction, gt)

    @staticmethod
    def _compute_mse_loss(prediction, gt, reduction='mean'):
        return torch.nn.MSELoss(reduction=reduction)(prediction, gt)

    def _reset_losses(self):
        self._losses = {k: 0 for k in self.loss_keys}

    def _add_losses(self, additive_losses):
        for k in self.loss_keys:
            loss = additive_losses[k]
            self._losses[k] += loss.item() if torch.is_tensor(loss) else loss

    def _divide_losses(self, value):
        for k in self.loss_keys:
            self._losses[k] /= value


