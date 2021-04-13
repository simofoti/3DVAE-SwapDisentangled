import numba
import random
import torch

import numpy as np

from torch_geometric.data import Data


class SwapFeatures:
    def __init__(self, template):
        self._template = template
        self._features_and_contours = template.feat_and_cont
        self._zones_keys = list(template.feat_and_cont.keys())

    def __call__(self, batched_data):
        batch_size = batched_data.x.shape[0]
        new_batch = torch.zeros([batch_size ** 2, *batched_data.x.shape[1:]],
                                device=batched_data.x.device,
                                dtype=batched_data.x.dtype)
        for i in range(batch_size):
            for j in range(batch_size):
                if i == j:
                    new_batch[i * batch_size + j, ::] = batched_data.x[i, ::]
                else:
                    key = random.choice(self._zones_keys)
                    vertices = batched_data.x.numpy()
                    new_batch[i * batch_size + j, ::] = self.swap(
                        vertices[i, ::], vertices[j, ::], key)
        batched_data = Data(x=new_batch)
        return batched_data

    def swap(self, verts0, verts1, feature_key):
        feature_idxs = self._features_and_contours[feature_key]['feature']
        contour_idxs = self._features_and_contours[feature_key]['contour']

        feature_verts0 = verts0[feature_idxs]
        contour_verts0 = verts0[contour_idxs]

        feature_verts1 = verts1[feature_idxs]
        contour_verts1 = verts1[contour_idxs]

        feature_verts_def = _swap(contour_verts0, feature_verts0,
                                  contour_verts1, feature_verts1)

        # assemble final face with new feature
        verts0[feature_idxs] = feature_verts_def
        return torch.tensor(verts0)


@numba.njit
def _swap(contour_verts0, feature_verts0, contour_verts1, feature_verts1):
    # compute lest square transformation to fit feature on face
    hom_contour_verts0 = to_homogeneous(contour_verts0)
    hom_contour_verts1 = to_homogeneous(contour_verts1)
    transform, _, _, _ = np.linalg.lstsq(hom_contour_verts1,
                                         hom_contour_verts0,
                                         rcond=-1.0)

    # apply transformation
    hom_feature_verts1 = to_homogeneous(feature_verts1)
    hom_feature_verts1 = hom_feature_verts1 @ transform
    feature_verts1 = to_cartesian(hom_feature_verts1)

    # deform feature0 on feature1
    distances = compute_minimum_distances(feature_verts1, contour_verts0)
    max_dist = max(distances)
    displacement_weights = np.tanh(3 * (distances / max_dist))
    displacement_weights = np.expand_dims(displacement_weights, axis=1)
    displacement = feature_verts1 - feature_verts0
    feature_verts_def = feature_verts0 + displacement * displacement_weights
    return feature_verts_def


@numba.njit
def to_homogeneous(x):
    return np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)


@numba.njit
def to_cartesian(x):
    x = x[:, :3] / x[:, 3:4]
    return x[:, :3]


@numba.njit
def compute_minimum_distances(feature, contour):
    distances = []
    for f in feature:
        distances.append(min(np.sum((contour - f) ** 2, axis=1)))
    return np.array(distances)


@numba.njit
def smooth_step(x, end_step=1.):
    return np.where(x > end_step, 1,
                    3 * (x / end_step) ** 2 - 2 * (x / end_step) ** 3)


