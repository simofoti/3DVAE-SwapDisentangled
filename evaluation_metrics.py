"""
Refactored from
https://github.com/stevenygd/PointFlow/blob/master/metrics/evaluation_metrics.py
Replaced Chamfer and EMD implementations with the more efficients one from
pytorch3d and geomloss respectively.
"""
import torch
import geomloss
import warnings
import numpy as np

from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from pytorch3d.loss import chamfer_distance


# Modified from original implementation to avoid compilation of CUDA kernels
def emd_approx(sample, ref):
    return geomloss.SamplesLoss()(sample.cpu(), ref.cpu())


def _pairwise_emd_cd_(sample_pcs, ref_pcs, batch_size):
    n_sample = sample_pcs.shape[0]
    n_ref = ref_pcs.shape[0]
    all_cd = []
    all_emd = []
    iterator = range(n_sample)
    for sample_b_start in iterator:
        sample_batch = sample_pcs[sample_b_start]

        cd_lst = []
        emd_lst = []
        for ref_b_start in range(0, n_ref, batch_size):
            ref_b_end = min(n_ref, ref_b_start + batch_size)
            ref_batch = ref_pcs[ref_b_start:ref_b_end]

            batch_size_ref = ref_batch.size(0)
            sample_batch_exp = sample_batch.view(1, -1, 3).expand(
                batch_size_ref, -1, -1)
            sample_batch_exp = sample_batch_exp.contiguous()

            cd_lst.append(chamfer_distance(sample_batch_exp, ref_batch,
                                           batch_reduction=None)[0])

            emd_batch = emd_approx(sample_batch_exp, ref_batch)
            emd_lst.append(emd_batch.view(1, -1))

        cd_lst = torch.cat(cd_lst, dim=1)
        emd_lst = torch.cat(emd_lst, dim=1)
        all_cd.append(cd_lst)
        all_emd.append(emd_lst)

    all_cd = torch.cat(all_cd, dim=0)  # n_sample, n_ref
    all_emd = torch.cat(all_emd, dim=0)  # n_sample, n_ref
    return all_cd, all_emd


# Adapted from
# https://github.com/xuqiantong/GAN-Metrics/blob/master/framework/metric.py
def knn(m_xx, m_xy, m_yy, k, sqrt=False):
    n0 = m_xx.size(0)
    n1 = m_yy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1))).to(m_xx)
    mat = torch.cat((torch.cat((m_xx, m_xy), 1),
                    torch.cat((m_xy.transpose(0, 1), m_yy), 1)), 0)
    if sqrt:
        mat = mat.abs().sqrt()

    val, idx = (mat + torch.diag(
        float('inf') * torch.ones(n0 + n1).to(m_xx))).topk(k, 0, False)

    count = torch.zeros(n0 + n1).to(m_xx)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) *
                    torch.ones(n0 + n1).to(m_xx)).float()

    s = {
        'tp': (pred * label).sum(),
        'fp': (pred * (1 - label)).sum(),
        'fn': ((1 - pred) * label).sum(),
        'tn': ((1 - pred) * (1 - label)).sum(),
    }

    s.update({
        'precision': s['tp'] / (s['tp'] + s['fp'] + 1e-10),
        'recall': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_t': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_f': s['tn'] / (s['tn'] + s['fp'] + 1e-10),
        'acc': torch.eq(label, pred).float().mean(),
    })
    return s


def lgan_mmd_cov(all_dist):
    n_sample, n_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(n_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        'lgan_mmd': mmd,
        'lgan_cov': cov,
        'lgan_mmd_smp': mmd_smp,
    }


def compute_all_metrics(sample_pcs, ref_pcs, batch_size):
    results = {}

    m_rs_cd, m_rs_emd = _pairwise_emd_cd_(sample_pcs, ref_pcs, batch_size)

    res_cd = lgan_mmd_cov(m_rs_cd.t())
    results.update({
        "%s-CD" % k: v for k, v in res_cd.items()
    })

    res_emd = lgan_mmd_cov(m_rs_emd.t())
    results.update({
        "%s-EMD" % k: v for k, v in res_emd.items()
    })

    m_rr_cd, m_rr_emd = _pairwise_emd_cd_(ref_pcs, ref_pcs, batch_size)
    m_ss_cd, m_ss_emd = _pairwise_emd_cd_(sample_pcs, sample_pcs, batch_size)

    # 1-NN results
    one_nn_cd_res = knn(m_rr_cd, m_rs_cd, m_ss_cd, 1, sqrt=False)
    results.update({
        "1-NN-CD-%s" % k: v for k, v in one_nn_cd_res.items() if 'acc' in k
    })
    one_nn_emd_res = knn(m_rr_emd, m_rs_emd, m_ss_emd, 1, sqrt=False)
    results.update({
        "1-NN-EMD-%s" % k: v for k, v in one_nn_emd_res.items() if 'acc' in k
    })

    return results


#######################################################
# JSD : from https://github.com/optas/latent_3d_points
#######################################################
def unit_cube_grid_point_cloud(resolution, clip_sphere=False):
    """Returns the center coordinates of each cell of a 3D grid with
    resolution^3 cells, that is placed in the unit-cube.
    If clip_sphere it True it drops the "corner" cells that
    lie outside the unit-sphere.
    """
    grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / float(resolution - 1)
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                grid[i, j, k, 0] = i * spacing - 0.5
                grid[i, j, k, 1] = j * spacing - 0.5
                grid[i, j, k, 2] = k * spacing - 0.5

    if clip_sphere:
        grid = grid.reshape(-1, 3)
        grid = grid[norm(grid, axis=1) <= 0.5]

    return grid, spacing


def jsd_between_point_cloud_sets(sample_pcs, ref_pcs, resolution=28):
    """Computes the JSD between two sets of point-clouds,
    as introduced in the paper
    ```Learning Representations And Generative Models For 3D Point Clouds```.
    Args:
        sample_pcs: (np.ndarray S1xR2x3) S1 point-clouds, each of R1 points.
        ref_pcs: (np.ndarray S2xR2x3) S2 point-clouds, each of R2 points.
        resolution: (int) grid-resolution. Affects granularity of measurements.
    """
    in_unit_sphere = True
    sample_grid_var = entropy_of_occupancy_grid(sample_pcs,
                                                resolution, in_unit_sphere)[1]
    ref_grid_var = entropy_of_occupancy_grid(ref_pcs,
                                             resolution, in_unit_sphere)[1]
    return jensen_shannon_divergence(sample_grid_var, ref_grid_var)


def entropy_of_occupancy_grid(pclouds, grid_resolution,
                              in_sphere=False, verbose=False):
    """Given a collection of point-clouds, estimate the entropy of the random
    variables corresponding to occupancy-grid activation patterns.
    Inputs:
        pclouds: (numpy array) #point-clouds x points per point-cloud x 3
        grid_resolution (int) size of occupancy grid that will be used.
    """
    epsilon = 10e-4
    bound = 0.5 + epsilon
    if abs(np.max(pclouds)) > bound or abs(np.min(pclouds)) > bound:
        if verbose:
            warnings.warn('Point-clouds are not in unit cube.')

    if in_sphere and np.max(np.sqrt(np.sum(pclouds ** 2, axis=2))) > bound:
        if verbose:
            warnings.warn('Point-clouds are not in unit sphere.')

    grid_coordinates, _ = unit_cube_grid_point_cloud(grid_resolution, in_sphere)
    grid_coordinates = grid_coordinates.reshape(-1, 3)
    grid_counters = np.zeros(len(grid_coordinates))
    grid_bernoulli_rvars = np.zeros(len(grid_coordinates))
    nn = NearestNeighbors(n_neighbors=1).fit(grid_coordinates)

    for pc in pclouds:
        _, indices = nn.kneighbors(pc)
        indices = np.squeeze(indices)
        for i in indices:
            grid_counters[i] += 1
        indices = np.unique(indices)
        for i in indices:
            grid_bernoulli_rvars[i] += 1

    acc_entropy = 0.0
    n = float(len(pclouds))
    for g in grid_bernoulli_rvars:
        if g > 0:
            p = float(g) / n
            acc_entropy += entropy([p, 1.0 - p])

    return acc_entropy / len(grid_counters), grid_counters


def jensen_shannon_divergence(p_big, q_big):
    if np.any(p_big < 0) or np.any(q_big < 0):
        raise ValueError('Negative values.')
    if len(p_big) != len(q_big):
        raise ValueError('Non equal size.')

    p_ = p_big / np.sum(p_big)  # Ensure probabilities.
    q_ = q_big / np.sum(q_big)

    e1 = entropy(p_, base=2)
    e2 = entropy(q_, base=2)
    e_sum = entropy((p_ + q_) / 2.0, base=2)
    res = e_sum - ((e1 + e2) / 2.0)

    res2 = _jsdiv(p_, q_)

    if not np.allclose(res, res2, atol=10e-5, rtol=0):
        warnings.warn('Numerical values of two JSD methods don\'t agree.')

    return res


def _jsdiv(p_big, q_big):
    """another way of computing JSD"""

    def _kldiv(a_big, b_big):
        a = a_big.copy()
        b = b_big.copy()
        idx = np.logical_and(a > 0, b > 0)
        a = a[idx]
        b = b[idx]
        return np.sum([v for v in a * np.log2(a / b)])

    p_ = p_big / np.sum(p_big)
    q_ = q_big / np.sum(q_big)

    mat = 0.5 * (p_ + q_)

    return 0.5 * (_kldiv(p_, mat) + _kldiv(q_, mat))
