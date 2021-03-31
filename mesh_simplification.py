import heapq
import math
import time
import trimesh
import trimesh.proximity
import torch
import torch_geometric
from torch_geometric.data import Data

import numpy as np
import scipy.sparse as sp

import utils


class MeshSimplifier:
    def __init__(self, in_mesh_path=None, in_mesh=None, debug=False):
        if in_mesh_path:
            self._in_mesh = utils.load_template(in_mesh_path)
        elif in_mesh:
            self._in_mesh = in_mesh
        else:
            raise AttributeError("in_mesh_path or in_mesh needed")
        self._debug = debug
        self._quadrics = self._vertex_quadrics()

    def __call__(self, sampling_factor, region_weighted=False,
                 edge_length_weighted=False):
        sampled, down_mat = self.quadric_edge_collapse(
            sampling_factor, region_weighted, edge_length_weighted)
        up_mat = self._get_upsampling_transformation(sampled)
        return sampled, utils.to_torch_sparse(down_mat), \
            utils.to_torch_sparse(up_mat)

    @property
    def in_mesh(self):
        return self._in_mesh

    @property
    def quadrics(self):
        return self._quadrics

    def quadric_edge_collapse(self, sampling_factor, region_weighted=False,
                              edge_length_weighted=False):
        desired_verts_number = \
            math.ceil(self._in_mesh.pos.shape[0] / sampling_factor)
        edges = self._in_mesh.edge_index.t()
        edges = edges[edges[:, 0] < edges[:, 1]].numpy()

        region_weights = None
        if region_weighted:
            rw = {k: 1 / (len(fc['feature']) + len(fc['contour']))
                  for k, fc in self._in_mesh.feat_and_cont.items()}
            fc = self._in_mesh.feat_and_cont
            region_weights = np.ones(self._in_mesh.pos.shape[0])
            for key, w in rw.items():
                feat_and_cont = fc[key]['feature']
                feat_and_cont.extend(fc[key]['contour'])
                region_weights[feat_and_cont] = w

        faces = self._quadric_edge_collapse(
            edges, desired_verts_number, region_weights=region_weights,
            edge_length_weighted=edge_length_weighted)

        new_faces, downsampling_matrix = self._get_new_faces_and_downsampling(
            faces, self._in_mesh.num_nodes)
        new_mesh = self._get_sampled_mesh(downsampling_matrix, new_faces)
        new_mesh.feat_and_cont = \
            utils.extract_feature_and_contour_from_colour(new_mesh)
        return new_mesh, downsampling_matrix

    def _quadric_edge_collapse(self, edges, desired_verts_number,
                               region_weights=None, edge_length_weighted=False):
        # 24.27s vs 1167.70s before (sampling factor 0.8)
        verts_number = self._in_mesh.pos.shape[0]
        faces = self._in_mesh.face.clone().numpy()

        h = []
        for e_idx, e in enumerate(edges):
            cost = self._edge_collapse_cost(e, region_weights,
                                            edge_length_weighted)
            h.append((cost['collapse_cost'], e_idx))
        heapq.heapify(h)

        while verts_number > desired_verts_number:  # 0.0007076s vs 0.06s before
            top_elem_cost, top_elem_edge_index = heapq.heappop(h)

            if edges[top_elem_edge_index][0] == edges[top_elem_edge_index][1]:
                continue

            current_cost = self._edge_collapse_cost(
                [edges[top_elem_edge_index][0], edges[top_elem_edge_index][1]],
                region_weights, edge_length_weighted)
            if current_cost['collapse_cost'] > top_elem_cost:
                # if the cost was wrong, put back in heap with correct cost
                heapq.heappush(h, (current_cost['collapse_cost'],
                                   top_elem_edge_index))
            else:
                if current_cost['destroy_0_cost'] < \
                        current_cost['destroy_1_cost']:
                    to_destroy = edges[top_elem_edge_index][1]
                    to_keep = edges[top_elem_edge_index][0]
                else:
                    to_destroy = edges[top_elem_edge_index][0]
                    to_keep = edges[top_elem_edge_index][1]

                np.place(faces, faces == to_destroy, to_keep)
                np.place(edges, edges == to_destroy, to_keep)

                self._quadrics[to_keep] = current_cost['quadric_sum']
                self._quadrics[to_destroy] = current_cost['quadric_sum']
                verts_number -= 1

        # remove degenerate faces
        a = faces[0, :] == faces[1, :]
        b = faces[1, :] == faces[2, :]
        c = faces[2, :] == faces[0, :]
        faces_to_keep = np.logical_not(self.logical_or3(a, b, c))
        faces = faces[:, faces_to_keep].copy()
        return faces

    def _vertex_quadrics(self):  # 7.7s
        """Computes a quadric for each vertex in the Mesh.
        Returns:
           v_quadrics: an (N x 4 x 4) array, where N is # vertices.
        """
        v_quadrics = np.zeros((self._in_mesh.num_nodes, 4, 4))
        pos = self._in_mesh.pos.cpu().numpy()
        face = self._in_mesh.face.cpu().numpy()
        for f_idx in range(self._in_mesh.num_faces):
            vert_idxs = face[:, f_idx]
            verts = np.hstack((pos[vert_idxs],
                               np.array([1, 1, 1]).reshape(-1, 1)))
            u, s, v = np.linalg.svd(verts)
            eq = v[-1, :].reshape(-1, 1)
            eq = eq / (np.linalg.norm(eq[0:3]))
            # Add the outer product of the plane equation to the
            # quadrics of the vertices for this face
            for k in range(3):
                v_quadrics[face[k, f_idx], :, :] += np.outer(eq, eq)
        return v_quadrics

    def _edge_collapse_cost(self, edge, region_w=None, length_w=False):
        # 0.00012s
        vertices = self._in_mesh.pos.cpu().numpy()
        v0_id, v1_id = edge[0], edge[1]
        quadrics_sum = self._quadrics[v0_id, :, :] + self._quadrics[v1_id, :, :]
        p0 = np.vstack((vertices[v0_id].reshape(-1, 1),
                        np.array([1]).reshape(-1, 1)))
        p1 = np.vstack((vertices[v1_id].reshape(-1, 1),
                        np.array([1]).reshape(-1, 1)))

        destroy_0_cost = p0.T.dot(quadrics_sum).dot(p0).item()
        destroy_1_cost = p1.T.dot(quadrics_sum).dot(p1).item()
        collapse_cost = min([destroy_0_cost, destroy_1_cost])

        if length_w:
            collapse_cost += np.linalg.norm(vertices[v0_id] - vertices[v1_id])
        if region_w is not None:
            collapse_cost *= (region_w[v0_id] + region_w[v1_id]) / 2

        return {
            'destroy_0_cost': destroy_0_cost,
            'destroy_1_cost': destroy_1_cost,
            'collapse_cost': collapse_cost,
            'quadric_sum': quadrics_sum
        }

    @staticmethod
    def logical_or3(x, y, z):
        return np.logical_or(x, np.logical_or(y, z))

    @staticmethod
    def _get_new_faces_and_downsampling(faces, num_original_verts):
        faces = faces.T
        verts_left = np.unique(faces.flatten())
        i_s = np.arange(len(verts_left))
        j_s = verts_left
        data = np.ones(len(j_s))

        mp = np.arange(0, np.max(faces.flatten()) + 1)
        mp[j_s] = i_s
        new_faces = mp[faces.copy().flatten()].reshape((-1, 3))

        ij = np.vstack((i_s.flatten(), j_s.flatten()))
        sampling_mat = sp.csc_matrix(
            (data, ij), shape=(len(verts_left), num_original_verts))
        return new_faces, sampling_mat

    @staticmethod
    def compute_sparse_adjacency(data):
        size = [data.num_nodes] * 2
        edge_attr = torch.ones(data.edge_index.size(1), dtype=torch.float)
        data.adj = torch.sparse_coo_tensor(data.edge_index, edge_attr, size)
        return data

    def _get_sampled_mesh(self, downsampling_matrix, new_faces):
        mesh_verts = torch.tensor(
            downsampling_matrix.dot(self._in_mesh.pos.numpy()),
            dtype=torch.float, requires_grad=False)
        mesh_colors = torch.tensor(
            downsampling_matrix.dot(self._in_mesh.colors.numpy()),
            requires_grad=False
        )
        face = torch.from_numpy(new_faces).t().to(torch.long).contiguous()
        data = Data(pos=mesh_verts, face=face, colors=mesh_colors)
        data = torch_geometric.transforms.FaceToEdge(False)(data)
        if self._debug:
            trim = torch_geometric.utils.to_trimesh(data)
            trim.visual.vertex_colors = mesh_colors.numpy()
            trim.show()
        return data

    def _get_upsampling_transformation(self, sampled_mesh):
        trim_sampled = torch_geometric.utils.to_trimesh(sampled_mesh)
        _, _, closest_face_ids = trimesh.proximity.closest_point(
            trim_sampled, self._in_mesh.pos.numpy())
        rows, cols, coeffs = [], [], []
        for i, (pt, f_idx) in enumerate(zip(self._in_mesh.pos.numpy(),
                                            closest_face_ids)):
            triangle_idxs = list(sampled_mesh.face[:, f_idx].numpy())
            triangle_verts = sampled_mesh.pos[triangle_idxs].numpy()

            # from W. Heidrich et al. 2005
            u = triangle_verts[1, :] - triangle_verts[0, :]
            v = triangle_verts[2, :] - triangle_verts[0, :]
            n = np.cross(u, v)
            w = pt - triangle_verts[0, :]
            gamma = np.dot(np.cross(u, w), n) / np.dot(n, n)
            beta = np.dot(np.cross(w, v), n) / np.dot(n, n)
            alpha = 1 - gamma - beta
            rows.extend([i] * 3)
            cols.extend(triangle_idxs)
            coeffs.extend([alpha, beta, gamma])
            # Project pt in triangle
            # pt_projected = alpha * triangle_verts[0, :] + \
            #     beta * triangle_verts[1, :] + gamma * triangle_verts[2, :]
        matrix = sp.csc_matrix(
            (coeffs, (rows, cols)),
            shape=(self._in_mesh.num_nodes, sampled_mesh.num_nodes))

        if self._debug:
            upsampled_verts = matrix.dot(sampled_mesh.pos.numpy())
            faces = self._in_mesh.face.numpy().T
            trim = trimesh.Trimesh(upsampled_verts, faces)
            trim.show()
        return matrix


if __name__ == '__main__':
    # NB trimesh wraps open3d to get Quadric sampling, but it is implemented in
    # cuda and it would be difficult to hack
    # mesh = trimesh.load_mesh('UHM_models/mean_nme_fcolor_b.ply', 'ply',
    #                          process=False)
    # mesh = mesh.simplify_quadratic_decimation(mesh.faces.shape[0] / 100)
    t = time.time()
    simplifier = MeshSimplifier('UHM_models/mean_nme_fcolor_b.ply', debug=True)
    m, down, up = simplifier(10, region_weighted=True,
                             edge_length_weighted=False)
    print(time.time() - t)
