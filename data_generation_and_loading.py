import os
import pickle
import tqdm
import trimesh

import numpy as np


class DataGenerator:
    def __init__(self, uhm_path, data_dir='./data'):
        infile = open(uhm_path, 'rb')
        uhm_dict = pickle.load(infile)
        infile.close()

        self._data_dir = data_dir
        self._components = uhm_dict['Eigenvectors'].shape[1]
        self._mu = uhm_dict['Mean']
        self._eigenvectors = uhm_dict['Eigenvectors']
        self._eigenvalues = uhm_dict['EigenValues']

    def generate_random_face(self, weight):
        w = weight * np.random.normal(size=self._components) * \
            self._eigenvalues ** 0.5
        w = np.expand_dims(w, axis=1)
        vertices = self._mu + self._eigenvectors @ w
        return vertices.reshape(-1, 3)

    def save_vertices(self, vertices, name):
        m = trimesh.Trimesh(vertices, process=False)
        m.export(os.path.join(self._data_dir, name + '.ply'))

    def __call__(self, number_of_meshes, weight=1., overwrite_data=False):
        if not os.path.isdir(self._data_dir):
            os.mkdir(self._data_dir)

        if not os.listdir(self._data_dir) or overwrite_data:  # directory empty
            print("Generating Data from PCA")
            for i in tqdm.tqdm(range(number_of_meshes)):
                v = self.generate_random_face(weight)
                self.save_vertices(v, str(i))

