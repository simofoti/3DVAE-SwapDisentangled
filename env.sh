#!/bin/bash

virtualenv -p python3 ./id-generator-env
source ./id-generator-env/bin/activate

export CUDA=cu101

pip install trimesh pyrender tqdm matplotlib rtree

pip install torch==1.7.1+${CUDA} torchvision==0.8.2+${CUDA} -f https://download.pytorch.org/whl/torch_stable.html

# install pytorch geometric
export TORCH=1.7.0  # if torch 1.7.0 or 1.7.1
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric

export PYTHON_V=38
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py${PYTHON_V}_${CUDA}_pyt171/download.html
