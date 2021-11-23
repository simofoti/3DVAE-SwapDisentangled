#!/bin/bash

virtualenv -p python3 ./id-generator-env
source ./id-generator-env/bin/activate

export CUDA=cu101

pip install cmake
pip install trimesh pyrender tqdm matplotlib rtree openmesh tb-nightly av seaborn

pip install torch==1.7.1+${CUDA} torchvision==0.8.2+${CUDA} -f https://download.pytorch.org/whl/torch_stable.html

# install pytorch geometric
export TORCH=1.7.1
pip install torch-scatter==2.0.6 -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse==0.6.9 -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster==1.5.9 -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric==$TORCH

export PYTHON_V=38
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py${PYTHON_V}_${CUDA}_pyt171/download.html
pip install geomloss pykeops
