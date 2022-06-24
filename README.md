# 3D Shape Variational Autoencoder Latent Disentanglement via Mini-Batch Feature Swapping for Bodies and Faces
[[arxiv]](https://arxiv.org/pdf/2111.12448.pdf) [[CVF]](https://openaccess.thecvf.com/content/CVPR2022/papers/Foti_3D_Shape_Variational_Autoencoder_Latent_Disentanglement_via_Mini-Batch_Feature_Swapping_CVPR_2022_paper.pdf) 

## Installation

After cloning the repo open a terminal and go to the project directory. 

Change the permissions of install_env.sh by running `chmod +x ./install_env.sh` 
and run it with:
```shell script
./install_env.sh
```
This will create a virtual environment with all the necessary libraries.

Note that it was tested with Python 3.8, CUDA 10.1, and Pytorch 1.7.1. The code 
should work also with newer versions of  Python, CUDA, and Pytorch. If you wish 
to try running the code with more recent versions of these libraries, change the 
CUDA, TORCH, and PYTHON_V variables in install_env.sh

Then activate the virtual environment :
```shell script
source ./id-generator-env/bin/activate
```


## Datasets

To obtain access to the UHM models and generate the dataset, please follow the 
instructions on the 
[github repo of UHM](https://github.com/steliosploumpis/Universal_Head_3DMM).

 Data will be automatically generated from the UHM during the first training. 
 In this case the training must be launched with the argument `--generate_data` 
 (see below).
 
 ## Prepare Your Configuration File
 
 We made available a configuration file for each experiment (default.yaml is 
 the configuration file of the proposed method). Make sure 
 the paths in the config file are correct. In particular, you might have to 
 change `pca_path` according to the location where UHM was downloaded.
 
 ## Train and Test
 
 To start the training from the project repo simply run:
 ```shell script
python train.py --config=configurations/<A_CONFIG_FILE>.yaml --id=<NAME_OF_YOUR_EXPERIMENT>
```

If this is your first training and you wish to generate the data, run:
```shell script
python train.py --generate_data --config=configurations/<A_CONFIG_FILE>.yaml --id=<NAME_OF_YOUR_EXPERIMENT>
``` 

Basic tests will automatically run at the end of the training. If you wish to 
run additional tests presented in the paper you can uncomment any function call 
at the end of `test.py`. If your model has alredy been trained or you are using 
our pretrained model, you can run tests without training:
```shell script
python test.py --id=<NAME_OF_YOUR_EXPERIMENT>
```
Note that NAME_OF_YOUR_EXPERIMENT is also the name of the folder containing the
pretrained model.

## Additional Notes
We make available the files storing:
 - the precomputed down- and up-sampling transformation
 - the precomputed spirals
 - the mesh template with the face regions
 - the network weights
