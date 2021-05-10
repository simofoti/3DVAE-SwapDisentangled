import os
import argparse
import shutil
import tqdm
import torch.nn
from torch.utils.tensorboard import SummaryWriter

import utils
from data_generation_and_loading import FaceGenerator, BodyGenerator
from data_generation_and_loading import get_data_loaders
from model_manager import ModelManager
from test import Tester

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configurations/default.yaml',
                    help="Path to the configuration file.")
parser.add_argument('--id', type=str, default='none', help="ID of experiment")
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument('--generate_data', action='store_true')
parser.add_argument('--resume', action='store_true')
opts = parser.parse_args()
config = utils.get_config(opts.config)

if opts.id != 'none':
    model_name = opts.id
else:
    model_name = os.path.splitext(os.path.basename(opts.config))[0]
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_dir = utils.prepare_sub_folder(output_directory)

writer = SummaryWriter(output_directory + '/logs')
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))

if not torch.cuda.is_available():
    device = torch.device('cpu')
    print("GPU not available, running on CPU")
else:
    device = torch.device('cuda')

if opts.generate_data:
    if config['data']['dataset_type'] == 'faces':
        data_generator = FaceGenerator(config['data']['pca_path'],
                                       config['data']['dataset_path'])
    else:
        data_generator = BodyGenerator(config['data']['dataset_path'])
    data_generator(config['data']['number_of_meshes'],
                   config['data']['std_pca_latent'], opts.generate_data)

manager = ModelManager(configurations=config, device=device)

train_loader, validation_loader, test_loader, normalization_dict = \
    get_data_loaders(config, manager.template)

train_visualization_batch = next(iter(train_loader))
validation_visualization_batch = next(iter(validation_loader))

# manager.render_and_show_batch(train_visualization_batch, normalization_dict)

if opts.resume:
    start_epoch = manager.resume(checkpoint_dir)
else:
    start_epoch = 0

for epoch in tqdm.tqdm(range(start_epoch, config['optimization']['epochs'])):
    manager.run_epoch(train_loader, device, train=True)
    manager.log_losses(writer, epoch, 'train')

    manager.run_epoch(validation_loader, device, train=False)
    manager.log_losses(writer, epoch, 'validation')

    if (epoch + 1) % config['logging_frequency']['tb_renderings'] == 0:
        manager.log_images(train_visualization_batch, writer, epoch,
                           normalization_dict, 'train', error_max_scale=2)
        manager.log_images(validation_visualization_batch, writer, epoch,
                           normalization_dict, 'validation', error_max_scale=2)
    if (epoch + 1) % config['logging_frequency']['save_weights'] == 0:
        manager.save_weights(checkpoint_dir, epoch)

Tester(manager, normalization_dict, train_loader, test_loader,
       output_directory, config)()
