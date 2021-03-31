import os
import argparse
import shutil
import tqdm
import torch.nn
from torch.utils.tensorboard import SummaryWriter

import utils
from model_manager import ModelManager

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configurations/default.yaml',
                    help="Path to the configuration file.")
parser.add_argument('--id', type=str, default='none', help="ID of experiment")
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
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

manager = ModelManager(configurations=config, device=device)
