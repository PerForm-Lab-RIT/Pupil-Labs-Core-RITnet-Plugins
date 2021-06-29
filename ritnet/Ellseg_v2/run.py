#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 15:39:49 2021

@author: rakshit
"""
import os
import torch
import random
import warnings
import numpy as np

from datetime import datetime
from distutils.dir_util import copy_tree

from main import train
from args_maker import make_args

# Suppress warnings
warnings.filterwarnings('ignore')


def create_experiment_folder_tree(repo_root,
                                  path_exp_records,
                                  exp_name,
                                  is_test=False,
                                  create_tree=True):

    if is_test:
        exp_name_str = exp_name
    else:
        now = datetime.now()
        date_time_str = now.strftime('%d_%m_%y_%H_%M_%S')
        exp_name_str = exp_name + '_' + date_time_str

    path_exp = os.path.join(path_exp_records, exp_name_str)

    path_dict = {}
    for ele in ['results', 'figures', 'logs', 'src']:
        path_dict[ele] = os.path.join(path_exp, ele)
        os.makedirs(path_dict[ele], exist_ok=True)

    path_dict['exp'] = path_exp

    if (not is_test) and create_tree:
        # Do not copy data because in test only condition, this folder would
        # already be populated
        copy_tree(repo_root, os.path.join(path_exp, 'src'))
    return path_dict


def cleanup():
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    args = vars(make_args())

    path_dict = create_experiment_folder_tree(args['repo_root'],
                                              args['path_exp_tree'],
                                              args['exp_name'],
                                              args['only_test'],
                                              create_tree=args['local_rank']==0 if args['do_distributed'] else True)

    path_dict['repo_root'] = args['repo_root']
    path_dict['path_data'] = args['path_data']

    # %% DDP essentials

    if args['do_distributed']:
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        world_size = torch.distributed.get_world_size()

    else:
        world_size = 1

    global batch_size_global
    batch_size_global = int(args['batch_size']*world_size)

    torch.cuda.set_device(args['local_rank'])
    args['world_size'] = world_size
    args['batch_size'] = int(args['batch_size']/world_size)

    # %%
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # Set seeds
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    random.seed(args['seed'])

    # Train and save validated model
    if not args['only_test']:
        train(args, path_dict, test_mode=False)
    # Close process group
    if args['do_distributed']:
        torch.distributed.barrier()
        cleanup()
    else:
        # Test out best model and save results
        train(args, path_dict, test_mode=True)
