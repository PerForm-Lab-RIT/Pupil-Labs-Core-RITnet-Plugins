#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 18:02:16 2021

@author: rakshit
"""

import os
import torch
import pickle
import logging

import torch.nn as nn
import torch.nn.functional as F

from models_mux import model_dict
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from scripts import forward
from helperfunctions.helperfunctions import mod_scalar
from helperfunctions.utils import EarlyStopping, make_logger
from helperfunctions.utils import SpikeDetection, get_nparams
from helperfunctions.utils import move_to_single, FRN_TLU, do_nothing


def train(args, path_dict, test_mode=False):

    rank_cond = (args['local_rank'] == 0) or not args['do_distributed']

    # %% Load model
    if args['use_frn_tlu']:
        net = model_dict[args['model']](args,
                                        norm=FRN_TLU,
                                        act_func=do_nothing)
    elif args['use_instance_norm']:
        net = model_dict[args['model']](args,
                                        norm=nn.InstanceNorm2d,
                                        act_func=F.leaky_relu)
    elif args['use_group_norm']:
        net = model_dict[args['model']](args,
                                        norm='group_norm',
                                        act_func=F.leaky_relu)
    elif args['use_ada_instance_norm']:
        net = model_dict[args['model']](args,
                                        norm='ada_instance_norm',
                                        act_func=F.leaky_relu)
    else:
        net = model_dict[args['model']](args,
                                        norm=nn.BatchNorm2d,
                                        act_func=F.leaky_relu)

    # %% Weight loaders
    # if it is pretrained, then load pretrained weights
    if args['pretrained'] or args['continue_training']:

        if args['pretrained']:
            path_pretrained = os.path.join(path_dict['repo_root'],
                                           '..',
                                           'pretrained',
                                           'pretrained.git_ok')

        if args['continue_training']:
            path_pretrained = os.path.join(args['continue_training'])

        net_dict = torch.load(path_pretrained,
                              map_location=torch.device('cpu'))
        state_dict_single = move_to_single(net_dict['state_dict'])
        net.load_state_dict(state_dict_single, strict=False)
        print('Pretrained model loaded')

    if test_mode:
        print('Test mode detected. Loading best model.')
        if args['path_model']:
            net_dict = torch.load(args['path_model'],
                                  map_location=torch.device('cpu'))
        else:
            net_dict = torch.load(os.path.join(path_dict['results'],
                                               'best_model.pt'),
                                  map_location=torch.device('cpu'))
        state_dict_single = move_to_single(net_dict['state_dict'])
        net.load_state_dict(state_dict_single, strict=False)

        # Do not initialize a writer
        writer = []
    else:
        # Initialize tensorboard if rank 0
        if rank_cond:
            writer = SummaryWriter(path_dict['logs'])
        else:
            writer = []

    if args['use_GPU']:
        net.cuda()

    # %% move network to DDP
    if args['do_distributed']:
        # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = DDP(net,
                  device_ids=[args['local_rank']],
                  find_unused_parameters=True)

    # %% Initialize logger
    logger = make_logger(path_dict['logs']+'/train_log.log',
                         rank=args['local_rank'] if args['do_distributed'] else 0)
    logger.write_summary(str(net.parameters))
    logger.write('# of parameters: {}'.format(get_nparams(net)))
    logger.write('Training!')

    # %% Training and validation loops or test only
    train_validation_loops(net,
                           logger,
                           args,
                           path_dict,
                           writer,
                           test_mode)

    # %% Closing functions and logging
    if writer:
        writer.close()


def train_validation_loops(net, logger, args,
                           path_dict, writer, test_mode):

    # %% Load curriculum objects
    path_cur_obj = os.path.join(path_dict['repo_root'],
                                'cur_objs',
                                args['mode'],
                                'cond_'+args['cur_obj']+'.pkl')

    with open(path_cur_obj, 'rb') as f:
        train_obj, valid_obj, test_obj = pickle.load(f)

    # %% Specify flags of importance
    train_obj.augFlag = args['aug_flag']
    valid_obj.augFlag = False
    test_obj.augFlag = False

    train_obj.equi_var = args['equi_var']
    valid_obj.equi_var = args['equi_var']
    test_obj.equi_var = args['equi_var']

    # %% Modify path information
    train_obj.path2data = path_dict['path_data']
    valid_obj.path2data = path_dict['path_data']
    test_obj.path2data = path_dict['path_data']

    # %% Create distributed samplers
    train_sampler = DistributedSampler(train_obj,
                                       rank=args['local_rank'],
                                       shuffle=False,
                                       num_replicas=args['world_size'],
                                       )

    valid_sampler = DistributedSampler(valid_obj,
                                       rank=args['local_rank'],
                                       shuffle=False,
                                       num_replicas=args['world_size'],
                                       )

    test_sampler = DistributedSampler(test_obj,
                                      rank=args['local_rank'],
                                      shuffle=False,
                                      num_replicas=args['world_size'],
                                      )

    # %% Define dataloaders
    logger.write('Initializing loaders')
    if not test_mode:
        train_loader = DataLoader(train_obj,
                                  shuffle=False,
                                  num_workers=args['workers'],
                                  drop_last=True,
                                  pin_memory=True,
                                  batch_size=args['batch_size'],
                                  sampler=train_sampler if args['do_distributed'] else None,
                                  )

        valid_loader = DataLoader(valid_obj,
                                  shuffle=False,
                                  num_workers=args['workers'],
                                  drop_last=True,
                                  pin_memory=True,
                                  batch_size=args['batch_size'],
                                  sampler=valid_sampler if args['do_distributed'] else None,
                                  )

    test_loader = DataLoader(test_obj,
                             shuffle=False,
                             num_workers=0,
                             drop_last=True,
                             batch_size=args['batch_size'],
                             sampler=test_sampler if args['do_distributed'] else None,
                             )

    # %% Early stopping criterion
    early_stop = EarlyStopping(patience=args['early_stop'],
                               verbose=True,
                               delta=0.001,  # 0.1% improvement needed
                               rank_cond=args['local_rank'] == 0 if args['do_distributed'] else True,
                               mode='max',
                               fName='best_model.pt',
                               path_save=path_dict['results'],
                               )

    # %% Define alpha and beta scalars
    if args['curr_learn_losses']:
        alpha_scalar = mod_scalar([0, args['epochs']], [0, 1])
        beta_scalar = mod_scalar([10, 20], [0, 1])

    # %% Optimizer
    param_list = [param for name, param in net.named_parameters() if 'adv' not in name]
    optimizer = torch.optim.Adam(param_list,
                                 lr=args['lr'], amsgrad=False)

    if args['adv_DG']:
        param_list = [param for name, param in net.named_parameters() if 'adv' in name]
        optimizer_disc = torch.optim.Adam(param_list,
                                          lr=args['lr'],
                                          amsgrad=True)
    else:
        optimizer_disc = False

    # %% Loops and what not

    # Create a checkpoint based on current scores
    checkpoint = {}
    checkpoint['args'] = args  # Save arguments

    # Randomize the dataset again the next time you exit
    # to the main loop.
    args['time_to_update'] = True

    if test_mode:

        logging.info('Entering test only mode ...')
        args['alpha'] = 0.5
        args['beta'] = 0.5

        test_result = forward(net,
                              [],
                              logger,
                              test_loader,
                              optimizer,
                              args,
                              path_dict,
                              writer=writer,
                              epoch=0,
                              mode='test',
                              batches_per_ep=len(test_loader))

        checkpoint['test_result'] = test_result
        if args['save_results_here']:

            # Ensure the directory exists
            os.makedirs(os.path.dirname(args['save_results_here']), exist_ok=True)

            # Save out test results here instead
            with open(args['save_results_here'], 'wb') as f:
                pickle.dump(checkpoint, f)
        else:

            # Ensure the directory exists
            os.makedirs(path_dict['results'], exist_ok=True)

            # Save out the test results
            with open(path_dict['results'] + '/test_results.pkl', 'wb') as f:
                pickle.dump(checkpoint, f)

    else:
        spiker = SpikeDetection() if args['remove_spikes'] else False
        logging.info('Entering train mode ...')

        epoch = 0

        # Disable early stop and keep training until it maxes out, this allows
        # us to test at the regular best model while saving intermediate result
        #while (epoch < args['epochs']) and not early_stop.early_stop:

        while (epoch < args['epochs']):
            if args['time_to_update']:

                # Toggle flag back to False
                args['time_to_update'] = False

                if args['one_by_one_ds']:
                    train_loader.dataset.sort('one_by_one_ds', args['batch_size'])
                    valid_loader.dataset.sort('one_by_one_ds', args['batch_size'])
                else:
                    train_loader.dataset.sort('mutliset_random')
                    valid_loader.dataset.sort('mutliset_random')

            # Set epochs for samplers
            train_sampler.set_epoch(epoch)
            valid_sampler.set_epoch(epoch)

            # %%
            logging.info('Starting epoch: %d' % epoch)

            if args['curr_learn_losses']:
                args['alpha'] = alpha_scalar.get_scalar(epoch)
                args['beta'] = beta_scalar.get_scalar(epoch)
            else:
                args['alpha'] = 0.5
                args['beta'] = 0.5

            if args['dry_run']:
                train_batches_per_ep = len(train_loader)
                valid_batches_per_ep = len(valid_loader)
            else:
                train_batches_per_ep = args['batches_per_ep']
                if args['reduce_valid_samples']:
                    valid_batches_per_ep = args['reduce_valid_samples']
                else:
                    valid_batches_per_ep = len(valid_loader)

            train_result = forward(net,
                                   spiker,
                                   logger,
                                   train_loader,
                                   optimizer,
                                   args,
                                   path_dict,
                                   optimizer_disc=optimizer_disc,
                                   writer=writer,
                                   epoch=epoch,
                                   mode='train',
                                   batches_per_ep=train_batches_per_ep)

            if args['reduce_valid_samples']:
                valid_result = forward(net,
                                       spiker,
                                       logger,
                                       valid_loader,
                                       optimizer,
                                       args,
                                       path_dict,
                                       writer=writer,
                                       epoch=epoch,
                                       mode='valid',
                                       batches_per_ep=valid_batches_per_ep)
            else:
                valid_result = forward(net,
                                       spiker,
                                       logger,
                                       valid_loader,
                                       optimizer,
                                       args,
                                       path_dict,
                                       writer=writer,
                                       epoch=epoch,
                                       mode='valid',
                                       batches_per_ep=len(valid_loader))

            # Update the check point weights. VERY IMPORTANT!
            checkpoint['state_dict'] = move_to_single(net.state_dict())

            checkpoint['epoch'] = epoch
            checkpoint['train_result'] = train_result
            checkpoint['valid_result'] = valid_result

            # Save out the best validation result and model
            early_stop(checkpoint)

            # If epoch is a multiple of args['save_every'], then write out
            if (epoch%args['save_every']) == 0:

                # Ensure that you do not update the validation score at this
                # point and simply save the model
                early_stop.save_checkpoint(checkpoint['valid_result']['score_mean'],
                                           checkpoint,
                                           update_val_score=False,
                                           use_this_name_instead='{}.pt'.format(epoch))

            epoch += 1


if __name__ == '__main__':
    print('Entry script is run.py')
