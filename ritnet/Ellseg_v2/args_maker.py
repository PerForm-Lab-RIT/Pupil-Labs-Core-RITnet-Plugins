#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 13:30:38 2021

@author: rakshit
"""
from pprint import pprint
import argparse


def make_args():

    default_repo = '/home/rsk3900/Documents/Python_Scripts/multiset_gaze/src'

    parser = argparse.ArgumentParser()

    # %% Hyperparams
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='base learning rate')
    parser.add_argument('--seed', type=int, default=108,
                        help='seed value for all packages')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size for training')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank to set GPU')
    parser.add_argument('--lr_decay', type=int, default=0,
                        help='learning rate decay')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout? anything above 0 activates dropout')

    # %% Model specific parameters
    parser.add_argument('--base_channel_size', type=int, default=32,
                        help='base channel size around which model grows')
    parser.add_argument('--growth_rate', type=float, default=1.2,
                        help='growth rate of channels in network')
    parser.add_argument('--track_running_stats', type=int, default=0,
                        help='disable running stats for better transfer')
    parser.add_argument('--extra_depth', type=int, default=0,
                        help='extra convolutions to the encoder')
    parser.add_argument('--grad_rev', type=int, default=0,
                        help='gradient reversal for dataset identity')
    parser.add_argument('--adv_DG', type=int, default=0,
                        help='enable discriminator')
    parser.add_argument('--equi_var', type=int, default=0,
                        help='normalize data to respect image dimensions')
    parser.add_argument('--num_blocks', type=int, default=4,
                        help='number of encoder decoder blocks')
    parser.add_argument('--use_frn_tlu', type=int, default=0,
                        help='replace BN+L.RELU with FRN+TLU')
    parser.add_argument('--use_instance_norm', type=int, default=0,
                        help='replace BN with IN')
    parser.add_argument('--use_group_norm', type=int, default=0,
                        help='replace BN with GN, 8 channels per group')
    parser.add_argument('--use_ada_instance_norm', type=int, default=0,
                        help='use adaptive instance normalization')

    # %% Discriminator stats
    parser.add_argument('--disc_base_channel_size', type=int, default=8,
                        help='discriminator base channels?')
    '''
    Notes on discriminator channel size.
    Rakshit - from my experiments, I find that the discriminator easily learns
    the domains apart and adding more channels simply eats up the memory
    without contributing any more discriminative power. Hence, I leave this at
    a small value.
    '''

    # %% Experiment parameters
    parser.add_argument('--path_exp_tree', type=str,
                        default='/results/multiset_results',
                        help='path to all experiments result folder')
    parser.add_argument('--path_data', type=str,
                        default='/data/datasets/All',
                        help='path to all H5 file data')
    parser.add_argument('--path_model', type=str, default=[],
                        help='path to model for test purposes')
    parser.add_argument('--repo_root', type=str,
                        default=default_repo,
                        help='path to repo root')
    parser.add_argument('--exp_name', type=str, default='dev',
                        help='experiment string or identifier')
    parser.add_argument('--reduce_valid_samples', type=int, default=400,
                        help='reduce the number of\
                            validaton samples to speed up')
    parser.add_argument('--save_every', type=int, default=5,
                        help='save weights every 5 iterations')

    # %% Train or test parameters
    parser.add_argument('--mode', type=str, default='one_vs_one',
                        help='training mode:\
                            one_vs_one, all_vs_one, all-one_vs_one')
    parser.add_argument('--epochs', type=int, default=80,
                        help='number of training epochs')
    parser.add_argument('--cur_obj', type=str, default='OpenEDS',
                        help='which dataset to train on or remove?\
                            in all_vs_one, this flag does nothing')
    parser.add_argument('--only_test', type=str, default=0,
                        help='test only mode')
    parser.add_argument('--aug_flag', type=int, default=0,
                        help='enable augmentations?')
    parser.add_argument('--one_by_one_ds', type=int, default=0,
                        help='train on a single dataset, one after the other')
    parser.add_argument('--early_stop', type=int, default=20,
                        help='early stop epoch count')
    parser.add_argument('--mixed_precision', type=int, default=0,
                        help='enable mixed precision training and testing')
    parser.add_argument('--batches_per_ep', type=int, default=2000,
                        help='number of batches per training epoch')
    parser.add_argument('--use_GPU', type=int, default=1,
                        help='train on GPU?')
    parser.add_argument('--remove_spikes', type=int, default=1,
                        help='remove noisy batches for smooth training')
    parser.add_argument('--pseudo_labels', type=int, default=0,
                        help='generate pseudo labels on datasets with missing\
                            labels')

    # %% Model specific parameters
    parser.add_argument('--use_scSE', type=int, default=0,
                        help='use concurrent spatial and channel excitation \
                        at the end of every encoder or decoder block')
    parser.add_argument('--make_aleatoric', type=int, default=0,
                        help='add aleatoric formulation\
                            during latent regression')
    parser.add_argument('--make_uncertain', type=int, default=0,
                        help='activate aleatoric and epistemic')
    parser.add_argument('--continue_training', type=str, default='',
                        help='continue training from these weights')
    parser.add_argument('--regression_from_latent', type=int, default=1,
                        help='disable regression from the latent space?')
    parser.add_argument('--curr_learn_losses', type=int, default=1,
                        help='add two rampers and use them as you see fit')
    parser.add_argument('--regress_channel_grow', type=float, default=0,
                        help='grow channels in the regression module. Default\
                            0 means the channel size stays the same.')
    parser.add_argument('--maxpool_in_regress_mod', type=int, default=-1,
                        help='replace avg pool with max pooling in regression\
                            module, if -1, then pooling is disabled')
    parser.add_argument('--dilation_in_regress_mod', type=int, default=1,
                        help='enable dilation in the regression module')

    # %% Model selection
    parser.add_argument('--model', type=str, default='DenseElNet',
                        help='DenseElNet, RITNet')

    # %% Pretrained conditions
    parser.add_argument('--pretrained', type=int, default=0,
                        help='load weights from model\
                            pretrained on full datasets')

    # %% General parameters
    parser.add_argument('--workers', type=int, default=8,
                        help='# of workers')
    parser.add_argument('--num_batches_to_plot', type=int, default=10,
                        help='number of batches to plot')
    parser.add_argument('--detect_anomaly', type=int, default=0,
                        help='enable anomaly detection?')
    parser.add_argument('--grad_clip_norm', type=int, default=0,
                        help='to enable clipping, enter a norm value')
    parser.add_argument('--num_samples_for_embedding', type=int, default=200,
                        help='batches for t-SNE projection')
    parser.add_argument('--do_distributed', type=int, default=0,
                        help='move to distributed training?')
    parser.add_argument('--dry_run', action='store_true',
                        help="run a single epoch with entire train/valid sets")
    parser.add_argument('--save_test_maps', action='store_true',
                        help='save out test maps')

    # %% Test only conditions
    parser.add_argument('--save_results_here', type=str, default='',
                        help='if path is provided, it will override path to \
                        save the final test results')

    # %% Parse arguments
    args = parser.parse_args()

    if args.mode == 'one_vs_one':
        print('One vs One mode')
        args.num_sets = 1

    if args.mode == 'all_vs_one':
        print('All vs one mode detected. Ignoring cur_obj flag.')
        args.cur_obj = 'allvsone'
        args.num_sets = 9

    if args.mode == 'pretrained':
        print('Pretrain mode detected.')
        args.cur_obj = 'pretrained'
        args.num_sets = 4

    if args.mode == 'all-one_vs_one':
        args.num_sets = 8

    if args.one_by_one_ds:
        print('Disabling spike removal')
        args.remove_spikes = 0

    if args.dry_run:
        args.epochs = 1

    if args.make_uncertain:
        args.make_aleatoric = True
        args.dropout = 0.2

    print('{} sets detected'.format(args.num_sets))

    opt = vars(args)
    print('---------')
    print('Parsed arguments')
    pprint(opt)
    return args
