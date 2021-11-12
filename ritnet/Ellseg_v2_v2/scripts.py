#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 10:38:59 2021

@author: rakshit
"""

import os
import gc
import sys
import time
import h5py
import torch
import numpy as np

from helperfunctions.loss import get_seg_loss, get_uncertain_l1_loss

from helperfunctions.helperfunctions import plot_images_with_annotations
from helperfunctions.helperfunctions import convert_to_list_entries
from helperfunctions.helperfunctions import merge_two_dicts, fix_batch

from helperfunctions.utils import get_seg_metrics, get_distance, compute_norm
from helperfunctions.utils import getAng_metric, generate_pseudo_labels
from helperfunctions.utils import remove_underconfident_psuedo_labels


def detach_cpu_numpy(data_dict):
    out_dict = {}
    for key, value in data_dict.items():
        if 'torch' in str(type(value)):
            out_dict[key] = value.detach().cpu().numpy()
        else:
            out_dict[key] = value
    return out_dict


def send_to_device(data_dict, device):
    out_dict = {}
    for key, value in data_dict.items():
        if 'torch' in str(type(value)):
            out_dict[key] = value.to(device)
        else:
            out_dict[key] = value
    return out_dict


def forward(net,
            spiker,
            logger,
            loader,
            optimizer,
            args,
            path_dict,
            epoch=0,
            mode='test',
            writer=[],
            optimizer_disc=False,
            batches_per_ep=2000):

    logger.write('{}. Epoch: {}'.format(mode, epoch))

    device = next(net.parameters()).device

    rank_cond = ((args['local_rank'] == 0) or not args['do_distributed'])

    if mode == 'train':
        net.train()
    else:
        net.eval()

    io_time = []
    loader_iter = iter(loader)

    metrics = []
    dataset_id = []
    embeddings = []

    if (mode == 'test') and args['save_test_maps']:
        logger.write('Generating test object')
        test_results_obj = h5py.File(path_dict['results']+'/test_results.h5',
                                     'w', swmr=True)

    for bt in range(batches_per_ep):
        start_time = time.time()

        try:
            data_dict = next(loader_iter)
        except StopIteration:
            print('Loader reset')
            loader_iter = iter(loader)
            data_dict = next(loader_iter)
            args['time_to_update'] = True

        if torch.any(data_dict['is_bad']):
            logger.write('Bad batch found!', do_warn=True)

            # DDP crashes if we skip over a batch and no gradients are matched
            # To avoid this, remove offending samples by replacing it with a
            # good sample randomly drawn from rest of the batch
            data_dict = fix_batch(data_dict)

        end_time = time.time()
        io_time.append(end_time - start_time)

        if args['do_distributed']:
            torch.distributed.barrier()

        assert torch.all(data_dict['pupil_ellipse'][:, -1] > 0), \
            'pupil ellipse orientation > 0'
        assert torch.all(data_dict['pupil_ellipse'][:, -1] < np.pi), \
            'pupil ellipse orientation < pi'
        assert torch.all(data_dict['iris_ellipse'][:, -1] > 0), \
            'iris ellipse orientation > 0'
        assert torch.all(data_dict['iris_ellipse'][:, -1] < np.pi), \
            'iris ellipse orientation < pi'

        with torch.autograd.set_detect_anomaly(bool(args['detect_anomaly'])):
            with torch.cuda.amp.autocast(enabled=bool(args['mixed_precision'])):

                # Change behavior in train and test mode
                if mode == 'train':
                    out_dict = net(data_dict)
                else:
                    with torch.no_grad():
                        out_dict = net(data_dict)

                try:
                    total_loss, disc_loss, loss_dict = get_loss(send_to_device(data_dict, device),
                                                                out_dict,
                                                                float(args['alpha']),
                                                                float(args['beta']),
                                                                adv_loss=args['adv_DG'],
                                                                bias_removal=args['grad_rev'],
                                                                make_aleatoric=args['make_aleatoric'],
                                                                pseudo_labels=args['pseudo_labels'],
                                                                regress_loss=args['regression_from_latent'])

                    total_loss_value = total_loss.item()

                except Exception as e:

                    print(e)

                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)

                    #  Something broke during loss computation, safe skip this
                    print('Something broke during loss computation')
                    print(data_dict['archName'])
                    print(data_dict['im_num'])

                    print('--------------- Skipping this batch from training ---------------')

                    # Zero out gradients
                    optimizer.zero_grad()
                    net.zero_grad()

                    # del out_dict

                    # Skip everything else in the current loop
                    # and proceed with the next batch cause this
                    # batch sucks
                    sys.exit('Exiting ...')


            is_spike = spiker.update(total_loss_value) if spiker else False

            if mode == 'train':

                # NOTE: If mixed precision is enabled, it automatically scales
                # the loss to full precision
                if args['adv_DG']:
                    # Teach the disc to classify the domains based on predicted
                    # segmentation mask
                    disc_loss.backward(retain_graph=True)

                    # Remove gradients accumulated in the encoder and decoder
                    net.enc.zero_grad()
                    net.dec.zero_grad()
                    net.elReg.zero_grad()

                total_loss.backward()

                if not is_spike:

                    total_norm = compute_norm(net)

                    # Gradient clipping, if needed, goes here
                    if args['grad_clip_norm'] > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(),
                                                                   max_norm=args['grad_clip_norm'],
                                                                   norm_type=2)
                        if grad_norm > args['grad_clip_norm']:
                            logger.write('Clipping gradients from %f to %f' % (grad_norm, args['grad_clip_norm']), do_warn=True)

                    # Step the optimizer and update weights
                    if args['adv_DG']:
                        optimizer_disc.step()

                    optimizer.step()

                else:
                    total_norm = np.inf
                    print('-------------')
                    print('Spike detected! Loss: {}'.format(total_loss_value))
                    print('-------------')

            # Zero out gradients no matter what
            optimizer.zero_grad()
            net.zero_grad()

            if optimizer_disc:
                optimizer_disc.zero_grad()

        if args['do_distributed']:
            torch.cuda.synchronize()

        if bt < args['num_samples_for_embedding']:
            embeddings.append(out_dict['latent'])
            dataset_id.append(data_dict['ds_num'])

        # Get metrics
        batch_results = get_metrics(detach_cpu_numpy(data_dict),
                                    detach_cpu_numpy(out_dict))

        # Record network outputs
        if (mode == 'test') and args['save_test_maps']:

            test_dict = merge_two_dicts(detach_cpu_numpy(out_dict),
                                        detach_cpu_numpy(batch_results))
            test_dict_list = convert_to_list_entries(test_dict)

            for idx, entry in enumerate(test_dict_list):

                sample_id = data_dict['archName'][idx] + '/' + \
                                    str(data_dict['im_num'][idx].item())

                for key, value in entry.items():

                    sample_id_key = sample_id + '/' + key

                    try:
                        if 'predict' not in key:
                            if 'mask' in key:
                                # Save mask as integer objects to avoid
                                # overloading harddrive
                                test_results_obj.create_dataset(sample_id_key,
                                                                data=value,
                                                                dtype='uint8',
                                                                compression='lzf')
                            else:
                                # Save out remaining data points with float16 to
                                # avoid overloading harddrive
                                test_results_obj.create_dataset(sample_id_key,
                                                                data=np.array(value),
                                                                dtype='float16')
                    except Exception:
                        print('Repeated sample because of corrupt entry in H5')
                        print('Skipping sample {} ... '.format(sample_id))

        batch_results['loss'] = total_loss_value

        batch_metrics = aggregate_metrics([batch_results])
        metrics.append(batch_results)

        logger.write('{}. Bt: {}. Loss: {}'.format(mode, bt, total_loss_value))

        # Use this if you want to save out spiky conditions
        # if ((bt % 100 == 0) or is_spike) and rank_cond and (mode == 'train'):

        if (bt % 100 == 0) and rank_cond and (mode == 'train'):
            # Saving spiky conditions unnecessarily bloats the drive
            out_dict['image'] = data_dict['image']

            if is_spike:
                # Save out lossy batches in a separate folder and avoid
                # learning on them
                im_name = 'spike/ep_{}_bt_{}.jpg'.format(epoch, bt)
            else:
                im_name = '{}/bt_{}.jpg'.format(epoch, bt)

            out_path_im_name = os.path.join(path_dict['figures'], im_name)
            plot_images_with_annotations(detach_cpu_numpy(out_dict),# data_dict
                                         write=out_path_im_name,
                                         remove_saturated=False,
                                         is_list_of_entries=False,
                                         is_predict=True,
                                         show=False)

        step_index = epoch*batches_per_ep + bt

        if (mode == 'train') and rank_cond:
            # Save out gradients
            writer.add_scalar('Local/' + mode + '/grad_norm',
                              total_norm,
                              global_step=step_index)

        # Write out batch statistics
        if (mode != 'test') and rank_cond:
            # Log batch data
            writer.add_scalars('local/' + mode+'/metrics/',
                               {key:val for key,val in batch_metrics.items() if 'mean' in key},
                               global_step=step_index)
            writer.add_scalars('local/' + 'loss/' + mode + '_',
                               loss_dict,
                               global_step=step_index)

        del out_dict  # Explicitly free up memory

    results_dict = aggregate_metrics(metrics)

    if (mode != 'test') and rank_cond:
        # Log data

        writer.add_scalar(mode+'/io_secs',
                          np.mean(io_time),
                          global_step=epoch)
        writer.add_scalars(mode+'/metrics',
                           {key:val for key, val in results_dict.items() if 'mean' in key},
                           global_step=epoch)

        # Log embeddings
        writer.add_embedding(torch.cat(embeddings, dim=0),
                             metadata=torch.cat(dataset_id),
                             global_step=epoch,
                             tag=mode)

    if (mode == 'test') and args['save_test_maps']:
        test_results_obj.close()

    # Clear out RAM accumulation if any
    del loader_iter

    # Clear out the cuda and RAM cache
    torch.cuda.empty_cache()
    gc.collect()

    return results_dict


# %% Loss function
def get_loss(gt_dict, pd_dict, alpha, beta,
             make_aleatoric=False, regress_loss=True, bias_removal=False,
             pseudo_labels=False, adv_loss=False, label_tracker=False):

    # Segmentation loss
    loss_seg = get_seg_loss(gt_dict, pd_dict, 0.5)

    # L1 w/o uncertainity loss for segmentation center
    loss_pupil_c = get_uncertain_l1_loss(gt_dict['pupil_center_norm'],
                                         pd_dict['pupil_ellipse_norm'][:, :2],
                                         None,
                                         uncertain=False,
                                         cond=gt_dict['pupil_center_available'],
                                         do_aleatoric=False)

    loss_iris_c  = get_uncertain_l1_loss(gt_dict['iris_ellipse_norm'][:, :2],
                                         pd_dict['iris_ellipse_norm'][:, :2],
                                         None,
                                         uncertain=False,
                                         cond=gt_dict['iris_ellipse_available'],
                                         do_aleatoric=False)

    # L1 with uncertainity loss for pupil center regression
    loss_pupil_c_reg = get_uncertain_l1_loss(gt_dict['pupil_center_norm'],
                                             pd_dict['pupil_ellipse_norm_regressed'][:, :2],
                                             None,
                                             uncertain=pd_dict['pupil_conf'][:, :2],
                                             cond=gt_dict['pupil_center_available'],
                                             do_aleatoric=make_aleatoric)

    # L1 with uncertainity loss for ellipse parameter regression from latent
    loss_pupil_el = get_uncertain_l1_loss(gt_dict['pupil_ellipse_norm'][:, 2:],
                                          pd_dict['pupil_ellipse_norm_regressed'][:, 2:],
                                          [4, 4, 3],
                                          uncertain=pd_dict['pupil_conf'][:, 2:],
                                          cond=gt_dict['pupil_ellipse_available'],
                                          do_aleatoric=make_aleatoric)

    loss_iris_el = get_uncertain_l1_loss(gt_dict['iris_ellipse_norm'],
                                         pd_dict['iris_ellipse_norm_regressed'],
                                         [1, 1, 4, 4, 3],
                                         uncertain=pd_dict['iris_conf'],
                                         cond=gt_dict['iris_ellipse_available'],
                                         do_aleatoric=make_aleatoric)

    # Gradient reversal
    if bias_removal:
        loss_da = torch.nn.functional.cross_entropy(pd_dict['ds_onehot'],
                                                    gt_dict['ds_num'].to(torch.long))

    else:
        loss_da = torch.tensor([0.0]).to(loss_seg.device)

    if not regress_loss:
        loss = 0.5*(20*loss_seg + loss_da) + (1-0.5)*(loss_pupil_c + loss_iris_c)
    else:
        loss = 0.5*(20*loss_seg + loss_da) + \
               (1-0.5)*(loss_pupil_c + loss_iris_c +
                         loss_pupil_el + loss_iris_el + loss_pupil_c_reg)

    if adv_loss:
        da_loss_dec = torch.nn.functional.cross_entropy(pd_dict['disc_onehot'],
                                                        gt_dict['ds_num'].to(torch.long))

        # Inverse domain classification, we want to increase domain confusion
        # as training progresses. Ensure that the weight does not exceed
        # the weight of the main loss at any given epoch or else it could
        # lead to unexpected solutions in order to confused the discriminator
        loss = loss - 0.4*beta*da_loss_dec
    else:
        da_loss_dec = torch.tensor([0.0]).to(loss_seg.device)

    if pseudo_labels:

        # Generate pseudo labels and confidence based on entropy
        pseudo_labels, conf = generate_pseudo_labels(pd_dict['predict'])

        # Based on samples with groundtruth information, classify each
        # prediction as "good" or "bad" from entropy-based-confidence
        loc = remove_underconfident_psuedo_labels(conf.detach(),
                                                  label_tracker,
                                                  gt_dict=False) # gt_dict

        # Remove under confident pseudo labels
        # pseudo_labels[loc] = -1
        # conf[loc] = 0.0

        # Remove pseudo labels for samples with groundtruth
        loc = gt_dict['mask'] != -1  # Samples with groundtruth
        pseudo_labels[loc] = -1  # Disable pseudosamples
        conf[loc] = 0.0

        # Number of pixels and samples which are non-zero confidence
        num_valid_pxs = torch.sum(conf.flatten(start_dim=-2) > 0, dim=-1)+1
        num_valid_samples = torch.sum(num_valid_pxs > 0)

        pseudo_loss = torch.nn.functional.cross_entropy(pd_dict['predict'],
                                                        pseudo_labels,
                                                        ignore_index=-1,
                                                        reduction='none')
        pseudo_loss = (conf*pseudo_loss).flatten(start_dim=-2)
        pseudo_loss = torch.sum(pseudo_loss, dim=-1)/num_valid_pxs  # Average across pixels
        pseudo_loss = torch.sum(pseudo_loss, dim=0)/num_valid_samples  # Average across samples

        loss = loss + beta*pseudo_loss
    else:
        pseudo_loss = torch.tensor([0.0])

    loss_dict = {'da_loss': loss_da.item(),
                 'seg_loss': loss_seg.item(),
                 'pseudo_loss': pseudo_loss.item(),
                 'da_loss_dec': da_loss_dec.item(),
                 'iris_c_loss': loss_iris_c.item(),
                 'pupil_c_loss': loss_pupil_c.item(),
                 'iris_el_loss': loss_iris_el.item(),
                 'pupil_c_reg_loss': loss_pupil_c_reg.item(),
                 'pupil_params_loss': loss_pupil_el.item(),
                 }

    return loss, da_loss_dec, loss_dict


# %% Get performance metrics
def get_metrics(gt_dict, pd_dict):
    # Results metrics of important per sample

    metric_dict = {}

    height, width = gt_dict['mask'].shape[-2:]
    scale = min([height, width])

    # Segmentation IoU
    metric_dict['iou'] = get_seg_metrics(gt_dict['mask'],
                                         pd_dict['mask'],
                                         gt_dict['mask_available'])

    metric_dict['iou_recon'] = get_seg_metrics(gt_dict['mask'],
                                               pd_dict['mask_recon'],
                                               gt_dict['mask_available'])

    # Pupil and Iris center metric
    metric_dict['pupil_c_dst'] = get_distance(gt_dict['pupil_center'],
                                              pd_dict['pupil_ellipse'][:, :2],
                                              gt_dict['pupil_center_available'])

    metric_dict['iris_c_dst'] = get_distance(gt_dict['iris_ellipse'][:, :2],
                                             pd_dict['iris_ellipse'][:, :2],
                                             gt_dict['iris_ellipse_available'])

    # Pupil and Iris axis metric
    metric_dict['pupil_axes_dst'] = get_distance(gt_dict['pupil_ellipse'][:, 2:-1],
                                                 pd_dict['pupil_ellipse'][:, 2:-1],
                                                 gt_dict['pupil_ellipse_available'])


    metric_dict['iris_axes_dst'] = get_distance(gt_dict['iris_ellipse'][:, 2:-1],
                                                pd_dict['iris_ellipse'][:, 2:-1],
                                                gt_dict['iris_ellipse_available'])

    # Pupil and Iris angle metric
    metric_dict['pupil_ang_dst'] = getAng_metric(gt_dict['pupil_ellipse'][:, -1],
                                                 pd_dict['pupil_ellipse'][:, -1],
                                                 gt_dict['pupil_ellipse_available'])


    metric_dict['iris_ang_dst'] = getAng_metric(gt_dict['iris_ellipse'][:, -1],
                                                pd_dict['iris_ellipse'][:, -1],
                                                gt_dict['iris_ellipse_available'])

    # Evaluation metric
    # max value will be 1, min value  will be 0. All individual metrics are
    # scaled approximately equally
    term_A = metric_dict['iou'].mean(axis=1) \
        if np.any(gt_dict['mask_available']) \
        else np.nan*np.zeros((gt_dict['mask_available'].shape[0], ))
    term_A[~gt_dict['mask_available']] = np.nan

    term_B = metric_dict['iou_recon'].mean(axis=1) \
        if np.any(gt_dict['mask_available']) \
        else np.nan*np.zeros((gt_dict['mask_available'].shape[0], ))
    term_B[~gt_dict['mask_available']] = np.nan

    term_C = 1 - (1/scale)*metric_dict['pupil_c_dst'] \
        if np.any(gt_dict['pupil_center_available']) \
        else np.nan*np.zeros((gt_dict['mask_available'].shape[0], ))
    term_C[~gt_dict['pupil_center_available']] = np.nan

    term_D = 1 - (1/scale)*metric_dict['iris_c_dst'] \
        if np.any(gt_dict['iris_ellipse_available']) \
        else np.nan*np.zeros((gt_dict['mask_available'].shape[0], ))
    term_D[~gt_dict['iris_ellipse_available']] = np.nan

    term_mat = np.stack([term_A, term_B, term_C, term_D], axis=1)

    # Score per sample
    metric_dict['score'] = np.nanmean(term_mat, axis=1)
    return metric_dict


def aggregate_metrics(list_metric_dicts):
    # Aggregate and compute global stats
    keys_list = list_metric_dicts[0].keys()
    agg_dict = {}
    for key_entry in keys_list:
        try:
            if 'loss' in key_entry:
                agg_dict[key_entry] = np.array([ele[key_entry] for ele in list_metric_dicts])
            else:
                agg_dict[key_entry] = np.concatenate([ele[key_entry] for ele in list_metric_dicts], axis=0)
            if  'iou' in key_entry:
                # If more than 1 dimension, then it corresponds to iou tag
                agg_dict[key_entry] = np.nanmean(agg_dict[key_entry], axis=1)

                agg_dict[key_entry+'_mean'] = np.nanmean(agg_dict[key_entry], axis=0)
            else:
                agg_dict[key_entry+'_mean'] = np.nanmean(agg_dict[key_entry], axis=0)
        except:
            pass
    return agg_dict



