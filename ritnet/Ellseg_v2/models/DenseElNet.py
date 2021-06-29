#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rakshit
"""
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('..')

from helperfunctions.loss import get_com
from helperfunctions.utils import detach_cpu_np, conv_layer
from helperfunctions.utils import regressionModule, linStack, convBlock

from helperfunctions.helperfunctions import plot_images_with_annotations
from helperfunctions.helperfunctions import construct_mask_from_ellipse
from helperfunctions.helperfunctions import fix_ellipse_axis_angle
from helperfunctions.helperfunctions import my_ellipse

from extern.pytorch_revgrad.src.module import RevGrad
from extern.squeeze_and_excitation.squeeze_and_excitation.squeeze_and_excitation import ChannelSpatialSELayer


def getSizes(chz, growth, blks=4):
    # For a base channel size, growth rate and number of blocks,
    # this function computes the input and output channel sizes for
    # al layers.

    # Encoder sizes
    sizes = {'enc': {'inter':[], 'ip':[], 'op': []},
             'dec': {'skip':[], 'ip': [], 'op': []}}
    sizes['enc']['inter'] = np.array([chz*(i+1) for i in range(0, blks)])
    sizes['enc']['op'] = np.array([np.int(growth*chz*(i+1)) for i in range(0, blks)])
    sizes['enc']['ip'] = np.array([chz] + [np.int(growth*chz*(i+1)) for i in range(0, blks-1)])

    # Decoder sizes
    sizes['dec']['skip'] = sizes['enc']['ip'][::-1] + sizes['enc']['inter'][::-1]
    sizes['dec']['ip'] = sizes['enc']['op'][::-1] #+ sizes['dec']['skip']
    sizes['dec']['op'] = np.append(sizes['enc']['op'][::-1][1:], chz)
    return sizes

class Transition_down(nn.Module):
    '''
    Downsampling block which uses average pooling to reduce the spatial dimensions
    '''
    def __init__(self, down_size):
        super(Transition_down, self).__init__()

        self.pool = nn.AvgPool2d(kernel_size=down_size) if down_size else False

    def forward(self, x):
        if self.pool:
            return self.pool(x)
        else:
            return x

class DenseNet2D_down_block(nn.Module):
    '''
    An encoder block inspired from DenseNet.
    '''
    def __init__(self,
                 in_c,
                 inter_c,
                 op_c,
                 down_size,
                 norm,
                 act_func,
                 scSE=False,  # Concurrent spatial and channel-wise SE
                 dropout=False,
                 track_running_stats=False):

        super(DenseNet2D_down_block, self).__init__()

        self.conv1  = conv_layer(in_c, inter_c, norm, act_func,
                                 dropout=dropout,
                                 kernel_size=3, bias=False, padding=1,
                                 track_running_stats=track_running_stats)

        self.conv21 = conv_layer(in_c+inter_c, inter_c, norm, act_func,
                                 dropout=dropout,
                                 kernel_size=1, bias=False, padding=0,
                                 track_running_stats=track_running_stats)

        self.conv22 = conv_layer(inter_c, inter_c, norm, act_func,
                                 dropout=dropout,
                                 kernel_size=3, bias=False, padding=1,
                                 track_running_stats=track_running_stats)

        self.conv31 = conv_layer(in_c+2*inter_c, inter_c, norm, act_func,
                                 kernel_size=1, bias=False, padding=0,
                                 dropout=dropout,
                                 track_running_stats=track_running_stats)

        self.conv32 = conv_layer(inter_c, inter_c, norm, act_func,
                                 kernel_size=3, bias=False, padding=1,
                                 dropout=dropout,
                                 track_running_stats=track_running_stats)

        self.conv4  = conv_layer(in_c+inter_c, op_c, norm, act_func,
                                 kernel_size=1, bias=False, padding=0,
                                 dropout=dropout,
                                 track_running_stats=track_running_stats)

        self.TD = Transition_down(down_size)

        if scSE:
            self.csE_layer = ChannelSpatialSELayer(in_c+inter_c)
        else:
            self.csE_layer = False

    def forward(self, x):
        # NOTE: input x is assumed to be batchnorm'ed

        y  = self.conv1(x)

        y = torch.cat([x, y], dim=1)
        out = self.conv22(self.conv21(y))

        out = torch.cat([y, out], dim=1)
        out = self.conv32(self.conv31(out))

        out = torch.cat([out, x], dim=1)

        if self.csE_layer:
            out = self.csE_layer(out)

        return out, self.TD(self.conv4(out))


class DenseNet2D_up_block(nn.Module):
    '''
    A lightweight decoder block which upsamples spatially using
    bilinear interpolation.
    '''
    def __init__(self, skip_c, in_c, out_c,
                 up_stride, act_func, norm,
                 scSE=False,
                 dropout=False,
                 track_running_stats=False):

        super(DenseNet2D_up_block, self).__init__()

        self.conv11 = conv_layer(skip_c+in_c, out_c, norm, act_func,
                                 kernel_size=1, bias=False, padding=0,
                                 dropout=dropout,
                                 track_running_stats=track_running_stats)
        self.conv12 = conv_layer(out_c, out_c, norm, act_func,
                                 kernel_size=3, bias=False, padding=1,
                                 dropout=dropout,
                                 track_running_stats=track_running_stats)
        self.conv21 = conv_layer(skip_c+in_c+out_c, out_c, norm, act_func,
                                 kernel_size=1, bias=False, padding=0,
                                 dropout=dropout,
                                 track_running_stats=track_running_stats)
        self.conv22 = conv_layer(out_c, out_c, norm, act_func,
                                 kernel_size=3, bias=False, padding=1,
                                 dropout=dropout,
                                 track_running_stats=track_running_stats)
        self.up_stride = up_stride

        if scSE:
            self.csE_layer = ChannelSpatialSELayer(out_c)
        else:
            self.csE_layer = False

    def forward(self, prev_feature_map, x):
        # NOTE: inputs are assumed to be batchnorm'ed

        x = F.interpolate(x,
                          mode='bilinear',
                          align_corners=False,
                          scale_factor=self.up_stride)

        x = torch.cat([x, prev_feature_map], dim=1)
        out = self.conv12(self.conv11(x))

        out = torch.cat([x, out],dim=1)
        out = self.conv22(self.conv21(out))

        if self.csE_layer:
            out = self.csE_layer(out)

        return out


class DenseNet_encoder(nn.Module):
    def __init__(self, args, in_c=1,
                 act_func=F.leaky_relu,
                 norm=nn.BatchNorm2d):

        super(DenseNet_encoder, self).__init__()

        chz = args['base_channel_size']
        growth = args['growth_rate']
        track_running_stats = args['track_running_stats']

        sizes = getSizes(chz, growth, blks=args['num_blocks'])

        opSize = sizes['enc']['op']
        ipSize = sizes['enc']['ip']
        interSize = sizes['enc']['inter']

        self.head = convBlock(in_c=in_c,
                              inter_c=chz,
                              out_c=chz,
                              act_func=act_func,
                              num_layers=1,
                              norm=norm,
                              track_running_stats=track_running_stats)

        self.down_block_list = nn.ModuleList([])
        for block_num in range(args['num_blocks']):
            block = DenseNet2D_down_block(in_c=ipSize[block_num],
                                          inter_c=interSize[block_num],
                                          op_c=opSize[block_num],
                                          down_size=2,
                                          scSE=args['use_scSE'],
                                          norm=norm,
                                          act_func=act_func,
                                          dropout=args['dropout'],
                                          track_running_stats=track_running_stats)
            self.down_block_list.append(block)

    def forward(self, x):
        x = self.head(x)

        skip_list = []
        for block in self.down_block_list:
            skip, x = block(x)
            skip_list.append(skip)
        return tuple(skip_list) + (x, )


class DenseNet_decoder(nn.Module):
    def __init__(self, args, out_c,
                 act_func=F.leaky_relu,
                 norm=nn.BatchNorm2d):

        chz = args['base_channel_size']
        growth = args['growth_rate']
        track_running_stats = args['track_running_stats']

        super(DenseNet_decoder, self).__init__()
        sizes = getSizes(chz, growth)
        skipSize = sizes['dec']['skip']
        opSize = sizes['dec']['op']
        ipSize = sizes['dec']['ip']

        self.up_block_list = nn.ModuleList([])
        for block_num in range(args['num_blocks']):
            block = DenseNet2D_up_block(skipSize[block_num],
                                        ipSize[block_num],
                                        opSize[block_num],
                                        2, act_func, norm,
                                        scSE=args['use_scSE'],
                                        dropout=args['dropout'],
                                        track_running_stats=track_running_stats)
            self.up_block_list.append(block)

        self.final = nn.Conv2d(opSize[-1], out_c, kernel_size=1, bias=True)

    def forward(self, skip_list, x):

        for block_num, block in enumerate(self.up_block_list):
            x = block(skip_list[-block_num-1], x)

        return self.final(x)


class DenseNet2D(nn.Module):
    def __init__(self,
                 args,
                 act_func=F.leaky_relu,
                 norm=nn.BatchNorm2d):
        super(DenseNet2D, self).__init__()

        self.extra_depth = args['extra_depth']

        self.sizes = getSizes(args['base_channel_size'],
                              args['growth_rate'])

        self.equi_var = args['equi_var']

        self.enc = DenseNet_encoder(args,
                                    in_c=1,
                                    act_func=act_func, norm=norm)

        # Fix the decoder to InstanceNorm
        self.dec = DenseNet_decoder(args,
                                    out_c=3,
                                    act_func=act_func, norm=nn.InstanceNorm2d)

        if args['maxpool_in_regress_mod'] > 0:
            regress_pool = nn.MaxPool2d
        elif args['maxpool_in_regress_mod'] == 0:
            regress_pool = nn.AvgPool2d
        else:
            regress_pool = False

        # Fix the Regression module to InstanceNorm
        self.elReg = regressionModule(self.sizes,
                                      sc=args['regress_channel_grow'],
                                      norm=nn.InstanceNorm2d,
                                      pool=regress_pool,
                                      dilate=args['dilation_in_regress_mod'],
                                      act_func=act_func,
                                      track_running_stats=args['track_running_stats'])

        self.make_aleatoric = args['make_aleatoric']

        if args['grad_rev']:
            self.grad_rev = True

            self.setDatasetInfo(args['num_sets'])
        else:
            self.grad_rev = False

        self.adv_disc = False

        if args['adv_DG']:
            # Use the encoder as a feature extractor and the regression module
            # to classify the domain of segmentation output. Reverse the
            # objective while training the segmentation module.
            self.adv_disc = True

            self.disc_sizes = getSizes(args['disc_base_channel_size'],
                                       args['growth_rate'])

            temp_args = copy.deepcopy(args)
            temp_args['base_channel_size'] = args['disc_base_channel_size']
            temp_args['growth_rate'] = args['growth_rate']

            self.adv_DG = DenseNet_encoder(temp_args,
                                           in_c=3,
                                           act_func=act_func,
                                           norm=nn.InstanceNorm2d)

            # Explicity assign the input channel size
            explicit_size = self.sizes['enc']['op'][-1] + \
                            self.disc_sizes['enc']['op'][-1]
            self.adv_classifier = regressionModule(self.disc_sizes,
                                                   norm=nn.InstanceNorm2d,
                                                   conf_pred=False,
                                                   ellipse_pred=False,
                                                   twice_ip_channels=False,
                                                   explicit_inChannels=explicit_size,
                                                   act_func=act_func,
                                                   dilate=False,
                                                   sc=0,  # Fixed channels
                                                   track_running_stats=args['track_running_stats'])

    def setDatasetInfo(self, numSets=2):
        # Produces a 2 layered MLP which directly maps bottleneck to the DS ID

        inChannels = self.sizes['enc']['op'][-1]

        self.numSets = numSets
        self.grad_rev_layer = RevGrad()
        self.dsIdentify_lin = linStack(num_layers=2,
                                       in_dim=inChannels,
                                       hidden_dim=128,
                                       out_dim=numSets,
                                       bias=True,
                                       actBool=False,
                                       dp=0.0)

    def forward(self, data_dict):

        # Cast to float32, move to device and add a dummy color channel
        if 'device' not in self.__dict__:
            self.device = next(self.parameters()).device

        # Move image data to GPU
        x = data_dict['image'].to(torch.float32).to(self.device,
                                                    non_blocking=True)
        x = x.unsqueeze(1)

        B, _, H, W = x.shape
        enc_op = self.enc(x)

        # Generate latent space bottleneck representation
        latent = enc_op[-1].flatten(start_dim=-2).mean(dim=-1)

        elOut, elConf = self.elReg(enc_op[-1])
        op = self.dec(enc_op[:-1], enc_op[-1])

        if torch.any(torch.isnan(op)) or torch.any(torch.isinf(op)):

            print(data_dict['archName'])
            print(data_dict['im_num'])
            print('WARNING! Convergence failed!')

            from scripts import detach_cpu_numpy

            plot_images_with_annotations(detach_cpu_numpy(data_dict),
                                         write='./FAILURE.jpg',
                                         remove_saturated=False,
                                         is_list_of_entries=False,
                                         is_predict=False,
                                         show=False)

            import sys
            sys.exit('Network predicted NaNs or Infs')

        # %% Gradient reversal on latent space
        # Observation: Adaptive Instance Norm removes domain info quite nicely
        if self.grad_rev:
            # Gradient reversal to remove DS bias
            ds_predict = self.dsIdentify_lin(self.grad_rev_layer(latent))
        else:
            ds_predict = []

        # %% Adv Discriminator
        if self.adv_disc:
            # adv_latent = self.adv_DG(F.softmax(op, dim=1))[-1]
            adv_latent = self.adv_DG(op)[-1]
            cat_latent = torch.cat([enc_op[-1], adv_latent], dim=1)
            domain_pred = self.adv_classifier(cat_latent)[0]
        else:
            domain_pred = []

        # %% Choose EllSeg proposed ellipse measures

        # Get center of ellipses from COM
        pred_pup_c = get_com(op[:, 2, ...], temperature=4)
        pred_iri_c = get_com(-op[:, 0, ...], temperature=4)

        if torch.any(torch.isnan(pred_pup_c)) or\
           torch.any(torch.isnan(pred_iri_c)):

            import sys
            print('WARNING! Convergence failed!')
            sys.exit('Pupil or Iris centers predicted as NaNs')

        # Append pupil and iris ellipse parameter predictions from latent space
        pupil_ellipse_norm = torch.cat([pred_pup_c, elOut[:, 7:10]], dim=1)
        iris_ellipse_norm = torch.cat([pred_iri_c, elOut[:, 2: 5]], dim=1)

        # %% Convert predicted ellipse back to proper scale
        if self.equi_var:
            sc = max([W, H])
            Xform_to_norm = np.array([[2/sc, 0, -1],
                                      [0, 2/sc, -1],
                                      [0, 0,     1]])
        else:
            Xform_to_norm = np.array([[2/W, 0, -1],
                                      [0, 2/H, -1],
                                      [0, 0,    1]])

        Xform_from_norm = np.linalg.inv(Xform_to_norm)

        out_dict = {}
        out_dict['iris_ellipse'] = np.zeros((B, 5))
        out_dict['pupil_ellipse'] = np.zeros((B, 5))

        for b in range(B):
            # Read each normalized ellipse in a loop and unnormalize it
            try:
                temp_var = detach_cpu_np(iris_ellipse_norm[b, ])
                temp_var = fix_ellipse_axis_angle(temp_var)
                temp_var = my_ellipse(temp_var).transform(Xform_from_norm)[0][:5]
            except Exception:
                print(temp_var)
                print('Incorrect norm iris: {}'.format(temp_var.tolist()))
                temp_var = np.ones(5, )

            out_dict['iris_ellipse'][b, ...] = temp_var

            try:
                temp_var = detach_cpu_np(pupil_ellipse_norm[b, ])
                temp_var = fix_ellipse_axis_angle(temp_var)
                temp_var = my_ellipse(temp_var).transform(Xform_from_norm)[0][:5]
            except Exception:
                print(temp_var)
                print('Incorrect norm pupil: {}'.format(temp_var.tolist()))
                temp_var = np.ones(5, )

            out_dict['pupil_ellipse'][b, ...] = temp_var

        # %% Pupil and Iris mask construction from predicted ellipses
        pupil_mask_recon = construct_mask_from_ellipse(out_dict['pupil_ellipse'], (H,W))
        iris_mask_recon = construct_mask_from_ellipse(out_dict['iris_ellipse'], (H,W))

        pd_recon_mask = np.zeros(pupil_mask_recon.shape, dtype=np.int)
        pd_recon_mask[iris_mask_recon.astype(np.bool)] = 1
        pd_recon_mask[pupil_mask_recon.astype(np.bool)] = 2

        # %% Save out predicted data and return
        out_dict['mask'] = torch.argmax(op, dim=1).detach().cpu().numpy()
        out_dict['mask_recon'] = pd_recon_mask

        out_dict['pupil_ellipse_norm'] = pupil_ellipse_norm
        out_dict['iris_ellipse_norm'] = iris_ellipse_norm

        out_dict['pupil_ellipse_norm_regressed'] = elOut[:, 5:]
        out_dict['iris_ellipse_norm_regressed'] = elOut[:, :5]

        out_dict['pupil_center'] = out_dict['pupil_ellipse'][:, :2]
        out_dict['iris_center'] = out_dict['iris_ellipse'][:, :2]

        if self.make_aleatoric:
            out_dict['pupil_conf'] = elConf[:, 5:]
            out_dict['iris_conf'] = elConf[:, :5]
        else:
            out_dict['pupil_conf'] = torch.zeros_like(elConf)
            out_dict['iris_conf'] = torch.zeros_like(elConf)

        out_dict['ds_onehot'] = ds_predict
        out_dict['disc_onehot'] = domain_pred
        out_dict['predict'] = op
        out_dict['latent'] = latent.detach().cpu()

        return out_dict


if __name__ == '__main__':
    import sys
    sys.path.append('..')

    from args_maker import make_args
    args = make_args()

    data_dict = {'image': torch.zeros(1, 1, 240, 320)}
    model = DenseNet2D(vars(args))
    temp = model(data_dict)
