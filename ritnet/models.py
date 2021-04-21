#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 18:50:11 2019

@author: manoj
"""

from densenet import DenseNet2D
from Ellseg.modelSummary import model_dict as ellseg_model_dict
model_dict = {}

model_dict['densenet'] = DenseNet2D(dropout=True,prob=0.2)
model_dict['densenet_4ch'] = DenseNet2D(dropout=True,prob=0.2,out_channels=4)
model_dict['densenet_3ch'] = DenseNet2D(dropout=True,prob=0.2,out_channels=3)
model_dict['ellseg'] = ellseg_model_dict['ritnet_v2']

model_channel_dict = {}
model_channel_dict['best_model.pkl'] = ('densenet_4ch', 4, False, None)
model_channel_dict['ritnet_pupil.pkl'] = ('densenet_4ch', 4, False, None)
model_channel_dict['ritnet_400400.pkl'] = ('densenet_3ch', 3, False, None)

seg2elactivated = 1
if seg2elactivated:
    path_intermediate='_0_0'#'with_seg2el'
else:
    path_intermediate='_1_0'#'without_seg2el' 

model_channel_dict['ellseg_baseline'] = ('ellseg', 3, True, 'RC_e2e_baseline_ritnet_v2_'+'riteyes_general'+path_intermediate)
model_channel_dict['ellseg_allvsone'] = ('ellseg', 3, True, 'RC_e2e_allvsone_ritnet_v2_allvsone'+path_intermediate)
