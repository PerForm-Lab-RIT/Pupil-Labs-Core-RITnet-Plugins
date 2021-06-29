# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 21:36:48 2020

@author: Rudra
"""

import pickle as pkl

# NVGaze
nv_subs1 = ['nvgaze_female_{:02}_public_50K_{}'.format(i+1, j+1) for i in range(0, 4) for j in range(0, 4)]
nv_subs2 = ['nvgaze_male_{:02}_public_50K_{}'.format(i+1, j+1) for i in range(0, 4) for j in range(0, 4)]
nv_subs_train = nv_subs1 + nv_subs2

nv_subs1 = ['nvgaze_female_{:02}_public_50K_{}'.format(5, j+1) for j in range(0, 4)]
nv_subs2 = ['nvgaze_male_{:02}_public_50K_{}'.format(5, j+1) for j in range(0, 4)]
nv_subs_test = nv_subs1 + nv_subs2

# OpenEDS
openeds_train = ['train']
openeds_test = ['validation']

# LPW
lpw_subs_train = ['LPW_{}'.format(i+1) for i in [1,3,4,6,9,10,12,13,16,18,20,21]]
lpw_subs_test = ['LPW_{}'.format(i+1) for i in [2,5,7,8,11,14,15,17,19]]

# S-General
riteyes_subs_train_gen = ['riteyes-s-general_{}'.format(i+1) for i in range(0, 18)]
riteyes_subs_test_gen = ['riteyes-s-general_{}'.format(i+1) for i in range(18, 24)]

# S-Natural
riteyes_subs_train_nat = ['riteyes-s-natural_{}'.format(i+1) for i in range(0, 18)]
riteyes_subs_test_nat = ['riteyes-s-natural_{}'.format(i+1) for i in range(18, 24)]

# Fuhl
fuhl_subsets =  ['data set I', 'data set II', 'data set III', 'data set IV',
                 'data set IX', 'data set V', 'data set VI', 'data set VII',
                 'data set VIII', 'data set X', 'data set XI', 'data set XII',
                 'data set XIII', 'data set XIV', 'data set XIX', 'data set XVI',
                 'data set XVII', 'data set XVIII', 'data set XX', 'data set XXI',
                 'data set XXII', 'data set XXIII', 'data set XV', 'data set XXIV',
                 'data set new I', 'data set new II', 'data set new III', 'data set new IV', 'data set new V']

fuhl_subs_train = [fuhl_subsets[ele] for ele in
                   [0, 2, 4, 6, 7, 10, 11, 13, 15, 17, 18, 19, 20, 23, 25, 27]]
fuhl_subs_test = [fuhl_subsets[ele] for ele in
                  [1, 3, 5, 8, 9, 12, 14, 16, 21, 22, 24, 26, 28]]

# UnityEyes
ueyes_train = ['UnityEyes_{}'.format(i) for i in range(1, 10)][:8]
ueyes_test = ['UnityEyes_{}'.format(i) for i in range(1, 10)][8:]

# Swirski
swirkski_train = ['Swirski_p1-left', 'Swirski_p1-right']
swirkski_test = ['Swirski_p2-left', 'Swirski_p2-right']

# Santini
santini_train = ['Santini_1', 'Santini_2', 'Santini_3']
santini_test = ['Santini_4', 'Santini_5', 'Santini_6']

# %% Generate split dictionaries

DS_train = {'LPW': lpw_subs_train,
            'Fuhl': fuhl_subs_train,
            'NVGaze': nv_subs_train,
            'Swirski': swirkski_train,
            'OpenEDS': openeds_train,
            'Santini': santini_train,
            'UnityEyes': ueyes_train,
            'riteyes-s-general': riteyes_subs_train_gen,
            'riteyes-s-natural': riteyes_subs_train_nat,
            }

DS_test = {'LPW': lpw_subs_test,
           'Fuhl': fuhl_subs_test,
           'NVGaze': nv_subs_test,
           'Swirski': swirkski_test,
           'OpenEDS': openeds_test,
           'Santini': santini_test,
           'UnityEyes': ueyes_test,
           'riteyes-s-general': riteyes_subs_test_gen,
           'riteyes-s-natural': riteyes_subs_test_nat,
           }

DS_selections = {'train': DS_train,
                 'test': DS_test}

pkl.dump(DS_selections, open('dataset_selections.pkl', 'wb'))
