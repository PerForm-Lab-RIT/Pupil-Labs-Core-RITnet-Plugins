#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 10:17:31 2021

@author: rsk3900
"""
import os
import sys
import h5py
import tqdm
import scipy
import pickle
import argparse
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

sys.path.append('..')

from helperfunctions.utils import normPts
from helperfunctions.helperfunctions import plot_2D_hist

def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_data', type=str,
                        default='/data/datasets/All')
    parser.add_argument('--sel_ds', type=str,
                        default='riteyes-s-natural')
    args = parser.parse_args()
    return args


def find_fname(path, name):
    options = [ele for ele in os.listdir(path) if name in ele]
    return options


def accumulate_stats_per_entry(im_num, h5_obj):

    # Histogram of normalized image
    image = h5_obj['Images'][im_num, ...]
    image = (image - np.mean(image))/np.std(image)
    counts = scipy.ndimage.histogram(image.flatten(), -3, 3, 300)
    counts = counts/np.sum(counts) # Actual probability score

    height_width = image.shape[:2]

    # Pupil center location
    if h5_obj['pupil_loc'].__len__() != 0:
        pupil_center = h5_obj['pupil_loc'][im_num, ...]
        # pupil_center = normPts(pupil_center, np.array(height_width))
    else:
        pupil_center = -np.ones(2, )

    # Iris center location
    if h5_obj['Fits']['iris'].__len__() != 0:
        iris_center = h5_obj['Fits']['iris'][im_num, :2]
        # iris_center = normPts(iris_center, np.array(height_width[::-1]))
    else:
        iris_center = -np.ones(2, )

    datum = {}
    datum['hist'] = counts
    datum['pupil_c'] = pupil_center
    datum['iris_c'] = iris_center
    return datum


def accumulate_stats_per_subset(path_H5):
    h5_obj = h5py.File(path_H5, 'r', swmr=True)

    num_images = h5_obj['Images'].shape[0]
    stats = []

    for idx in range(num_images):
        # stats = pool.apply(accumulate_stats_per_entry,
                           # args=(idx, h5_obj))
        stats += [accumulate_stats_per_entry(idx, h5_obj)]

    return stats


def accumulate_stats(args, subsets):

    num_ss = len(subsets)
    pool = mp.Pool(mp.cpu_count() if num_ss > mp.cpu_count() else num_ss)

    stats = []

    def log_result(result):
        stats.append(result)

    for subset in tqdm.tqdm(subsets):
        try:
            subset_fname = find_fname(args['path_data'], subset)[0]
        except:
            import pdb; pdb.set_trace()
        path_H5 = os.path.join(args['path_data'], subset_fname)
        pool.apply_async(accumulate_stats_per_subset,
                         args=(path_H5, ),
                         callback=log_result)

    pool.close()
    pool.join()

    return stats


def collapse_stats(stats):
    data = []
    for value in stats:
        data += value
    return data


def print_stats(stats, mode):

    # Number of images
    print('%s. # of images: %d' % (mode, len(stats)))


def plot_stats(stats, mode):
    pupil_c = [ele['pupil_c'] for ele in stats]
    pupil_c = np.stack(pupil_c, axis=0).squeeze()

    iris_c = [ele['iris_c'] for ele in stats]
    iris_c = np.stack(iris_c, axis=0).squeeze()

    l_hist = [ele['hist'] for ele in stats]
    l_hist = np.stack(l_hist, axis=0).squeeze()

    plot_2D_hist(pupil_c[:, 0], pupil_c[:, 1],
                 [0, 640], [0, 480],
                 str_save=mode+'_pupil_hist.jpg')

    plot_2D_hist(iris_c[:, 0], iris_c[:, 1],
                 [0, 640], [0, 480],
                 str_save=mode+'_iris_hist.jpg')

    x_range = np.linspace(-3, 3, 300)

    l_std = l_hist.std(axis=0)
    l_mean = l_hist.mean(axis=0)
    l_median = np.median(l_hist, axis=0)

    fig, ax = plt.subplots()
    ax.plot(x_range, l_mean, '--')
    ax.plot(x_range, l_median, '-')
    # ax.fill_between(x_range, l_mean - l_std, l_mean + l_std, alpha=0.2)
    fig.savefig(mode+'_L_hist.jpg', dpi=600)


if __name__=='__main__':
    args = vars(make_args())

    with open('../cur_objs/dataset_selections.pkl', 'rb') as f:
        DS_selections = pickle.load(f)

    stats_dict = {}
    stats_dict['train'] = accumulate_stats(args,
                                           DS_selections['train'][args['sel_ds']])
    stats_dict['test'] = accumulate_stats(args,
                                          DS_selections['test'][args['sel_ds']])

    stats_train = collapse_stats(stats_dict['train'])
    stats_test = collapse_stats(stats_dict['test'])

    # Save memory
    del stats_dict

    print_stats(stats_train, args['sel_ds']+'_train')
    print_stats(stats_test,  args['sel_ds']+'_test')

    plot_stats(stats_train,  args['sel_ds']+'_train')
    plot_stats(stats_test,  args['sel_ds']+'_test')



