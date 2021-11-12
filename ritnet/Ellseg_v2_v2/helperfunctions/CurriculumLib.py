#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:06:33 2019

@author: rakshit
"""

import re
import os
import cv2
import h5py
import copy
import torch
import pickle
import random

import numpy as np
import scipy.io as scio

from ..helperfunctions.data_augment import augment, flip
from torch.utils.data import Dataset

from ..helperfunctions.helperfunctions import simple_string, one_hot2dist
from ..helperfunctions.helperfunctions import pad_to_shape, get_ellipse_info
from ..helperfunctions.helperfunctions import extract_datasets, scale_by_ratio
from ..helperfunctions.helperfunctions import fix_ellipse_axis_angle, dummy_data

from ..helperfunctions.utils import normPts

from sklearn.model_selection import StratifiedKFold, train_test_split

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"  # Deactive file locking


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


class DataLoader_riteyes(Dataset):
    def __init__(self,
                 dataDiv_Obj,
                 path2data,
                 cond,
                 augFlag=False,
                 size=(480, 640),
                 fold_num=0,
                 sort='random',
                 scale=False):

        self.mode = cond

        cond = 'train_idx' if 'train' in cond else cond
        cond = 'valid_idx' if 'valid' in cond else cond
        cond = 'test_idx' if 'test' in cond else cond

        # Operational variables
        self.arch = dataDiv_Obj.arch               # Available archives
        self.size = size                           # Expected size of images
        self.scale = scale
        self.imList = dataDiv_Obj.folds[fold_num][cond]  # Image list
        self.augFlag = augFlag                           # Augmentation flag
        self.equi_var = True                       # Default is always True
        self.path2data = path2data                 # Path to expected H5 files

        #  You can specify which augs you want as input to augment
        self.augger = augment() if augFlag else []
        self.flipper = flip()

        # Get dataset index by archive ID
        ds_present, ds_index = extract_datasets(self.arch[self.imList[:, 1]])
        self.imList = np.hstack([self.imList, ds_index[:, np.newaxis]])
        self.fileObjs = {}

        avail_ds, counts = np.unique(ds_index, return_counts=True)

        # Repeat poorly represented datasets such that equal number of images
        # exist per dataset
        if len(counts) > 1:
            extra_samples = []
            for ii, ds_itr in enumerate(avail_ds.tolist()):
                num_more_images_needed = max(counts) - counts[ii]
                if num_more_images_needed > 0:
                    loc = np.where(self.imList[:, -1] == ds_itr)[0]
                    extra_loc = np.random.choice(loc,
                                                 size=num_more_images_needed)
                    extra_samples.append(self.imList[extra_loc, :])

            extra_samples = np.concatenate(extra_samples, axis=0)
            self.imList = np.concatenate([self.imList, extra_samples])
            len_cond = self.imList.shape[0] == len(avail_ds)*max(counts)
            assert len_cond, 'Samples must equal N X the max samples present'

        self.sort(sort)                                 # Sort order of images

    def sort(self, sort, batch_size=None):

        if sort=='ordered':
            # Completely ordered
            loc = np.unique(self.imList,
                            return_counts=True,
                            axis=0)
            print('Warning. Non-unique file list.') if np.any(loc[1]!=1) else print('Sorted list')
            self.imList = loc[0]

        elif sort=='semiordered':
            # Randomize first, then sort by archNum
            self.sort(sort='random')
            loc = np.argsort(self.imList[:, 1])
            self.imList = self.imList[loc, :]

        elif sort=='random':
            # Completely random selection. DEFAULT.
            loc = np.random.permutation(self.imList.shape[0])
            self.imList = self.imList[loc, :]

        elif sort=='mutliset_random':
            # Randomize first, then rearrange by BS / num_sets images per set.
            # This ensures that equal number of images from each dataset are
            # read in per batch read.
            self.sort('random')
            avail_ds, counts = np.unique(self.imList[:, -1],
                                         return_counts=True)
            temp_imList = []
            for ds_itr in np.nditer(avail_ds):
                loc = self.imList[:, -1] == ds_itr
                temp_imList.append(self.imList[loc, :])
            temp_imList = np.stack(temp_imList, axis=1).reshape(-1, 3)
            assert temp_imList.shape == self.imList.shape, 'Incorrect reshaping'
            self.imList = temp_imList

        elif sort=='one_by_one_ds':
            # Randomize first, then rearrange such that each BS contains image
            # from a single dataset
            self.sort('random')
            avail_ds, counts = np.unique(self.imList[:, -1],
                                         return_counts=True)

            # Create a list of information for each individual dataset
            # present within the selection
            temp_imList = []
            for ds_itr in np.nditer(avail_ds):
                loc = self.imList[:, -1] == ds_itr
                temp_imList.append(self.imList[loc, :])

            cond = True
            counter = 0

            imList = [] # Blank initialization
            while cond:
                counter+=1
                # Keep extracting batch_size elements from each entry
                ds_order = random.sample(range(avail_ds.max()),
                                         avail_ds.max())
                
                for i in range(avail_ds.max()):                        
                    idx = ds_order[i] if ds_order else 0
                    start = (counter-1)*batch_size
                    stop = counter*batch_size
                    
                    if stop < temp_imList[idx].shape[0]:
                        imList.append(temp_imList[idx][start:stop, ...])
                    else:
                        # A particular dataset has been completely sampled
                        counter = 0
                        cond = False # Break out of main loop
                        break # Break out of inner loop
            self.imList = np.concatenate(imList, axis=0)

        else:
            import sys
            sys.exit('Incorrect sorting options')

    def __len__(self):
        return self.imList.shape[0]

    def __del__(self, ):
        for entry, h5_file_obj in self.fileObjs.items():
            h5_file_obj.close()

    def __getitem__(self, idx):
        '''
        Reads in an image and all the required sources of information.
        Also returns a flag tensor where a 0 in:
            pos 0: indicates pupil center exists
            pos 1: indicates mask exists
            pos 2: indicates pupil ellipse exists
            pos 3: indicates iris ellipse exists
            ##modified:
        '''

        try:
            numClasses = 3
            data_dict = self.readEntry(idx)
            data_dict = pad_to_shape(data_dict, to_size=(480, 640))

            if self.scale:
                data_dict = scale_by_ratio(data_dict, 0.5)

            data_dict = self.augger(data_dict) if self.augFlag else data_dict
            
            assert (data_dict['image'].max() <= 255) and (data_dict['image'].min() >= 0), 'Preprocess failure'

            # Always keep flipping the image with 0.5 prob
            data_dict = self.flipper(data_dict)

        except Exception:
            print('Error reading and processing data!')
            im_num = self.imList[idx, 0]
            arch_num = self.imList[idx, 1]
            archStr = self.arch[arch_num]
            print('Bad sampled number: {}'.format(im_num))
            print('Bad archive number: {}'.format(arch_num))
            print('Bad archive name: {}'.format(archStr))
            data_dict = dummy_data(shape=(480//2, 640//2))

        height, width = data_dict['image'].shape

        data_dict['pupil_ellipse'] = fix_ellipse_axis_angle(data_dict['pupil_ellipse'])
        data_dict['iris_ellipse'] = fix_ellipse_axis_angle(data_dict['iris_ellipse'])

        if data_dict['mask_available']:
            # Modify labels by removing Sclera class
            data_dict['mask'][data_dict['mask'] == 1] = 0  # Move Sclera to 0
            data_dict['mask'][data_dict['mask'] == 2] = 1  # Move Iris to 1
            data_dict['mask'][data_dict['mask'] == 3] = 2  # Move Pupil to 2

            # Compute edge weight maps
            spatial_weights = cv2.Canny(data_dict['mask'].astype(np.uint8), 0, 1)/255
            spatial_weights = 1 + cv2.dilate(spatial_weights, (3, 3),
                                             iterations=1)*20
            data_dict['spatial_weights'] = spatial_weights

            # Calculate distance_maps for only Iris and Pupil.
            # Pupil: 2. Iris: 1. Rest: 0.
            distance_map = np.zeros(((3, ) + data_dict['image'].shape))

            # Find distance map for each class for surface loss
            if data_dict['mask_available']:
                for i in range(0, numClasses):
                    distance_map[i, ...] = one_hot2dist(data_dict['mask'].astype(np.uint8)==i)
            data_dict['distance_map'] = distance_map
        else:
            data_dict['distance_map'] = np.zeros(((3, ) + data_dict['image'].shape))
            data_dict['spatial_weights'] = np.zeros_like(data_dict['mask'])

        # Convert data to torch primitives
        data_dict['image'] = (data_dict['image'] - data_dict['image'].mean())/data_dict['image'].std()
        
        if np.any(np.isinf(data_dict['image'])) or np.any(np.isnan(data_dict['image'])):
            data_dict['image'] = np.zeros_like(data_dict['image']).astype(np.uint8)
            data_dict['is_bad'] = True

        # Groundtruth annotation mask to torch long
        data_dict['mask'] = MaskToTensor()(data_dict['mask']).to(torch.long)

        # Generate normalized pupil and iris information
        if self.equi_var:
            sc = max([width, height])
            H = np.array([[2/sc, 0, -1], [0, 2/sc, -1], [0, 0, 1]])
        else:
            H = np.array([[2/width, 0, -1], [0, 2/height, -1], [0, 0, 1]])

        if not data_dict['is_bad']:
            data_dict['iris_ellipse_norm'] = get_ellipse_info(data_dict['iris_ellipse'], H,
                                                              data_dict['iris_ellipse_available'])[1]
            data_dict['pupil_ellipse_norm'] = get_ellipse_info(data_dict['pupil_ellipse'], H,
                                                               data_dict['pupil_ellipse_available'])[1]

            # Generate normalized pupil center location
            data_dict['pupil_center_norm'] = normPts(data_dict['pupil_center'],
                                                     np.array([width, height]),
                                                     by_max=self.equi_var)
        else:
            data_dict['iris_ellipse_norm'] = -1*np.ones((5, ))
            data_dict['pupil_center_norm'] = -1*np.ones((2, ))
            data_dict['pupil_ellipse_norm'] = -1*np.ones((5, ))

        return data_dict

    def readEntry(self, idx):
        '''
        Read an individual image and all its groundtruth using partial loading
        Mask annotations. This is followed by OpenEDS definitions:
            0 -> Background
            1 -> Sclera (if available)
            2 -> Iris
            3 -> Pupil
        '''
        im_num = self.imList[idx, 0]
        set_num = self.imList[idx, 2]
        arch_num = self.imList[idx, 1]

        archStr = self.arch[arch_num]
        archName = archStr.split(':')[0]

        # Use H5 files already open for data I/O. This enables catching.
        if archName not in self.fileObjs.keys():
            self.fileObjs[archName] = h5py.File(os.path.join(self.path2data,
                                                             str(archName)+'.h5'),
                                                'r', swmr=True)
        f = self.fileObjs[archName]

        # Read information
        image = f['Images'][im_num, ...]

        # Get pupil center
        if f['pupil_loc'].__len__() != 0:
            pupil_center = f['pupil_loc'][im_num, ...]
            pupil_center_available = True
        else:
            pupil_center_available = False
            pupil_center = -np.ones(2, )

        # Get mask without skin
        if f['Masks_noSkin'].__len__() != 0:
            mask_noSkin = f['Masks_noSkin'][im_num, ...]
            mask_available = True
            any_pupil = np.any(mask_noSkin == 3)
            any_iris = np.any(mask_noSkin == 2)
            if not (any_pupil and any_iris):
                # atleast one pixel must belong to all classes
                mask_noSkin = -np.ones(image.shape[:2])
                mask_available = False
        else:
            mask_noSkin = -np.ones(image.shape[:2])
            mask_available = False

        # Pupil ellipse parameters
        if f['Fits']['pupil'].__len__() != 0:
            pupil_ellipse_available = True
            pupil_param = f['Fits']['pupil'][im_num, ...]
        else:
            pupil_ellipse_available = False
            pupil_param = -np.ones(5, )

        # Iris ellipse parameters
        if f['Fits']['iris'].__len__() != 0:
            iris_ellipse_available = True
            iris_param = f['Fits']['iris'][im_num, ...]
        else:
            iris_ellipse_available = False
            iris_param = -np.ones(5, )

        data_dict = {}
        data_dict['mask'] = mask_noSkin
        data_dict['image'] = image
        data_dict['ds_num'] = set_num
        data_dict['pupil_center'] = pupil_center.astype(np.float32)
        data_dict['iris_ellipse'] = iris_param.astype(np.float32)
        data_dict['pupil_ellipse'] = pupil_param.astype(np.float32)

        data_dict['is_bad'] = False

        # Extra check to not return bad batches
        if (np.any(data_dict['mask']<0) or np.any(data_dict['mask']>3)) and mask_available:
            # This is a basic sanity check and should never be triggered
            # unless a freak accident caused something to change
            data_dict['is_bad'] = True

        # Ability to traceback
        data_dict['im_num'] = im_num
        data_dict['archName'] = archName

        # Keep flags as separate entries
        data_dict['mask_available'] = mask_available
        data_dict['pupil_center_available'] = pupil_center_available \
            if not np.all(pupil_center == -1) else False
        data_dict['iris_ellipse_available'] = iris_ellipse_available\
            if not np.all(iris_param == -1) else False
        data_dict['pupil_ellipse_available'] = pupil_ellipse_available\
            if not np.all(pupil_param == -1) else False
        return data_dict


def listDatasets(AllDS):
    dataset_list = np.unique(AllDS['dataset'])
    subset_list = np.unique(AllDS['subset'])
    return (dataset_list, subset_list)


def readArchives(path2arc_keys):
    D = os.listdir(path2arc_keys)
    AllDS = {'archive': [], 'dataset': [], 'subset': [], 'subject_id': [],
             'im_num': [], 'pupil_loc': [], 'iris_loc': []}

    for chunk in D:

        # Load archive key
        chunkData = scio.loadmat(os.path.join(path2arc_keys, chunk))
        N = np.size(chunkData['archive'])
        pupil_loc = chunkData['pupil_loc']
        subject_id = chunkData['subject_id']

        if not chunkData['subset']:
            print('{} does not have subsets.'.format(chunkData['dataset']))
            chunkData['subset'] = 'none'

        if type(pupil_loc) is list:
            # Replace pupil locations with -1
            print('{} does not have pupil center locations'.format(chunkData['dataset']))
            pupil_loc = -1*np.ones((N, 2))

        if chunkData['Fits']['iris'][0, 0].size == 0:
            # Replace iris locations with -1
            print('{} does not have iris center locations'.format(chunkData['dataset']))
            iris_loc = -1*np.ones((N, 2))
        else:
            iris_loc = chunkData['Fits']['iris'][0, 0][:, :2]

        loc = np.arange(0, N)
        res = np.flip(chunkData['resolution'], axis=1)  # Flip the resolution to [W, H]

        AllDS['im_num'].append(loc)
        AllDS['subset'].append(np.repeat(chunkData['subset'], N))
        AllDS['dataset'].append(np.repeat(chunkData['dataset'], N))
        AllDS['archive'].append(chunkData['archive'].reshape(-1)[loc])
        AllDS['iris_loc'].append(iris_loc[loc, :]/res[loc, :])
        AllDS['pupil_loc'].append(pupil_loc[loc, :]/res[loc, :])
        AllDS['subject_id'].append(subject_id)

    # Concat all entries into one giant list
    for key, val in AllDS.items():
        AllDS[key] = np.concatenate(val, axis=0)
    return AllDS


def rmDataset(AllDS, rmSet):
    '''
    Remove datasets.
    '''
    dsData = copy.deepcopy(AllDS)
    dataset_list = listDatasets(dsData)[0]
    loc = [True if simple_string(ele) in simple_string(rmSet)
           else False for ele in dataset_list]
    rmIdx = np.where(loc)[0]
    for i in rmIdx:
        loc = dsData['dataset'] == dataset_list[i]
        dsData = copy.deepcopy(rmEntries(dsData, loc))
    return dsData


def rmSubset(AllDS, rmSet):
    '''
    Remove subsets.
    '''
    dsData = copy.deepcopy(AllDS)
    dataset_list = listDatasets(dsData)[0]
    loc = [True if simple_string(ele) in simple_string(rmSet)
           else False for ele in dataset_list]
    rmIdx = np.where(loc)[0]
    for i in rmIdx:
        loc = dsData['subset'] == dataset_list[i]
        dsData = copy.deepcopy(rmEntries(dsData, loc))
    return dsData


def selDataset(AllDS, selSet):
    '''
    Select datasets of interest.
    '''
    dsData = copy.deepcopy(AllDS)
    dataset_list = listDatasets(dsData)[0]
    loc = [False if simple_string(ele) in simple_string(selSet)
           else True for ele in dataset_list]
    rmIdx = np.where(loc)[0]
    for i in rmIdx:
        loc = dsData['dataset'] == dataset_list[i]
        dsData = copy.deepcopy(rmEntries(dsData, loc))
    return dsData


def selSubset(AllDS, selSubset):
    '''
    Select subsets of interest.
    '''
    dsData = copy.deepcopy(AllDS)
    subset_list = listDatasets(dsData)[1]
    loc = [False if simple_string(ele) in simple_string(selSubset)
           else True for ele in subset_list]
    rmIdx = np.where(loc)[0]
    for i in rmIdx:
        loc = dsData['subset'] == subset_list[i]
        dsData = copy.deepcopy(rmEntries(dsData, loc))
    return dsData


def rmEntries(AllDS, ent):
    dsData = copy.deepcopy(AllDS)
    dsData['subject_id'] = AllDS['subject_id'][~ent, ]
    dsData['pupil_loc'] = AllDS['pupil_loc'][~ent, :]
    dsData['iris_loc'] = AllDS['iris_loc'][~ent, :]
    dsData['archive'] = AllDS['archive'][~ent, ]
    dsData['dataset'] = AllDS['dataset'][~ent, ]
    dsData['im_num'] = AllDS['im_num'][~ent, ]
    dsData['subset'] = AllDS['subset'][~ent, ]
    return dsData


def generate_strat_indices(AllDS):
    '''
    Removing images with pupil center values which are 10% near borders.
    Does not remove images with a negative pupil center.
    Returns the indices and a pruned data record.
    '''
    loc_oBounds = (AllDS['pupil_loc'] < 0.10) | (AllDS['pupil_loc'] > 0.90)
    loc_oBounds = np.any(loc_oBounds, axis=1)
    loc_nExist = np.any(AllDS['pupil_loc'] < 0, axis=1)
    loc = loc_oBounds & ~loc_nExist  # Location of images to remove
    AllDS = rmEntries(AllDS, loc)

    # Get ellipse centers, in case pupil is missing use the iris centers
    loc_nExist = np.any(AllDS['pupil_loc'] < 0, axis=1)
    ellipse_centers = AllDS['pupil_loc']
    ellipse_centers[loc_nExist, :] = AllDS['iris_loc'][loc_nExist, :]

    # Generate 2D histogram of pupil centers
    numBins = 5
    _, edgeList = np.histogramdd(ellipse_centers, bins=numBins)
    xEdges, yEdges = edgeList

    archNum = np.unique(AllDS['archive'],
                        return_index=True,
                        return_inverse=True)[2]

    # Bin the pupil center location and return that bin ID
    binx = np.digitize(ellipse_centers[:, 0], xEdges, right=True)
    biny = np.digitize(ellipse_centers[:, 1], yEdges, right=True)

    # Convert 2D bin locations into indices
    indx = np.ravel_multi_index((binx, biny, archNum),
                                (numBins+1, numBins+1, np.max(archNum)+1))
    indx = indx - np.min(indx)

    # Remove entries which occupy a single element in the grid
    print('Original # of entries: {}'.format(np.size(binx)))
    countInfo = np.unique(indx, return_counts=True)

    for rmInd in np.nditer(countInfo[0][countInfo[1] <= 2]):
        ent = indx == rmInd
        indx = indx[~ent]
        AllDS = copy.deepcopy(rmEntries(AllDS, ent))
    print('# of entries after stratification: {}'.format(np.size(indx)))
    return indx, AllDS


def generate_fileList(AllDS, mode='vanilla', notest=True):
    indx, AllDS = generate_strat_indices(AllDS) # This function removes samples with pupil center close to edges

    subject_identifier = list(map(lambda x,y:x+':'+y, AllDS['archive'], AllDS['subject_id']))

    archNum = np.unique(subject_identifier,
                        return_index=True,
                        return_inverse=True)[2]

    feats = np.stack([AllDS['im_num'], archNum, indx], axis=1)
    validPerc = .20

    if 'vanilla' in mode:
        # vanilla splits from the selected datasets.
        # Stratification by pupil center and dataset.
        params = re.findall('\d+', mode)
        if len(params) == 1:
            trainPerc = float(params[0])/100
            print('Training data set to {}%. Validation data set to {}%.'.format(
                        int(100*trainPerc), int(100*validPerc)))
        else:
            trainPerc = 1 - validPerc
            print('Training data set to {}%. Validation data set to {}%.'.format(
                        int(100*trainPerc), int(100*validPerc)))

        data_div = Datasplit(1, subject_identifier)

        if not notest:
            # Split into train and test
            train_feats, test_feats = train_test_split(feats,
                                                       train_size = trainPerc,
                                                       stratify = indx)
        else:
            # Do not split into train and test
            train_feats = feats
            test_feats = []

        # Split training further into validation
        train_feats, valid_feats = train_test_split(train_feats,
                                                    test_size = 0.2,
                                                    stratify = train_feats[:, -1])
        data_div.assignIdx(0, train_feats, valid_feats, test_feats)

    if 'fold' in mode:
        # K fold validation.
        K = int(re.findall('\d+', mode)[0])

        data_div = Datasplit(K, subject_identifier)
        skf = StratifiedKFold(n_splits=K, shuffle=True)
        train_feats, test_feats = train_test_split(feats,
                                                   train_size = 1 - validPerc,
                                                   stratify = indx)
        i=0
        for train_loc, valid_loc in skf.split(train_feats, train_feats[:, -1]):
            data_div.assignIdx(i, train_feats[train_loc, :],
                               train_feats[valid_loc, :],
                               test_feats)
            i+=1

    if 'none' in mode:
        # No splits. All images are placed in train, valid and test.
        # This process ensure's there no confusion.
        data_div = Datasplit(1, subject_identifier)
        data_div.assignIdx(0, feats, feats, feats)

    return data_div


def generateIdx(samplesList, batch_size):
    '''
    Takes in 2D array <samplesList>
    samplesList: 1'st dimension image number
    samplesList: 2'nd dimension hf5 file number
    batch_size: Number of images to be present in a batch
    If no entries are found, generateIdx will return an empty list of batches
    '''
    if np.size(samplesList) > 0:
        num_samples = samplesList.shape[0]
        num_batches = np.ceil(num_samples/batch_size).astype(np.int)
        np.random.shuffle(samplesList) # random.shuffle works on the first axis
        batchIdx_list = []
        for i in range(0, num_batches):
            y = (i+1)*batch_size if (i+1)*batch_size<num_samples else num_samples
            batchIdx_list.append(samplesList[i*batch_size:y, :])
    else:
        batchIdx_list = []
    return batchIdx_list

def foldInfo():
    D = {'train_idx': [], 'valid_idx': [], 'test_idx': []}
    return D

class Datasplit():
    def __init__(self, K, archs):
        self.splits = K
        self.folds = [foldInfo() for i in range(0, self.splits)]
        self.arch = np.unique(archs)

    def assignIdx(self, foldNum, train_idx, valid_idx, test_idx):
        # train, valid and test idx contains image number, h5 file and stratify index
        self.checkUnique(train_idx)
        self.checkUnique(valid_idx)
        self.checkUnique(test_idx)

        self.folds[foldNum]['train_idx'] = train_idx[:, :2] if type(train_idx) is not list else []
        self.folds[foldNum]['valid_idx'] = valid_idx[:, :2] if type(valid_idx) is not list else []
        self.folds[foldNum]['test_idx'] = test_idx[:, :2] if type(test_idx) is not list else []

    def checkUnique(self, ID):
        if type(ID) is not list:
            imNums = ID[:, 0]
            chunks = ID[:, 1]
            chunks_present = np.unique(chunks)
            for chunk in chunks_present:
                loc = chunks == chunk
                unq_flg = np.size(np.unique(imNums[loc])) != np.size(imNums[loc])
                if unq_flg:
                    print('Not unique! WARNING')

if __name__=="__main__":
    # This scripts verifies all datasets and returns the total number of images
    # Run sandbox.py to verify dataloader.
    path2data = '/media/rakshit/Monster/Datasets'
    path2arc_keys = os.path.join(path2data, 'MasterKey')

    AllDS = readArchives(path2arc_keys)
    datasets_present, subsets_present = listDatasets(AllDS)

    print('Datasets selected ---------')
    print(datasets_present)
    print('Subsets selected ---------')
    print(subsets_present)

    dataDiv_Obj = generate_fileList(AllDS, mode='vanilla')
    N = [value.shape[0] for key, value in dataDiv_Obj.folds[0].items() if len(value) > 0]
    print('Total number of images: {}'.format(np.sum(N)))

    with open('CurCheck.pkl', 'wb') as fid:
        pickle.dump(dataDiv_Obj, fid)
