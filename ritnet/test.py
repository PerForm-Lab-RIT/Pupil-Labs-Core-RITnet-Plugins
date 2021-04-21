#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:37:59 2019

@author: aaa
"""
import torch
from dataset import IrisDataset
from torch.utils.data import DataLoader 
import numpy as np
import matplotlib.pyplot as plt
from dataset import transform
import os
from opt import parse_args
from models import model_dict
from tqdm import tqdm
from utils import get_predictions
#%%

if __name__ == '__main__':
    
    args = parse_args()
   
    if args.model not in model_dict:
        print ("Model not found !!!")
        print ("valid models are:",list(model_dict.keys()))
        exit(1)

    if args.useGPU:
        device=torch.device("cuda")
    else:
        device=torch.device("cpu")
        
    model = model_dict[args.model]
    model  = model.to(device)
    filename = args.load
    if not os.path.exists(filename):
        print("model path not found !!!")
        exit(1)
        
    model.load_state_dict(torch.load(filename))
    model = model.to(device)
    model.eval()

    test_set = IrisDataset(filepath = 'Semantic_Segmentation_Dataset/',\
                                 split = 'test',transform = transform)
    
    testloader = DataLoader(test_set, batch_size = args.bs,
                             shuffle=False, num_workers=2)
    counter=0
    
    os.makedirs('test/labels/',exist_ok=True)
    os.makedirs('test/output/',exist_ok=True)
    os.makedirs('test/mask/',exist_ok=True)
    
    with torch.no_grad():
        # for i, batchdata in tqdm(enumerate(testloader),total=len(testloader)):
        for i in range(3):
            print (i)
            img,labels,index,x,y= test_set[i] #batchdata
            img = img.unsqueeze(1)
            data = img.to(device)   
            output = model(data)
            predict = get_predictions(output)
            
            ## --- EXPERIMENTING ---
            # print(predict)
            # if i == 1:
            #     for p in output[0]:
            #         for q in p:
            #             print(q)
                        
            # for j in range (len(index)):     
            j = 0
            np.save('test/labels/{}.npy'.format(index),predict[j].cpu().numpy())
            try:
                plt.imsave('test/output/{}.jpg'.format(index),255*labels[j].cpu().numpy())
            except:
                pass
            
            ## --- EXPERIMENTING ---
                # pred_img = output[0][0].cpu().numpy()/3.0
                
            pred_img = predict[j].cpu().numpy()/3.0
            inp = img[j].squeeze() * 0.5 + 0.5
            img_orig = np.clip(inp,0,1)
            img_orig = np.array(img_orig)
            combine = np.hstack([img_orig,pred_img])
            plt.imsave('test/mask/{}.jpg'.format(index),combine)

    # os.rename('test',args.save)
