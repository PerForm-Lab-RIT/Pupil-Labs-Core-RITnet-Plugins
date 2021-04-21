# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 19:09:16 2020

@author: Kevin Barkevich
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torchvision
from torchvision import transforms
import numpy as np

from models import model_dict
from dataset import transform
import cv2
import matplotlib.pyplot as plt

from utils import get_predictions
from PIL import Image
from helperfunctions import get_pupil_parameters, my_ellipse
from skimage import io, measure

def draw_ellipse(
        img, center, axes, angle,
        startAngle, endAngle, color,
        thickness=3, lineType=cv2.LINE_AA, shift=10):
    center = (
        int(round(center[0] * 2**shift)),
        int(round(center[1] * 2**shift))
    )
    axes = (
        int(round(axes[0] * 2**shift)),
        int(round(axes[1] * 2**shift))
    )
    try:
        return cv2.ellipse(
            img, center, axes, angle*180/np.pi,
            startAngle, endAngle, color,
            thickness, lineType, shift)
    except:
        return None
    
def init_model(devicestr="cuda", modelname="best_model.pkl"):
    device = torch.device(devicestr)
    model = model_dict["densenet"]
    model  = model.to(device)
    filename = os.path.dirname(os.path.abspath(__file__)) + "/"+modelname;
        
    model.load_state_dict(torch.load(filename))
    model = model.to(device)
    model.eval()
    return model

def process_PIL_image(frame, do_corrections=True, clahe=None, table=None):
    if clahe is None:
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    if table is None:
        table = 255.0*(np.linspace(0, 1, 256)**0.8)
    img = Image.fromarray(frame).convert("L")
    if do_corrections:
        img = cv2.LUT(np.array(img), table)
        img = clahe.apply(np.array(np.uint8(img)))
        img = Image.fromarray(img)
    img = transform(img)
    return img

def get_mask_from_path(path: str, model, useGpu=True):
    if useGpu:
        device=torch.device("cuda")
    else:
        device=torch.device("cpu")
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    
    pilimg = Image.open(path).convert("L")
    table = 255.0*(np.linspace(0, 1, 256)**0.8)
    pilimg = cv2.LUT(np.array(pilimg), table)
    img = clahe.apply(np.array(np.uint8(pilimg)))    
    img = Image.fromarray(img)
    img = img.unsqueeze(1)
    data = img.to(device)   
    output = model(data)
    predict = get_predictions(output)
    return predict
    
def get_mask_from_cv2_image(image, model, useGpu=True, pupilOnly=False, includeRawPredict=False, channels=3, trim_pupil=False, isEllseg=False, ellsegPrecision=None, useEllsegEllipseAsMask=False):
    if useGpu:
        device=torch.device("cuda")
    else:
        device=torch.device("cpu")
    
    if not isEllseg:
        img = image.unsqueeze(1)
        data = img.to(device)   
        output = model(data)
        rawpredict = get_predictions(output)
        predict = rawpredict + 1
        # print(np.unique(predict[0].cpu().numpy()))
        pred_img = 1 - predict[0].cpu().numpy()/channels
    else:
        img = np.array(Image.fromarray(image).convert("L"))
        img = (img - img.mean())/img.std()
        img = torch.from_numpy(img).unsqueeze(0).to(ellsegPrecision)  # Adds a singleton for channels
        img = img.unsqueeze(0)
        img = img.to(device).to(ellsegPrecision)
        x4, x3, x2, x1, x = model.enc(img)
        op = model.dec(x4, x3, x2, x1, x)
        rawpredict=get_predictions(op)
        plt.imshow(rawpredict[0], cmap="BrBG", alpha=0.3)
        if useEllsegEllipseAsMask:
            ellpred = model.elReg(x, 0).view(-1)
            #i1, i2, i3, i4, i5, p1, p2, p3, p4, p5 = ellpred[0].cpu().detach().numpy()
            _, _, H, W = img.shape
            H_mat = np.array([[W/2, 0, W/2], [0, H/2, H/2], [0, 0, 1]])
            
            #import pdb
            #pdb.set_trace()
            i_cx, i_cy, i_a, i_b, i_theta, _ = my_ellipse(ellpred[:5].tolist()).transform(H_mat)[0]
            p_cx, p_cy, p_a, p_b, p_theta, _ = my_ellipse(ellpred[5:].tolist()).transform(H_mat)[0]
            
            ellimage = np.full((int(H), int(W)), 2/3)
            startAngle = 0
            endAngle = 360
            iris_color=1/3
            pupil_color = 0.0
            pred_img = draw_ellipse(ellimage, (i_cx, i_cy), (i_a, i_b),
                         i_theta, startAngle, endAngle, iris_color, -1)
            pred_img = draw_ellipse(ellimage, (p_cx, p_cy), (p_a, p_b),
                         p_theta, startAngle, endAngle, pupil_color, -1)
        else:
            predict = rawpredict + 1
            pred_img = 1 - predict[0].cpu().numpy()/channels
    
    #print(pred_img)
    # trim pupil if asked to
    if trim_pupil:
        newimg = np.invert(pred_img>0)
        labeled_img = measure.label(newimg)
        labels = np.unique(labeled_img)
        newimg = np.zeros((newimg.shape[0],newimg.shape[1]))
        old_sum = 0
        old_label = None
        for label in (y for y in labels if y != 0):
            if np.sum(labeled_img==label) > old_sum:
                old_sum = np.sum(labeled_img==label)
                old_label = label
        if old_label is not None:
            newimg = newimg + (labeled_img == old_label)
        newimg[newimg == 0] = 2
        newimg[newimg == 1] = 0
        newimg[newimg == 2] = 1
        pred_img[pred_img == 0] = 1-(1/channels)
        pred_img[newimg == 0] = 0
        
    #print(np.unique(pred_img))
    if pupilOnly:
        pred_img = np.ceil(pred_img) * 0.5
    if includeRawPredict:
        return pred_img, rawpredict
    return pred_img

def get_mask_from_PIL_image(pilimage, model, useGpu=True, pupilOnly=False, includeRawPredict=False, channels=3, trim_pupil=False, isEllseg=False, ellsegPrecision=None, useEllsegEllipseAsMask=False):
    if not isEllseg:
        img = process_PIL_image(pilimage)
    else:
        img = pilimage
    return get_mask_from_cv2_image(img, model, useGpu, pupilOnly, includeRawPredict, channels, trim_pupil, isEllseg, ellsegPrecision, useEllsegEllipseAsMask)
    
def get_pupil_ellipse_from_cv2_image(image, model, useGpu=True, predict=None, isEllseg=False, ellsegPrecision=None, ellsegEllipse=False, debugWindowName=None):
    """
    OUTPUT FORMAT
    {
        0: center x,
        1: center y,
        2: ellipse major axis radius,
        3: ellipse minor axis radius,
        4: ellipse angle
    }
    """
    if useGpu:
        device=torch.device("cuda")
    else:
        device=torch.device("cpu")
    if predict is None:
        if not isEllseg:
            img = image.unsqueeze(1)
            data = img.to(device)   
            output = model(data)
            predict = get_predictions(output)
            pred_img = predict[0].numpy()
        else:  # w:320 h:240
            img = np.array(transforms.ToPILImage()(image).convert("L"))
            img = (img - img.mean())/img.std()
            img = torch.from_numpy(img).unsqueeze(0).to(ellsegPrecision)  # Adds a singleton for channels
            img = img.unsqueeze(0)
            img = img.to(device).to(ellsegPrecision)
            x4, x3, x2, x1, x = model.enc(img)
            op = model.dec(x4, x3, x2, x1, x)
            
            if ellsegEllipse:  # Option to get ellipse directly from ellseg output
                ellpred = model.elReg(x, 0).view(-1)
                _, _, H, W = img.shape
                H_mat = np.array([[W/2, 0, W/2], [0, H/2, H/2], [0, 0, 1]])
                p_cx, p_cy, p_a, p_b, p_theta, _ = my_ellipse(ellpred[5:].tolist()).transform(H_mat)[0]
                return [p_cx, p_cy, p_a, p_b, p_theta]
                # [centerX, centerY, axis1, axis2, angle]
            
            #elOut = model.elReg(x, 0) # Linear regression to ellipse parameters
            
            #print(elOut.shape)
            
            predict=get_predictions(op)
            pred_img = predict[0].numpy()
            
            # cv2.imshow("ELLIPSE", pred_img/2)
    else:
        pred_img = predict[0].numpy()
            
    if debugWindowName is not None:
        outIm = pred_img/np.max(pred_img)
        cv2.imshow(debugWindowName, outIm)
    return get_pupil_parameters(pred_img)
    
def get_pupil_ellipse_from_PIL_image(pilimage, model, useGpu=True, predict=None, isEllseg=False, ellsegPrecision=None, ellsegEllipse=False, debugWindowName=None):
    if not isEllseg:
        img = process_PIL_image(pilimage)
    else:
        img = pilimage
    res = get_pupil_ellipse_from_cv2_image(img, model, useGpu, predict, isEllseg, ellsegPrecision, ellsegEllipse, debugWindowName)
    if res is not None:
        res[4] = res[4] * 180 / np.pi
    return res

def get_area_perimiters_from_mask(image, iris_thresh=0.9, pupil_thresh=0.1):    

    #get threshold image
    ret,thresh_img = cv2.threshold(image, iris_thresh, 255, cv2.THRESH_BINARY_INV)
    thresh_img = thresh_img.astype(np.uint8)
    iris_area = np.sum(thresh_img != 0)
    #find iris contours
    iris_image, iris_contours, iris_hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #get threshold image
    ret,thresh_img = cv2.threshold(image, pupil_thresh, 255, cv2.THRESH_BINARY_INV)
    thresh_img = thresh_img.astype(np.uint8)
    pupil_area = np.sum(thresh_img != 0)
    #find pupil contours
    pupil_image, pupil_contours, pupil_hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    iris_perimeter = 0
    pupil_perimeter = 0
    
    for c in iris_contours:
        peri = cv2.arcLength(c, True)
        iris_perimeter = iris_perimeter + peri
    for c in pupil_contours:
        peri = cv2.arcLength(c, True)
        pupil_perimeter = pupil_perimeter + peri
    
    #print(f'Perimeter = {int(round(perimeter,0))} pixels')
    
    # print("IRIS PERIMETER: " + str(iris_perimeter))
    # print("PUPIL PERIMETER: " + str(pupil_perimeter))

    return iris_perimeter, pupil_perimeter, iris_area, pupil_area

def get_polsby_popper_score(perimeter, area):
    try:
        return (4 * np.pi * area) / (np.square(perimeter))
    except:
        return 0
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    