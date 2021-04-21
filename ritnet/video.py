# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:36:02 2020

@author: Kevin Barkevich
"""
import torch
import numpy as np
from PIL import Image
import os
import cv2
from opt import parse_args
from models import model_dict, model_channel_dict
import matplotlib.pyplot as plt
from image import get_mask_from_PIL_image, process_PIL_image, get_area_perimiters_from_mask, get_polsby_popper_score, get_pupil_ellipse_from_PIL_image
import asyncio
import math
import datetime
import json
from graph import print_stats

from helperfunctions import get_pupil_parameters, ellipse_area, ellipse_circumference
from Ellseg.pytorchtools import load_from_file
from Ellseg.utils import get_nparams, get_predictions
from Ellseg.args import parse_precision

# INITIAL LOADING OF ARGS
args = parse_args()
filename = args.load
if not os.path.exists(filename):
    print("model path not found !!!")
    exit(1)
MODEL_DICT_STR, CHANNELS, IS_ELLSEG, ELLSEG_MODEL = model_channel_dict[filename]
ELLSEG_FOLDER = 'Ellseg'
ELLSEG_FILEPATH = './'+ELLSEG_FOLDER
ELLSEG_PRECISION = 32  # precision. 16, 32, 64

ELLSEG_PRECISION = parse_precision(ELLSEG_PRECISION)

# SETTINGS
FPS = None#10
ROTATION = 0
THREADED = False
SEPARATE_ORIGINAL_VIDEO = False
SAVE_SEPARATED_PP_FRAMES = False  # Setting enables Polsby-Popper scoring, which slows down processing
SHOW_PP_OVERLAY = False  # Setting enables Polsby-Popper scoring, which slows down processing
SHOW_PP_GRAPH = False  # Setting enables Polsby-Popper scoring, which slows down processing
OUTPUT_PP_DATA_TO_JSON = False  # Setting enables Polsby-Popper scoring, which slows down processing
OVERLAP_MASK = True
SHOW_ELLIPSE_FIT = False
KEEP_BIGGEST_PUPIL_BLOB_ONLY = True
OUTPUT_TIME_MEASUREMENTS = False
START_FRAME = 0
END_FRAME = None
USE_ELLSEG_ELLIPSE_AS_MASK = True  # If ellseg is the model, make the mask a perfect ellipse around its found pupil and iris
PAD = (0, 80)  #None # Tuple of two integers representing even-numbered horizontal padding and vertical padding or None 432x576
RESIZE = None#(320, 240)  # Tuple of two integers or None - preferrably 320x240 for ellseg
ISOLATE_FRAMES = [21,216,551,741,846,6529,9336,10081,13729,13816,14581]  # Set to save independent frames of the output into a dedicated folder, for easy mass-data-gathering.

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
            img, center, axes, angle,
            startAngle, endAngle, color,
            thickness, lineType, shift)
    except:
        return None


def main():
    if args.model not in model_dict:
        print ("Model not found !!!")
        print ("valid models are:",list(model_dict.keys()))
        exit(1)

    if args.useGPU:
        device=torch.device("cuda")
    else:
        device=torch.device("cpu")
    
    if torch.cuda.device_count() > 1:
        print('Moving to a multiGPU setup for Ellseg.')
        args.useMultiGPU = True
    else:
        args.useMultiGPU = False
        
    model = model_dict[MODEL_DICT_STR]
    
    if not IS_ELLSEG:
        model  = model.to(device)
        model.load_state_dict(torch.load(filename))
        model = model.to(device)
        model.eval()
    else:
        LOGDIR = os.path.join(os.getcwd(), ELLSEG_FOLDER, 'ExpData', 'logs',\
                          'ritnet_v2', ELLSEG_MODEL)
        path2checkpoint = os.path.join(LOGDIR, 'checkpoints')
        checkpointfile = os.path.join(path2checkpoint, 'checkpoint.pt')
        netDict = load_from_file([checkpointfile, ''])
        #print(checkpointfile)
        #print(netDict)
        model.load_state_dict(netDict['state_dict'])
        #print('Parameters: {}'.format(get_nparams(model)))
        model = model if not args.useMultiGPU else torch.nn.DataParallel(model)
        model = model.to(device)
        model = model.to(ELLSEG_PRECISION)
        model.eval()
    
    if not os.path.exists(args.video):
        print("input video not found!")
        exit(1)
    
    video = cv2.VideoCapture(args.video)
    fps = video.get(cv2.CAP_PROP_FPS) if FPS is None else FPS
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if PAD is not None:
        width, height = (width + PAD[0], height + PAD[1])
    if RESIZE is not None:
        width, height = RESIZE
    fourcc = cv2.VideoWriter_fourcc(*"X264")
    os.makedirs('video/images/',exist_ok=True)
    os.makedirs('video/outputs/',exist_ok=True)
    os.makedirs('video/pp-separation/0.'+str(0)+"-0."+str(1),exist_ok=True)
    os.makedirs('video/pp-diff-separation/0.'+str(0)+"-0."+str(1),exist_ok=True)
    for i in range(1, 9):
        os.makedirs('video/pp-separation/0.'+str(i)+"-0."+str((i+1)),exist_ok=True)
        os.makedirs('video/pp-diff-separation/0.'+str(i)+"-0."+str((i+1)),exist_ok=True)
    os.makedirs('video/pp-separation/0.'+str(9)+"-"+str(1)+".0",exist_ok=True)
    os.makedirs('video/pp-diff-separation/0.'+str(9)+"-"+str(1)+".0",exist_ok=True)
    os.makedirs('video/outputs/',exist_ok=True)
    os.makedirs('video/isolated/', exist_ok=True)
    
    mult = 2
    if OVERLAP_MASK:
        mult = 2
    if PAD and width == 192 and height == 192:
        videowriter = cv2.VideoWriter("video/outputs/out.mp4", fourcc, fps, (int(width*mult+(64*mult)),int(height*2)))
    elif PAD and width == 400 and height == 400:
        videowriter = cv2.VideoWriter("video/outputs/out.mp4", fourcc, fps, (int(width*mult+(133*mult)),int(height*2)))
    else:
        videowriter = cv2.VideoWriter("video/outputs/out.mp4", fourcc, fps, (int(width*mult),int(height)))
    # maskvideowriter = cv2.VideoWriter("video/mask.mp4", fourcc, fps, (int(width),int(height)))
    while not video.isOpened():
        video = cv2.VideoCapture(args.video)
        cv2.waitKey(1000)
        print("Wait for the header")
    
    pos_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
    
    # GAMMA CORRECTION STEP
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))  # EDIT NUMBERS HERE FOR POSSIBLE BETTTER LOW-LIGHT PERFORMANCE
    table = 255.0*(np.linspace(0, 1, 256)**0.6)  # CHANGE 0.8 TO 0.6 FOR THE DARKER VIDEO
    
    count = 0

    def get_stretched_combine(frame, pad):
        frame1 = cv2.copyMakeBorder(
                    frame,
                    top=0,
                    bottom=0,
                    left=int(pad),
                    right=int(pad),
                    borderType=cv2.BORDER_CONSTANT,
                    value=(0,0,0)
            )
        # Perform the rotation
        (h, w) = frame1.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, -25, 1.0)
        frame1 = cv2.warpAffine(frame1, M, (w, h))
        pred_img = get_mask_from_PIL_image(frame1, model, True, False)

        inp = process_PIL_image(frame1, False, clahe, table).squeeze() * 0.5 + 0.5

        img_orig = np.clip(inp,0,1)
        img_orig = np.array(img_orig)
        stretchedcombine = np.hstack([img_orig,get_mask_from_PIL_image(frame1, model, True, False),pred_img])
        return stretchedcombine
    
    max_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    pp_x = []
    pp_iris_y = []
    pp_pupil_y = []
    pp_pupil_diff_y = []
    iou_y = []
    pp_data = {}
    seconds_arr = []
    while True and (END_FRAME is None or count < END_FRAME):
        seconds_start = datetime.datetime.now()  # Start timer
        flag, frame = video.read()
        if flag:
            print()
            count += 1
            if count < START_FRAME:
                continue
            pp_x.append(count)
            # cv2.imshow('video', frame)
            # cv2.imshow('output', output[0][0].cpu().detach().numpy()/CHANNELS)
            # cv2.imshow('mask', predict[0].cpu().numpy()/CHANNELS)
            pos_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
            
            t = datetime.datetime.now()  # Time measurement - preprocessing time start
            
            if PAD is not None:
                frame = np.pad(frame, pad_width=((int(PAD[1]/2), int(PAD[1]/2)), (int(PAD[0]/2), int(PAD[0]/2)), (0, 0)), mode='constant', constant_values=0)
            
            #print(np.unique(frame))
            if RESIZE is not None:
                frame = Image.fromarray(frame)
                frame.thumbnail(RESIZE, Image.ANTIALIAS)
                frame = np.array(frame)
            #print(type(frame))
            
            # ---------------------------------------------------
            
            # Perform the rotation
            (h, w) = frame.shape[:2]
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, ROTATION, 1.0)
            frame = cv2.warpAffine(frame, M, (w, h))
            
            time_preprocessing = datetime.datetime.now() - t  # Time measurement - preprocessing time finish
            t = datetime.datetime.now()  # Time measurement - masking time start
            
            pred_img, predict = get_mask_from_PIL_image(frame, model, True, False, True, CHANNELS, KEEP_BIGGEST_PUPIL_BLOB_ONLY, isEllseg=IS_ELLSEG, ellsegPrecision=ELLSEG_PRECISION, useEllsegEllipseAsMask=USE_ELLSEG_ELLIPSE_AS_MASK)
            
            time_masking = datetime.datetime.now() - t  # Time measurement - masking time finish
            time_ellipsefit = 0  # Time measurement - ellipse fitting time init
            time_ellipsemetrics = 0  # Time measurement - ellipse metrics time init
            
            IOU = 0
            ellimage = None
            # Calculate PP score data only if PP score is used
            if SAVE_SEPARATED_PP_FRAMES or SHOW_PP_OVERLAY or SHOW_PP_GRAPH or OUTPUT_PP_DATA_TO_JSON:
                t = datetime.datetime.now()  # Time measurement - ellipse fitting time start
                
                # Scoring step 1: Get area/perimeter directly from mask
                if CHANNELS == 4:
                    iris_perimeter, pupil_perimeter, iris_area, pupil_area = get_area_perimiters_from_mask(pred_img, iris_thresh=0.6, pupil_thresh=0.1)
                else:
                    iris_perimeter, pupil_perimeter, iris_area, pupil_area = get_area_perimiters_from_mask(pred_img)
                # Scoring step 2: Get pupil & iris scores from area/perimeter
                pp_iris = get_polsby_popper_score(iris_perimeter, iris_area)
                pp_pupil = get_polsby_popper_score(pupil_perimeter, pupil_area)
                pp_pupil_diff = 0
                # Scoring step 3: Get ellipse from mask
                #params_get = predict[0].numpy()/CHANNELS
                params_get = 1 - pred_img#predict[0].numpy()/CHANNELS
                params_get[params_get <= 0.9] = 0
                params_get[params_get > 0.9] = 0.75
                pupil_ellipse = get_pupil_parameters(1-params_get)
                
                time_ellipsefit = datetime.datetime.now() - t  # Time measurement - ellipse fitting time finish
                
                pupil_mask_only_pixels = 0
                pupil_ellipse_only_pixels = 0
                pupil_overlap_pixels = 0
                major_axis = 0
                minor_axis = 0
                center_coordinates = (0, 0)
                angle = 0
                t = datetime.datetime.now()  # Time measurement - ellipse metrics time start
                
                if pupil_ellipse is not None and pupil_ellipse[0] >= -0:
                    center_coordinates = (int(pupil_ellipse[0]), int(pupil_ellipse[1]))
                    #axesLength = (int(pupil_ellipse[2])+2, int(pupil_ellipse[3])+2)
                    axesLength = (int(pupil_ellipse[2]), int(pupil_ellipse[3]))
                    angle = pupil_ellipse[4]*180/(np.pi)
                    #print("angle: ",angle)
                    startAngle = 0
                    endAngle = 360
                    color = (0, 0, 255)
                    
                    ellimage = np.zeros((int(height), int(width), 3), dtype="uint8")
                    ellimage = draw_ellipse(ellimage, center_coordinates, axesLength,
                                            angle, startAngle, endAngle, color, -1)
                    test = np.where(ellimage < 128)
                    ellimage[test] = 0
                    
                    image_copy = np.zeros((int(height), int(width), 3), dtype = "uint8")
                    
                    pupilimage = np.where(pred_img == 0)
                    image_copy[pupilimage] =  [0, 255, 0]
                    ellimage = (ellimage + image_copy) if ellimage is not None else ellimage
                    #print(~np.all(ellimage == [0,0,0], axis=2))
                    pupil_mask_only_pixels = np.sum(np.all(ellimage == [0, 255, 0], axis=2)) if ellimage is not None else 0
                    pupil_ellipse_only_pixels = np.sum(np.all(ellimage == [0, 0, 255], axis=2)) if ellimage is not None else 0
                    pupil_overlap_pixels = np.sum(np.all(ellimage == [0, 255, 255], axis=2)) if ellimage is not None else 0
                    
                    #if SHOW_ELLIPSE_FIT:
                    #    cv2.imshow("ELLIPSE2", ellimage)
                    
                    intersection = np.sum(np.all(ellimage == [0, 255, 255], axis=2)) if ellimage is not None else 0
                    union = np.sum(~np.all(ellimage == [0, 0, 0], axis=2)) if ellimage is not None else 0
                    if union != 0:
                        IOU = intersection/union
                    
                    major_axis = pupil_ellipse[2]
                    minor_axis = pupil_ellipse[3]
                    pupil_ellipse_area = ellipse_area(major_axis, minor_axis)
                    # print(pupil_ellipse_area)
                    pupil_ellipse_perimeter = ellipse_circumference(major_axis, minor_axis)
                    # Scoring step 4: Get pupil ellipse area/perimeter
                    pp_pupil_ellipse = get_polsby_popper_score(pupil_ellipse_perimeter, pupil_ellipse_area)
                    if math.isnan(pp_pupil) or pp_pupil >= 1 or pp_pupil <= 0:
                        pp_pupil_diff = 0
                    else:
                        pp_pupil_diff = abs(pp_pupil - pp_pupil_ellipse)
                else:
                    pp_pupil = 0
                if math.isnan(pp_pupil) or pp_pupil >= 1 or pp_pupil <= 0:
                    pp_pupil = 0
                    
                if math.isnan(pp_iris) or pp_iris >= 1 or pp_iris <= 0:
                    if len(pp_iris_y) > 0:
                        pp_iris = pp_iris_y[len(pp_iris_y)-1]
                    else:
                        pp_iris = 0
                
                time_ellipsemetrics = datetime.datetime.now() - t  # Time measurement - ellipse metrics time finish
                
                pp_iris_y.append(pp_iris)
                pp_pupil_y.append(pp_pupil)
                pp_pupil_diff_y.append(pp_pupil_diff)
                iou_y.append(IOU)
                if OUTPUT_PP_DATA_TO_JSON:
                    pp_data[count] = {
                            'pp': pp_pupil,
                            'pp_diff': pp_pupil_diff,
                            'shape_conf': IOU,
                            'pupil_ellipse_fit': {
                                "pixels": {
                                    "mask_only_pixels": int(pupil_mask_only_pixels),
                                    "ellipse_only_pixels": int(pupil_ellipse_only_pixels),
                                    "overlap_pixels": int(pupil_overlap_pixels),
                                },
                                "major_axis": major_axis,
                                "minor_axis": minor_axis,
                                "center_x": center_coordinates[0],
                                "center_y": center_coordinates[1],
                                "angle": angle
                            }
                    }
            
            t = datetime.datetime.now()  # Time measurement - graphing time start
            
            if SHOW_PP_GRAPH:
                plt.title("Pupil Polsby-Popper Score")
                plt.xlabel("frame")
                plt.ylabel("score")
                plt.plot(pp_x, pp_pupil_y, color='olive', label="Image Score")
                plt.plot(pp_x, pp_pupil_diff_y, color='blue', label="Difference Image Score, Ellipse Score")
                plt.plot(pp_x, iou_y, color='red', label="IOU Pupil Ellipse Fit & Pupil Mask")
                plt.ylim(bottom=0, top=1)
                plt.legend()
                plt.show()
            
            time_graphing = datetime.datetime.now() - t  # Time measurement - graphing time finish
            t = datetime.datetime.now()  # Time measurement - overlay generation time start
            
            # Add score overlay to image
            if SHOW_PP_OVERLAY:
                font = cv2.FONT_HERSHEY_SIMPLEX
                orgPP = (10, 15)
                orgPPDiff = (10, 35)
                orgIouDiff = (10, 55)
                fontScale = 0.5
                colorWhite = (255, 255, 255)
                colorBlack= (0, 0, 0)
                thickness = 2
                frame = cv2.putText(frame, "PP:           "+"{:.4f}".format(pp_pupil), orgPP, font, fontScale,
                                    colorBlack, thickness*2, cv2.LINE_AA)
                frame = cv2.putText(frame, "PP:           "+"{:.4f}".format(pp_pupil), orgPP, font, fontScale,
                                    colorWhite, thickness, cv2.LINE_AA)
                frame = cv2.putText(frame, "PP Diff:       "+"{:.4f}".format(pp_pupil_diff), orgPPDiff, font, fontScale,
                                    colorBlack, thickness*2, cv2.LINE_AA)
                frame = cv2.putText(frame, "PP Diff:       "+"{:.4f}".format(pp_pupil_diff), orgPPDiff, font, fontScale,
                                    colorWhite, thickness, cv2.LINE_AA)
                frame = cv2.putText(frame, "Shape Conf.:  "+"{:.4f}".format(IOU), orgIouDiff, font, fontScale,
                                    colorBlack, thickness*2, cv2.LINE_AA)
                frame = cv2.putText(frame, "Shape Conf.:  "+"{:.4f}".format(IOU), orgIouDiff, font, fontScale,
                                    colorWhite, thickness, cv2.LINE_AA)
            
            time_overlay = datetime.datetime.now() - t  # Time measurement - overlay generation time finish
            t = datetime.datetime.now()  # Time measurement - output image generation time start
            
            inp = process_PIL_image(frame, False, clahe, table).squeeze() * 0.5 + 0.5
            img_orig = np.clip(inp,0,1)
            img_orig = np.array(img_orig)
            if OVERLAP_MASK:
                #outtable = np.linspace(0, 1, 256)**0.9  # CHANGE 0.8 TO 0.6 FOR THE DARKER VIDEO
                #out_img_orig = cv2.LUT(np.uint8(np.array(img_orig)), outtable)
                clahe_param = cv2.createCLAHE(clipLimit=2, tileGridSize=(2,2))
                img_orig = clahe_param.apply(np.array(np.uint8(img_orig*255)))/255
                table = 0.5
                mean = np.mean(img_orig)
                gamma = math.log(table)/math.log(mean)
                img_gamma1 = np.power(img_orig, gamma).clip(0,1)
                combine = np.hstack([img_orig, (img_gamma1 + (1-pred_img)) / 2])
            else:
                combine = np.hstack([img_orig,pred_img])

                
            time_outputgeneration = datetime.datetime.now() - t  # Time measurement - output image generation time finish
            t = datetime.datetime.now()  # Time measurement - output display time start
                
            if SHOW_ELLIPSE_FIT and ellimage is not None:
                cv2.imshow("ELLIPSE", ellimage)
            cv2.imshow('RITnet', cv2.resize(combine, (1920, 1080)))
            if SEPARATE_ORIGINAL_VIDEO:
                cv2.imshow('Original', img_orig)
                
            time_outputdisplay = datetime.datetime.now() - t  # Time measurement - output display finish
            t = datetime.datetime.now()  # Time measurement - output image saving time start
                
            if SAVE_SEPARATED_PP_FRAMES:
                pp_folder = "{}-{}".format(str(round(int(math.floor(pp_pupil * 10.0)) / 10, 1)), str(round(int(math.floor(pp_pupil * 10.0)) / 10 + .10, 1)))
                pp_diff_folder = "{}-{}".format(str(round(int(math.floor(pp_pupil_diff * 10.0)) / 10, 1)), str(round(int(math.floor(pp_pupil_diff * 10.0)) / 10 + .10, 1)))
                plt.imsave('video/pp-separation/{}/{}.png'.format(pp_folder, str(count)), combine)
                plt.imsave('video/pp-diff-separation/{}/{}.png'.format(pp_diff_folder, str(count)), combine)
                if count in ISOLATE_FRAMES:
                    plt.imsave('video/isolated/{}.png'.format(str(count)), combine)
            pred_img_3=np.zeros((pred_img.shape[0],pred_img.shape[1],3))
            pred_img_3[:,:,0]=pred_img
            pred_img_3[:,:,1]=pred_img
            pred_img_3[:,:,2]=pred_img
            plt.imsave('video/images/{}.png'.format(count),np.uint8(pred_img_3 * 255))
            
            time_outputimage = datetime.datetime.now() - t  # Time measurement - output image saving time finish
            t = datetime.datetime.now()  # Time measurement - output video frame time start
            
            # maskvideowriter.write((pred_img * 255).astype('uint8'))  # write to mask video output
            videowriter.write((combine * 255).astype('uint8')) # write to video output
            
            time_output_video = datetime.datetime.now() - t  # Time measurement - output video frame time finish
            
            # Time measurement - display results for frame
            if OUTPUT_TIME_MEASUREMENTS:
                def sortTime(val):
                    return val[1]
                time_arr = [("preprocessing      ", time_preprocessing.microseconds),
                ("masking            ", time_masking.microseconds),
                ("ellipse fit        ", time_ellipsefit.microseconds),
                ("ellipse metrics    ", time_ellipsemetrics.microseconds),
                ("graphing           ", time_graphing.microseconds),
                ("overlay            ", time_overlay.microseconds),
                ("output generation  ", time_outputgeneration.microseconds),
                ("output display     ", time_outputdisplay.microseconds),
                ("output image save  ", time_outputimage.microseconds),
                ("output video frame ", time_output_video.microseconds)]
                time_arr.sort(key=sortTime, reverse=True)
                for time_entry in time_arr:
                    print(time_entry[0], ": ", time_entry[1], " microseconds")
            
            # Calculate time remaining using last (up to) 10 frames
            seconds_end = datetime.datetime.now()
            seconds_diff = seconds_end - seconds_start
            if len(seconds_arr) == 25:
                seconds_arr.pop(0)
            seconds_arr.append(seconds_diff.total_seconds())
            avg = 0
            for i in seconds_arr:
                avg += i
            avg = avg / len(seconds_arr)
            seconds_remaining = avg * (max_frames - pos_frame)
            time_remaining = "{}:{}:{}".format("{0:0=2d}".format(int(seconds_remaining/60/60)),"{0:0=2d}".format(int(seconds_remaining/60%60)),"{0:0=2d}".format(int(seconds_remaining%60)))
            print(str(pos_frame)+"/"+str(max_frames)+" frames  (ETA: " + time_remaining+")")
        else:
            # Wait for next frame
            video.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
            print("frame is not ready")
            cv2.waitKey(2)
        
        if cv2.waitKey(10) == 27:
            video.release()
            # maskvideowriter.release()
            videowriter.release()
            cv2.destroyAllWindows()
            break
        if video.get(cv2.CAP_PROP_POS_FRAMES) == video.get(cv2.CAP_PROP_FRAME_COUNT):
            video.release()
            # maskvideowriter.release()
            videowriter.release()
            cv2.destroyAllWindows()
            break
    if OUTPUT_PP_DATA_TO_JSON:
        with open('pp_data.txt', 'w') as outfile:
            json.dump(pp_data, outfile, indent=4)
        print_stats(file_name='pp_data.txt', spacing=1, frame_range=None, ignore_zeros=True)
    
    os.system('cd "'+os.path.dirname(os.path.realpath(__file__))+'" & ffmpeg -r '+str(fps)+' -i ".\\video\\images\\%d.png" -c:v mpeg4 -vcodec libx264 -r '+str(fps)+' ".\\video\\outputs\\mask.mp4"')
    

    # os.rename('test',args.save)


if __name__ == '__main__':
    if THREADED:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    else:
        main()