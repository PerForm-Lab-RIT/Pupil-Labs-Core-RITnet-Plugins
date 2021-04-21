# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 18:47:52 2020

@author: Kevin Barkevich
"""

import matplotlib.pyplot as plt
import json
from os import path
import numpy as np


FILE_NAME = "pp_data.txt"
SPACING = 10
RANGE = None  # None for entire viceo
IGNORE_ZEROS = False  # Frames where no pupil at all could be detected, such as during blinks and/or before the eye comes into frame, are treated as 0.0.

def print_stats(file_name=FILE_NAME, spacing=SPACING, frame_range=RANGE, ignore_zeros=IGNORE_ZEROS):
    x = []
    y_pp = []
    y_pp_diff = []
    y_shape_conf = []
    
    with open(file_name) as json_file:
        data = json.load(json_file)
        for p in data:
            if int(p) % spacing == 0 and (frame_range is None or (int(p) >= frame_range[0] and int(p) <= frame_range[1])) and (ignore_zeros == False or float(data[p]["pp"]) != 0.0):
                x.append(int(p))
                if "pp" in data[p].keys():
                    y_pp.append(float(data[p]["pp"]))
                if "pp_diff" in data[p].keys():
                    y_pp_diff.append(float(data[p]["pp_diff"]))
                if "IOU" in data[p].keys():
                    y_shape_conf.append(float(data[p]["IOU"]))
                elif "shape_conf" in data[p].keys():
                    y_shape_conf.append(float(data[p]["shape_conf"]))
                
        plt.title("Image Pupil Scoring (every " + str(spacing) + " pixel[s])")
        plt.xlabel("frame")
        plt.ylabel("score")
        plt.plot(x, y_pp, color='olive', label="Mask PP Score")
        plt.plot(x, y_pp_diff, color='blue', label="Difference Mask PP Score, Ellipse PP Score")
        plt.plot(x, y_shape_conf, color='red', label="Pupil Shape Confidence")
        plt.ylim(bottom=0, top=1)
        plt.legend()
        plt.show()
        
        pp_first_quartile = sum((val < .25) for val in y_pp) / len(y_pp) * 100
        pp_diff_first_quartile = sum((val >= .75) for val in y_pp_diff) / len(y_pp_diff) * 100
        shape_conf_first_quartile = sum((val < .25) for val in y_shape_conf) / len(y_shape_conf) * 100
        
        pp_second_quartile = sum((val >= .25 and val < .5) for val in y_pp) / len(y_pp) * 100
        pp_diff_second_quartile = sum((val < .75 and val >= .5) for val in y_pp_diff) / len(y_pp_diff) * 100
        shape_conf_second_quartile = sum((val >= .25 and val < .5) for val in y_shape_conf) / len(y_shape_conf) * 100
        
        pp_third_quartile = sum((val >= .5 and val < .75) for val in y_pp) / len(y_pp) * 100
        pp_diff_third_quartile = sum((val < .5 and val >= .25) for val in y_pp_diff) / len(y_pp_diff) * 100
        shape_conf_third_quartile = sum((val >= .5 and val < .75) for val in y_shape_conf) / len(y_shape_conf) * 100
        
        pp_fourth_quartile = sum((val >= .75) for val in y_pp) / len(y_pp) * 100
        pp_diff_fourth_quartile = sum((val < .25) for val in y_pp_diff) / len(y_pp_diff) * 100
        shape_conf_fourth_quartile = sum((val >= .75) for val in y_shape_conf) / len(y_shape_conf) * 100
        
        pp_quartiles = [pp_first_quartile, pp_second_quartile, pp_third_quartile, pp_fourth_quartile]
        pp_diff_quartiles = [pp_diff_first_quartile, pp_diff_second_quartile, pp_diff_third_quartile, pp_diff_fourth_quartile]
        shape_conf_quartiles = [shape_conf_first_quartile, shape_conf_second_quartile, shape_conf_third_quartile, shape_conf_fourth_quartile]
        data = [pp_quartiles, pp_diff_quartiles, shape_conf_quartiles]
        X = np.arange(4)
        
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        b1 = ax.bar(X + 0.00, data[0], color = 'b', width = 0.25)
        b2 = ax.bar(X + 0.25, data[1], color = 'g', width = 0.25)
        b3 = ax.bar(X + 0.50, data[2], color = 'r', width = 0.25)
        ax.set_ylabel('Percentage (%) in Quarter')
        ax.set_title("Image Pupil Scoring by Quartile (every " + str(spacing) + " pixel[s])")
        ax.legend(labels=['PP Score', 'PP Diff Score', 'Pupil Shape Confidence'])
        plt.xticks([r + .25 for r in range(4)], 
           ['1st Quartile', '2nd Quartile', '3rd Quartile', '4th Quartile'])
        
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{:.1f}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(b1)
        autolabel(b2)
        autolabel(b3)
        plt.show()
        
            
    y_pp_copy = y_pp.copy()
    y_pp_diff_copy = y_pp_diff.copy()
    y_shape_conf_copy = y_shape_conf.copy()
    y_pp_copy.sort()
    y_pp_diff_copy.sort(reverse=True)
    y_shape_conf_copy.sort()
    
    onePercent = int(len(x)/100)
    ptOnePercent = int(len(x)/1000)
    
    print("-------------PP: Higher Is Better-------------")
    if (len(y_pp_copy) > 0):
        val = sum(y_pp_copy)/len(y_pp_copy)
        print("            PP mean: ", val)
    if (len(y_pp_copy) > 0):
        val = y_pp_copy[int(len(y_pp_copy)/2)]
        print("          PP median: ", val)
    if len(y_pp_copy) >= onePercent and onePercent != 0:
        val = sum(y_pp_copy[0:onePercent])/onePercent
        print("  PP 1% low average: ", val)
    if len(y_pp_copy) >= ptOnePercent and ptOnePercent != 0:
        val = sum(y_pp_copy[0:ptOnePercent])/ptOnePercent
        print("PP 0.1% low average: ", val)
    print('\n-------------PP Difference: Lower Is Better-------------')
    if (len(y_pp_diff_copy) > 0):
        val = sum(y_pp_diff_copy)/len(y_pp_diff_copy)
        print("            PP Diff mean: ", val)
    if (len(y_pp_diff_copy) > 0):
        val = y_pp_diff_copy[int(len(y_pp_diff_copy)/2)]
        print("          PP Diff median: ", val)
    if len(y_pp_diff_copy) >= onePercent and onePercent != 0:
        val = sum(y_pp_diff_copy[0:onePercent])/onePercent
        print("  PP Diff 1% low average: ", val)
    if len(y_pp_diff_copy) >= ptOnePercent and ptOnePercent != 0:
        val = sum(y_pp_diff_copy[0:ptOnePercent])/ptOnePercent
        print("PP Diff 0.1% low average: ", val)
    print('\n-------------Pupil Shape Confidence: Higher Is Better-------------')
    if (len(y_shape_conf_copy) > 0):
        val = sum(y_shape_conf_copy)/len(y_shape_conf_copy)
        print("            Pupil Shape Conf mean: ", val)
    if (len(y_shape_conf_copy) > 0):
        val = y_shape_conf_copy[int(len(y_shape_conf_copy)/2)]
        print("          Pupil Shape Conf median: ", val)
    if len(y_shape_conf_copy) >= onePercent and onePercent != 0:
        val = sum(y_shape_conf_copy[0:onePercent])/onePercent
        print("  Pupil Shape Conf 1% low average: ", val)
    if len(y_shape_conf_copy) >= ptOnePercent and ptOnePercent != 0:
        val = sum(y_shape_conf_copy[0:ptOnePercent])/ptOnePercent
        print("Pupil Shape Conf 0.1% low average: ", val) 
    

if __name__ == '__main__':
    print_stats()
