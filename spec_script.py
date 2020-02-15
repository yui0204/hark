# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 14:12:45 2018

@author: yui_sudo

"""

import os
import re
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def peak_detect(spec):
    angle = ""
    for i in range(len(spec) - 1):     
        if i == 0:
            if spec[0] > 55 and (spec[0] - spec[len(spec)-1] > 1):
                angle = angle + "0 "
            
        else:
            if spec[i] > 55 and (spec[i] - spec[i-1] > 3) and (spec[i] - spec[i+1] > 3):
                angle = angle + str(i * 5) + " "
            
    return angle[:-1]


classes = 75
image_size = 256

#datasets_dir = "/media/yui-sudo/Samsung_T5/dataset/sound_data/datasets/"
datasets_dir = "/home/yui-sudo/document/dataset/sound_segmentation/datasets/"

datadir = "multi_segdata"+str(classes) + "_"+str(image_size)+"_no_sound_random_sep_72/"
dataset = datasets_dir + datadir    

segdata_dir = dataset + "train/"
valdata_dir = dataset + "val/"

labelfile = dataset + "label.csv"
label = pd.read_csv(filepath_or_buffer=labelfile, sep=",", index_col=0)            


with open("localization.n", "r") as f:
    loc_nfile = f.read()
error = 0

with open("sep.n", "r") as f:
    sep_nfile = f.read()

for mode in ["train_hark/", "val_hark/"]:
    if mode == "train":
        totalnum = 10000
    else:
        totalnum = 1000
        
    for i in range(0, totalnum):
        data_dir = segdata_dir + str(i) + "/"
        save_dir = dataset + mode + str(i) + "/"
        filelist = os.listdir(data_dir)  
        
        with open(data_dir + "/sound_direction.txt", "r") as f:
            direction = f.read().split("\n")[:-1]
            
        gt_angle = ""
        for n in range(len(filelist)):    
            if filelist[n][:7] == "0_multi":
                filename = filelist[n][:]
                #shutil.copy(data_dir + filename, os.getcwd())
                for k in range(len(direction)):
                    gt_angle = gt_angle + str(int(re.sub("\\D", "", direction[k].split("_")[1]))) + " "
                    
                print("No.", i)
                print(filename)
        gt_angle = gt_angle[:-1]
    
        ### localization
        loc_exefile = loc_nfile.replace('a.wav', data_dir + filename)
    
        with open('loc_exefile.n','w') as f:
            f.write(loc_exefile)
    
        os.system("./loc_exefile.n | grep MUSIC > spec.txt")
    
        spec = pd.read_csv("spec.txt", delimiter=" ", header=None) # 日本語パス不可
        spec = spec.T.iloc[2:-1]
        
        spec = np.array(spec, np.float32)
        
        #plt.pcolormesh(spec)
        #plt.pcolormesh(np.arange(0,4.1,0.1), np.arange(0,359,5), spec)
        #plt.pcolormesh(np.arange(0,4.1,0.1), np.arange(0,359,45), spec)
        #plt.colorbar()
        #plt.show()
        
        spec = spec.max(1)
        #plt.plot(spec)
        #plt.show()
        
        pred_angle = peak_detect(spec)
        print(gt_angle)
        print(pred_angle, "\n")
        
        if not len(gt_angle) == len(pred_angle) or not np.array(gt_angle.split(" "), dtype=np.int16).sum() == np.array(pred_angle.split(" "), dtype=np.int16).sum():
            error += 1
    
            
        sep_exefile = sep_nfile.replace('a.wav', data_dir + filename)
        sep_exefile = sep_exefile.replace('0 45 90 135 180 225 270 315', pred_angle)
        
        with open('sep_exefile.n','w') as f:
            f.write(sep_exefile)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        shutil.copy(data_dir + filename, save_dir + filename)
        shutil.copy(data_dir + "/sound_direction.txt", save_dir + "/sound_direction.txt")
        os.system("./sep_exefile.n")
        with open(save_dir + 'pred_direction','w') as f:
            f.write(pred_angle)
        shutil.copy("sep_0.wav", save_dir + "/sep_0.wav")
        shutil.copy("sep_1.wav", save_dir + "/sep_1.wav")
        shutil.copy("sep_2.wav", save_dir + "/sep_2.wav")    
    
    print("The number of errors = ", error)
