# -*- coding: utf-8 -*-
"""
@author: Sonic
"""
import scipy.io as sio  
import numpy as np  
import matplotlib.pyplot as plt

def DrawResult(labels,imageID):
    #ID=1:Pavia University
    #ID=2:Indian Pines    
    #ID=7:Houston
    global palette
    global row
    global col
    num_class = int(labels.max())
    if imageID == 1:
        row = 1398
        col = 942
        palette = np.array([[238,138,34],
                            [78, 238, 148],
                            [0, 255, 255],
                            [205, 170, 125],
                            [139, 139, 131],
                            [0, 139, 69],
                            [255, 215, 0],
                            [65, 105, 225],
                            [0, 0, 255],
                            [238, 99, 99],
                            [139, 0, 139],
                            [50, 205, 50],
                            [0, 191, 255],
                            [238, 238, 0],
                            [255, 69, 0],
                            [0, 139, 139]])
        palette = palette*1.0/255
    elif imageID == 2:
        row = 1147
        col = 1600
        palette = np.array([
            [50, 205, 50],
            [255, 69, 0],
            [139, 0, 139],
            [255, 0, 255],
            [255, 228, 181],
            [0, 139, 0],
            [127, 96, 190],
            [78, 238, 148],
            [0, 255, 255],
            [0, 0, 255],
            [205, 170, 125],
            [0, 139, 139],
            [255, 215, 0],
            [139, 125, 107],
            [238, 99, 99],
            [65, 105, 225],
            [238, 238, 0],
            [255, 165, 0],
            [139, 139, 131],
            [99, 109, 185],
            [139, 0, 0],
            [255, 246, 143],
            [255, 0, 0]])

        palette = palette*1.0/255
    elif imageID == 3:
        row = 145
        col = 145
        palette = np.array([
            [140, 67, 46],
            [0, 0, 255],
            [255, 100, 0],
            [0, 255, 123],
            [164, 75, 155],
            [101, 174, 255],
            [118, 254, 172],
            [60, 91, 112],
            [255,255,0],
            [255, 255, 125],
            [255, 0, 255],
            [100, 0, 255],
            [0, 172, 254],
            [0, 255, 0],
            [171, 175, 80],
            [101, 193, 60],
        ])
        palette = palette*1.0/255
    elif imageID == 4:
        row = 610
        col = 340
        palette = np.array([
            [192, 192, 192],
            [0, 255, 0],
            [0, 255, 255],
            [0, 128, 0],
            [255, 0, 255],
            [165, 82, 41],
            [128, 0, 128],
            [255, 0, 0],
            [255,255,0],
        ])
        palette = palette*1.0/255
    
    X_result = np.zeros((labels.shape[0],3))
    for i in range(1,num_class+1):
        X_result[np.where(labels==i),0] = palette[i-1,0]
        X_result[np.where(labels==i),1] = palette[i-1,1]
        X_result[np.where(labels==i),2] = palette[i-1,2]
    
    X_result = np.reshape(X_result,(row,col,3))
    plt.axis ( "off" ) 
    plt.imshow(X_result)    
    return X_result
    
