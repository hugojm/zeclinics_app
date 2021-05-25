
#
# VERSION 1.0
#

import numpy as np
import skimage
import cv2
from skimage.color import rgb2gray
from skimage import data, img_as_float
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)
import nnet_cardio as nn
#FOR debugging
import time
import matplotlib.pyplot as plt

def area(mask):
    _,counts = np.unique(mask, return_counts=True)
    return min(counts)

def max_heart_mask(video,debug=False):
    masks=nn.nnet_masks(video,batch_size=5,debug=debug)
    max_mask = 0
    max_val = 0
    for i in range(len(masks)):
        aux = area(masks[i])
        if aux>max_val:
            max_mask = i
            max_val = aux

    return masks[max_mask]

def store_evolution_in(lst):
    #Returns a callback function to store the evolution of the level sets in
    # the given list.

    def _store(x):
        lst.append(np.copy(x))

    return _store

def ac_masks(video,base_it=120,update_it=4,skip=1,memory_it=1,debug=False):

    if debug:
        print("AC Starting...")
        start = time.time()

    masks_v=[None]*len(video)
    masks_a=[None]*len(video)

    heart_mask = max_heart_mask(video,debug=debug)

    inici = None
    for i in range(len(video)):
        evolution_v = []
        evolution_a = []
        if (debug and (i+1)%100==0):
            print(" -> Processing frame ",i+1," of ",len(video))

        if not skip or i%(skip+1)==0:
            if i ==0:
                init_ls_v = heart_mask
                iters=base_it
            else:
                init_ls_v = inici_v
                init_ls_a = inici_a
                iters=update_it

            callback_v = store_evolution_in(evolution_v)
            callback_a = store_evolution_in(evolution_a)
            masks_v[i] = morphological_chan_vese(video[i], iters, init_level_set=init_ls_v, smoothing=3, iter_callback=callback_v)

            kernel_ventricle= np.ones((11,11), np.uint8)
            v_dil = cv2.dilate(np.uint8(masks_v[i]),kernel_ventricle)

            img_sense_v = video[i]*(1-v_dil)

            if i ==0:
                init_ls_a = heart_mask*(1-v_dil)

            masks_a[i]= morphological_chan_vese(img_sense_v, iters, init_level_set=init_ls_a, smoothing=3, iter_callback=callback_a)

            if i==0:
                inici_v = evolution_v[base_it-1]
                inici_a = evolution_a[base_it-1]
            else:
                inici_v = evolution_v[memory_it]
                inici_a = evolution_a[memory_it]

        else:
            masks_v[i] = masks_v[i-1]
            masks_a[i] = masks_a[i-1]


    if debug:
        print("PROCESS: Elapsed time = ",time.time()-start)

    #FREE MEMORY#
    del evolution_v
    del evolution_a
    #############

    return masks_a, masks_v
