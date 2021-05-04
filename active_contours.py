#
# VERSION 1.0
#

import numpy as np
import skimage
from skimage.color import rgb2gray
from skimage import data, img_as_float
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)

#FOR debugging
import time


def store_evolution_in(lst):
    #Returns a callback function to store the evolution of the level sets in
    # the given list.

    def _store(x):
        lst.append(np.copy(x))

    return _store

def ac_masks(video,update_it=4,base_it=60,skip=1,lowcut=(1/6),highcut=(35/6),freq_sample=76,memory_it=1,debug=False):

    if debug:
        start = time.time()

    masks=[None]*len(video)

    inici = None
    for i in range(len(video)):
        evolution = []
        if (debug and (i+1)%100==0):
            print(" -> Processing frame ",i+1," of ",len(video))

        if not skip or i%(skip+1)==0:
            if i ==0:
                init_ls = checkerboard_level_set(video[i].shape, 6)
                iters=base_it
            else:
                init_ls = inici
                iters=update_it

            callback = store_evolution_in(evolution)
            masks[i] = morphological_chan_vese(video[i], iters, init_level_set=init_ls, smoothing=3, iter_callback=callback)

            if i==0:
                inici = evolution[base_it-1]
            else:
                inici = evolution[memory_it]

        else:
            masks[i] = masks[i-1]


    if debug:
        print("PROCESS: Elapsed time = ",time.time()-start)

    #FREE MEMORY#
    del evolution
    #############

    return masks
