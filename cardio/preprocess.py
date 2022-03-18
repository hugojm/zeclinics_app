#
# VERSION 2.1
#

#
# IF USING dependencies DIRECTORY:
#   sys.path.append(path/to/dependencies/)
#

import sys
import os
import numpy as np

#FOR debugging
import time
import matplotlib.pyplot as plt

#FOR optimization
import gc

from numpy import save
from readlif.reader import LifFile
from os import listdir
from os.path import isfile, join
from PIL import Image
import cv2

def find_top_corner(in_shape,center_idx,out_shape,debug=False):
# out shape = 1 number, shape of dimensions
# in shape = array of input dimensions
    if out_shape == "original":
        return [0,0]

    top_left = [int(center_idx[0] - (out_shape[0]/2)),int(center_idx[1]-(out_shape[1]/2))]
    if top_left[0] < 0: # esquerra
        top_left[0] = 0
    if top_left[1] < 0: # top
        top_left[1] = 0
    if top_left[0]+ out_shape[0] >= in_shape[0]:
        top_left[0]-=(top_left[0]+out_shape[0]-in_shape[0])+1
    if top_left[1]+ out_shape[1] >= in_shape[1]:
        top_left[1]-=(top_left[1]+out_shape[1]-in_shape[1])+1

    if debug:
        print("     --> Centered image top corners: topleft=",top_left,"| topright=",[top_left[0],top_left[1]+out_shape[1]])
        print("     --> Centered image bot corners: botleft=",[top_left[0]+out_shape[0],top_left[1]],"| botright=",[top_left[0]+out_shape[0],top_left[1]+out_shape[1]])

    return top_left

def center_img(img,out_shape,top_left):
    if out_shape=='original':
        return img

    tl1,tl2=top_left[0],top_left[1]
    #ADD ZERO PADDING
    if(tl1<0):
        padding=np.zeros((-tl1,img.shape[1]),dtype='uint16')
        img=cv2.vconcat([padding,img])
        tl1=0


    if(tl2<0):
        padding=np.zeros((img.shape[0],-tl2),dtype='uint16')
        img=cv2.hconcat([padding,img])
        tl2=0

    img = img[int(tl1):int(tl1+out_shape[0]),int(tl2):int(tl2+out_shape[1])]
    return img


def lifpreprocess(path, out_dir='output', index_of_interest=2, out_shape='original', store=False, debug=False):
    start=time.time()

    #Read LIF files in the 'path' directory/file
    directory = os.path.isdir(path)
    if(directory):
        print('directory', directory, 'is a directory')
        path_files = [f for f in listdir(path) if isfile(join(path, f))]
        lif_imgs = [LifFile(path+os.sep+subpath).get_image(index_of_interest) for subpath in path_files]
    else:
        #We will not create a sub-directory output
        out_dir = ''
        path_files = [path.split(os.sep)[-1]]
        lif_imgs = [LifFile(path).get_image(index_of_interest)]
    num_imgs = len(lif_imgs)

    #Process the frames of all the videos
    #   -> Center the image and transform the format
    lif_imgs_frames=[None]*num_imgs
    for i in range(num_imgs):
        if debug:
            print("-> preprocessing video ",i+1,"of",num_imgs)
        #Find the first frame and center all frames wrt frame 0
        img_0 = np.uint16(np.array(lif_imgs[i].get_frame(),dtype="object"))
        center = np.unravel_index(np.argmax(img_0, axis=None), img_0.shape)
        top_left = find_top_corner(img_0.shape,center,out_shape,debug)

        lif_imgs_frames[i] = [center_img(np.uint16(np.array(tv,dtype="object")),out_shape,top_left) for tv in lif_imgs[i].get_iter_t()]

    if(debug):
        print("PREPROCESS: Elapsed time = ", time.time()-start)


    #If we dont want to store
    if(not store):
        if directory:
            return lif_imgs_frames
        else:
            return lif_imgs_frames[0]

    #Save arrays in disk
    out_dir+=os.sep
    if(directory and not os.path.exists(out_dir)):
        os.mkdir(out_dir)
    if(not directory):
        out_dir=''
    for i in range(num_imgs):
        current_img=lif_imgs_frames[i]
        os.mkdir(out_dir+path_files[i].split(".")[0])
        for j in range(len(current_img)):
            save(out_dir+path_files[i].split(".")[0]+os.sep+path_files[i].split(".")[0]+"_"+str(j),current_img[j])

    #FREE MEMORY#
    del lif_imgs_frames
    gc.collect()
    #############


#EXAMPLE:
# FOR FILE --> result =
#lifpreprocess('/Users/marcfuon/Desktop/LIFS/20170102_SME_085.lif',out_shape=(482,408),debug=True)
# FOR DIR --> result = lifpreprocess('raw_data')
# WITH: Store = TRUE --> There's no return
######
