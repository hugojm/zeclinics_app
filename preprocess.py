#
# VERSION 1.4
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

from numpy import save
from readlif.reader import LifFile
from os import listdir
from os.path import isfile, join
from PIL import Image

def find_top_corner(in_shape,center_idx,out_shape,debug=False):
# out shape = 1 number, shape of dimensions
# in shape = array of input dimensions
    top_left = [center_idx[0] - (out_shape/2),center_idx[1]-(out_shape/2)]
    if top_left[0] < 0: # esquerra
        top_left[0] = 0
    if top_left[1] < 0: # top
        top_left[1] = 0
    if top_left[0]+ out_shape >= in_shape[0]:
        top_left[0]-=(top_left[0]+out_shape-in_shape[0])+1
    if top_left[1]+ out_shape >= in_shape[1]:
        top_left[1]-=(top_left[1]+out_shape-in_shape[1])+1

    if debug:
        print("     --> Centered image top corners: topleft=",top_left,"| topright=",[top_left[0]+out_shape,top_left[1]])
        print("     --> Centered image bot corners: botleft=",[top_left[0],top_left[1]+out_shape],"| botright=",[top_left[0]+out_shape,top_left[1]+out_shape])

    return top_left

def center_img(img,out_shape,top_left):
    img = img[int(top_left[0]):int(top_left[0]+out_shape),int(top_left[1]):int(top_left[1]+out_shape)]
    return img

def lifpreprocess(path,out_dir='output',index_of_interest=2,out_shape=256,store=False,debug=False):
    start=time.time()

    #Read LIF files in the 'path' directory/file
    directory = os.path.isdir(path)
    if(directory):
        path_files = [f for f in listdir(path) if isfile(join(path, f))]
        lif_imgs = [LifFile(path+'/'+subpath).get_image(index_of_interest) for subpath in path_files]
    else:
        #We will not create a sub-directory output
        out_dir = ''
        path_files = [path.split('/')[-1]]
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
    out_dir+='/'
    if(directory and not os.path.exists(out_dir)):
        os.mkdir(out_dir)
    if(not directory):
        out_dir=''
    for i in range(num_imgs):
        current_img=lif_imgs_frames[i]
        os.mkdir(out_dir+path_files[i].split(".")[0])
        for j in range(len(current_img)):
            save(out_dir+path_files[i].split(".")[0]+"/"+path_files[i].split(".")[0]+"_"+str(j),current_img[j])


######
#EXECUTION:
# import preprocess
# lifpreprocess('path' --> path containging LIF files OR LIF file name (if only one), out_dir --> directory to store the output (only if path is a dir), index_of_interest --> index where the video is located in LIF, out_shape --> shape of output arrays (by default 256x256), store --> if we want to store the result in disk (default=False))
######
#EXAMPLE:
# FOR FILE --> result = lifpreprocess('raw_data/20170102_SME_085.lif')
# FOR DIR --> result = lifpreprocess('raw_data','output_batch_1')
# WITH: Store = TRUE --> There's no return
######
