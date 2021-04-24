#
# VERSION 2.1
#

import sys
import os
import numpy as np
from os import listdir
from os.path import isfile, join
from preprocess import lifpreprocess

import skimage
from skimage.color import rgb2gray
from skimage import data, img_as_float
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)
from scipy.fftpack import fft
from scipy.signal import butter, lfilter, filtfilt
from scipy.signal import freqz
from scipy import ndimage
import heartpy as hp
import cv2
import matplotlib.pyplot as plt

#FOR debugging
import time


def read(path):
    start=time.time()

    path_files = [f for f in listdir(path)]

    #Bad execution handling
    if len(path_files)==0:
        print("No files found in the directory")
        return None

    #Check if we have one or many videos to treat
    # if there are subfolders --> many
    # if there are files inside --> one
    directory = os.path.isdir(path+'/'+path_files[0])
    if directory:
        num_files=len(path_files)
        img_arrays= [None] * num_files
        for i in range(num_files):
            files = [f for f in listdir(path+"/"+path_files[i]) if isfile(join(path+"/"+path_files[i], f))]
            num_frames=len(files)
            frames_arrays= [None] * num_frames
            for j in range(num_frames):
                frames_arrays[j]=np.load(path+"/"+path_files[i]+"/"+files[j])
            img_arrays[i]=frames_arrays
    else:
        num_files=len(path_files)
        img_arrays= [None] * num_files
        for i in range(num_files):
            img_arrays[i]=np.load(path+'/'+path_files[i])

    print("READ & LOAD: Elapsed time = ", time.time()-start)
    return img_arrays


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def extract_metrics(atrium, ventricle, lowcut, highcut, fs):

    y_ventricle = butter_bandpass_filter(ventricle, lowcut, highcut, fs, order=1)
    y_ventricle = y_ventricle[200:]
    vv, vm = hp.process(y_ventricle, fs)

    v_bpm=vm['bpm'] #Get the ventricle bpm result to reduce atrium bpm variance

    y_atrium = butter_bandpass_filter(atrium, (v_bpm-5)/60, (v_bpm+5)/60, fs, order=1)
    y_atrium = y_atrium[200:]
    av, am = hp.process(y_atrium, fs)

    return av,am,vv,vm


def convert_16_to_8(img):
    imin = img.min()
    imax = img.max()

    a = 255 / (imax - imin)
    b = 255 - a * imax
    new_img = (a * img + b).astype(np.uint8)
    return new_img


def create_video(video_frames, masks, video_name, size, fps, debug):

    if debug:
        print("Started Video Generation")

    start = time.time()

    frames_vid = [None]*len(video_frames)
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    out = cv2.VideoWriter(video_name, fourcc, fps, (size, size), False)
    for i in range(0,len(video_frames)):
        grad = ndimage.morphological_gradient(masks[i], size=(3,3))
        frames_vid[i] = np.maximum(grad*(np.max(video_frames[i])+1),video_frames[i])
        imgu8 = convert_16_to_8(frames_vid[i])
        out.write(imgu8)
    out.release()

    if debug:
        print("VIDEO: Elapsed time ",time.time()-start)

    return video_name


def store_evolution_in(lst):
    #Returns a callback function to store the evolution of the level sets in
    # the given list.

    def _store(x):
        lst.append(np.copy(x))

    return _store


def process_video(input_video, base_it=50, update_it=4, skip=1, memory_it=1, border_removal=20, lowcut=(1/6), highcut=(35/6), freq_sample=76, p_index=2, p_out_shape=256, gen_video=False, video_name='output.webm', p_store=False, p_out_dir='output', debug=False):
    # FUNCTION TO PROCESS A LIST OF frames
    # This function applies morphological_chan_vese to each frame,
    #   implemented with memory for efficiency.
    #
    # --> input_video: EITHER A LifFile, DIR OR AN ARRAY
    # --> base_it: active contour iterations for step 0
    # --> update_it: active contour iterations for steps > 0 (lower precision required)
    # --> skip: parameter to add skipping to the algorithm (skip n frames every n+1)
    # --> memory_it: iteration which is passed to the following frame as baseline
    # --> border_removal: size of the border to remove in img and masks
    # --> lowcut & highcut: values for min and max bpm expected
    # --> freq_sample: fps of the raw video
    # --> p_index: index of the video in the LIF file (can be checked via ImageJ)
    # --> p_out_shape: shape (n x n) of the output images
    #
    # # # EXTRA FEATURES --> SPACE CONSUMING # # #
    # --> gen_video: if TRUE, stores a video showing the segmentation
    #   --> video_name: name of the output video (MUST END WITH .mp4)
    # --> p_store: boolean, if TRUE the result is stored locally
    #   --> p_out_dir: directory path to store the result
    #
    # # # FOR DEVELOPERS # # #
    # --> debug: To print the execution status
    #

    if isinstance(input_video, str):
        if(".lif" in input_video):
            video = lifpreprocess(input_video,out_dir=p_out_dir,index_of_interest=p_index,out_shape=p_out_shape,store=p_store,debug=debug)
        else:
            video = read(input_video)
    else:
        video = input_video

    masks=[None]*len(video)

    #For metric computation
    area_per_frame_atrium = [None]*len(video)
    area_per_frame_ventricle = [None]*len(video)
    intens_per_frame_atrium = [None]*len(video)
    intens_per_frame_ventricle = [None]*len(video)

    if debug:
        start = time.time()

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

        #CROP THE EDGES OF THE MASK AND IMAGE
        reduced = masks[i][border_removal:p_out_shape-border_removal, border_removal:p_out_shape-border_removal]
        img = video[i][border_removal:p_out_shape-border_removal, border_removal:p_out_shape-border_removal]

        unique, counts = np.unique(reduced, return_counts=True)
        area_per_frame_ventricle[i] = min(counts)
        area_per_frame_atrium[i] = None


        intens_per_frame_atrium[i]=np.sum(reduced*img)
        intens_per_frame_ventricle[i]=np.sum((1-reduced)*img)

    intens_per_frame_atrium = [x / max(intens_per_frame_atrium) for x in intens_per_frame_atrium]
    intens_per_frame_ventricle = [x / max(intens_per_frame_ventricle) for x in intens_per_frame_ventricle]

    atrium=[intens_per_frame_atrium, min(intens_per_frame_atrium), max(intens_per_frame_atrium), None, area_per_frame_atrium] #NONE ES MEAN
    ventricle=[intens_per_frame_ventricle, min(area_per_frame_ventricle), max(area_per_frame_ventricle), np.mean(area_per_frame_ventricle), area_per_frame_ventricle]

    if debug:
        print("PROCESS: Elapsed time = ",time.time()-start)

    av,am,vv,vm = extract_metrics(atrium=intens_per_frame_atrium,ventricle=area_per_frame_ventricle,lowcut=lowcut,highcut=highcut,fs=freq_sample)
    atrium.append(av)
    atrium.append(am)
    ventricle.append(vv)
    ventricle.append(vm)

    if gen_video:
        video_path = create_video(video,masks,video_name=video_name,fps=freq_sample,size=p_out_shape,debug=debug)

    return masks, atrium, ventricle, video_path


def process_dir(input_video_arrays, raw=True, p_out_dir='output', p_index=2, p_out_shape=256, p_store=False, debug=False):
    if isinstance(input_video_arrays, str):
        if raw:
            video_arrays = lifpreprocess(input_video_arrays,out_dir=p_out_dir,index_of_interest=p_index,out_shape=p_out_shape,store=p_store,debug=debug)
        else:
            video_arrays = read(input_video_arrays)
    else:
        video_arrays = input_video_arrays
    if(debug):
        print("Number of read videos: "+str(len(video_arrays)))

    masks = [None] * len(video_arrays)
    atrium, ventricle = [None] * len(video_arrays), [None] * len(video_arrays)

    for i in range(len(video_arrays)):
        if (debug):
            print("Processing video ",i+1," of ",len(video_arrays))
        masks[i],atrium[i],ventricle[i]=process_video(video_arrays[i],debug=debug)
    return masks,atrium,ventricle


#########
#EXECUTION:
# import process
# output = processdir('path' -> Directory with subfolders for each video, debug = T or F)
#########
#EXAMPLE:
#masks,a,v,_=process_video("./20170102_SME_085.lif",debug=True,gen_video=True,update_it=1,skip=40)
#print(len(a),len(v),v[6]['bpm'])
#plt.imshow(p_vid[2270], cmap='gray')
#plt.show()
#########
