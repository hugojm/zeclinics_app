#
# VERSION 3.3
#

import sys
import os
import numpy as np
from os import listdir
from os.path import isfile, join
from cardio.preprocess import lifpreprocess
from cardio.active_contours import ac_masks
from cardio.nnet_cardio import nnet_masks
import cardio.metrics as m

import heartpy as hp
import cv2
import matplotlib.pyplot as plt

from scipy.fftpack import fft
from scipy.signal import butter, lfilter, filtfilt
from scipy.signal import freqz
from scipy import ndimage

#FOR optimization
import gc

#FOR debugging
import time




def read(path):
    if debug:
        start=time.time()
        print("Reading files...")

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

    #FREE MEMORY#
    del frames_arrays
    gc.collect()
    #############

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

    vbpm=vm['bpm']

    y_atrium = butter_bandpass_filter(atrium, max((vbpm/3)/60,(1/6)), min((vbpm*3)/60,(35/6)), fs, order=1)
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


def create_video(video_frames, masks_a, masks_v, video_name, fps, debug):

    if debug:
        print("\nStarted Video Generation")

    start = time.time()

    size=video_frames[0].shape

    frames_vid = [None]*len(video_frames)
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    out = cv2.VideoWriter(video_name, fourcc, fps, (size[1],size[0]), False)
    for i in range(0,len(video_frames)):
        grad_v = ndimage.morphological_gradient(masks_v[i], size=(3,3))
        grad_a = ndimage.morphological_gradient(masks_a[i], size=(3,3))
        frames_vid[i] = np.maximum((grad_v+grad_a)*(np.max(video_frames[i])+1),video_frames[i])
        imgu8 = convert_16_to_8(frames_vid[i])
        out.write(imgu8)
    out.release()

    if debug:
        print("VIDEO: Elapsed time ",time.time()-start)

    #FREE MEMORY#
    del frames_vid
    gc.collect()
    #############

    return video_name


def process_video(input_video, base_it=50, update_it=4, skip=1, memory_it=1, border_removal=20, lowcut=(1/6), highcut=(35/6), freq_sample=76, p_index=2, p_out_shape='original', gen_video=False, video_name='output.webm', p_store=False, p_out_dir='output', mode='nn', debug=False):
    # FUNCTION TO PROCESS A LIST OF frames
    # This function applies morphological_chain_vese to each frame,
    #   implemented with memory for efficiency.
    #
    # --> input_video: EITHER A LifFile, DIR OR AN ARRAY
    # --> base_it: active contour iterations for step 0
    # --> update_it: active contour iterations for steps > 0 (lower precision required)
    # --> skip: parameter to add skipping to the algorithm (skip n frames every n+1)
    # --> memory_it: iteration which is passed to the following frame as baseline
    # --> border_removal: size of the border to remove in img and masks
    # --> lowcut & highcut: values for min and max bpm expected
    # --> freq_sample: fps of the raw video
    # --> p_index: index of the video in the LIF file (can be checked via ImageJ)
    # --> p_out_shape: shape (n x n) of the output images, if 'original' mantain shape
    # --> mode: 'nn', 'ac' or 'both'. If nn it uses nn, ac uses Active Contours, both uses nn for atrium and Active Contours for ventricle.
    #
    # # # EXTRA FEATURES --> SPACE CONSUMING # # #
    # --> gen_video: if TRUE, stores a video showing the segmentation
    #   --> video_name: name of the output video (MUST END WITH .webm)
    # --> p_store: boolean, if TRUE the result is stored locally
    #   --> p_out_dir: directory path to store the result
    #
    # # # FOR DEVELOPERS # # #
    # --> debug: To print the execution status
    #

    if isinstance(input_video, str):
        if(".lif" in input_video):
            video = lifpreprocess(input_video,out_dir=p_out_dir,index_of_interest=p_index,out_shape=p_out_shape,store=p_store,debug=debug)
        else:
            video = read(input_video)
    else:
        video = input_video

    #METRIC EXTRACTION
    per_frame_a = [None]*len(video)
    per_frame_v = [None]*len(video)

    if mode != 'ac':
        if mode == 'both':
            masks_a,_=nnet_masks(video,image=True,debug=debug)
            masks_v=ac_masks(video,lowcut=lowcut,highcut=highcut,freq_sample=freq_sample,update_it=update_it,base_it=base_it,skip=skip,memory_it=memory_it,debug=debug)
        else:
            masks_a,masks_v=nnet_masks(video,image=True,debug=debug)

        if debug:
            start = time.time()
            print("\nExtracting metrics...")

        for i in range(len(video)):
            #CROP THE EDGES OF THE MASK AND IMAGE
            reduced_v = masks_v[i][border_removal:-border_removal, border_removal:-border_removal]
            reduced_a = masks_a[i][border_removal:-border_removal, border_removal:-border_removal]

            img = video[i][border_removal:-border_removal, border_removal:-border_removal]

            _, counts_v = np.unique(reduced_v, return_counts=True)
            _, counts_a = np.unique(reduced_a, return_counts=True)

            per_frame_v[i] = min(counts_v)
            per_frame_a[i] = min(counts_a)

    else:
        masks_a=None
        masks_v=ac_masks(video,lowcut=lowcut,highcut=highcut,freq_sample=freq_sample,update_it=update_it,base_it=base_it,skip=skip,memory_it=memory_it,debug=debug)

        if debug:
            start = time.time()
            print("\nExtracting metrics...")

        for i in range(len(video)):
            #CROP THE EDGES OF THE MASK AND IMAGE
            reduced_v = masks_v[i][border_removal:-border_removal, border_removal:-border_removal]
            img = video[i][border_removal:-border_removal, border_removal:-border_removal]

            unique, counts = np.unique(reduced_v, return_counts=True)
            per_frame_v[i] = min(counts)
            per_frame_a[i] = np.sum((1-reduced_v)*img)

        per_frame_a = [x / max(per_frame_a) for x in per_frame_a]


    av,am,vv,vm=extract_metrics(per_frame_a, per_frame_v, lowcut, highcut, freq_sample)

    if debug:
        print("METRIC EXTRACTION: Elapsed time = ",time.time()-start)

    video_path=None
    if gen_video:
        video_path = create_video(video,masks_a=masks_a,masks_v=masks_v,video_name=video_name,fps=freq_sample,debug=debug)

    #FREE MEMORY#
    del video
    gc.collect()
    #############

    m.metrics([per_frame_a,av,am], [per_frame_v,vv,vm], freq_sample)

    return masks_a, masks_v, [per_frame_a,av,am], [per_frame_v,vv,vm], video_path


def process_dir(input_video_arrays, base_it=50, update_it=4, skip=1, memory_it=1, border_removal=20, lowcut=(1/6), highcut=(35/6), freq_sample=76, p_index=2, p_out_shape='original', gen_video=False, video_name='.webm', p_store=False, p_out_dir='output', debug=False):

    if isinstance(input_video_arrays, str):
        path_files = [f.split('.')[0] for f in listdir(input_video_arrays) if isfile(join(input_video_arrays, f))]
        video_arrays = lifpreprocess(input_video_arrays,out_dir=p_out_dir,index_of_interest=p_index,out_shape=p_out_shape,store=p_store,debug=debug)

    else:
        video_arrays = input_video_arrays

    if(debug):
        print("Number of read videos: "+str(len(video_arrays)))

    masks = [None] * len(video_arrays)
    atrium, ventricle = [None] * len(video_arrays), [None] * len(video_arrays)
    video_paths = [None] * len(video_arrays)

    for i in range(len(video_arrays)):
        if (debug):
            print("\nProcessing video ",i+1," of ",len(video_arrays))
        masks[i],atrium[i],ventricle[i],video_paths[i]=process_video(input_video=video_arrays[i],base_it=base_it,update_it=update_it,skip=skip,memory_it=memory_it,border_removal=border_removal,lowcut=lowcut,highcut=highcut,freq_sample=freq_sample,p_index=p_index,p_out_shape=p_out_shape,gen_video=gen_video,video_name=path_files[i]+video_name,p_store=p_store,p_out_dir=p_out_dir,debug=debug)

    return masks,atrium,ventricle,video_paths
