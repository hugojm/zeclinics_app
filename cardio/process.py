#
# VERSION 4.1
#

import sys
import os
import numpy as np
from os import listdir
from os.path import isfile, join
from cardio.preprocess import lifpreprocess
from cardio.active_contours import ac_masks
from cardio.nnet_cardio import nnet_masks

import cv2
from cardio.metrics import dict_metrics
import matplotlib.pyplot as plt
import matplotlib

from scipy.fftpack import fft
from scipy.signal import butter, lfilter, filtfilt
from scipy.signal import freqz
from scipy import ndimage
from scipy import signal
import heartpy as hp
import warnings
import mpld3
import csv

warnings.filterwarnings("error")
warnings.simplefilter("ignore", ResourceWarning)

#FOR optimization
import gc

#FOR debugging
import time

matplotlib.use('Agg')

def read(path,debug=False):
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


def extract_metrics(atrium, ventricle, filter_signal, lowcut, highcut, fps):

    if filter_signal:
        y_ventricle = butter_bandpass_filter(ventricle, lowcut, highcut, fps, order=1)
        y_ventricle = y_ventricle[200:]
    else:
        y_ventricle=ventricle

    if filter_signal:
        y_atrium = butter_bandpass_filter(atrium, lowcut, highcut, fps, order=1)
        y_atrium = y_atrium[200:]
    else:
        y_atrium=atrium

    try:
        n, m = hp.process(np.array(y_atrium), sample_rate = fps)
        peaks_a = n['peaklist']
        abpm = m['bpm']
        bad_atrium=False
    except Warning:
        peaks_a = signal.find_peaks_cwt(y_atrium,np.arange(1,15))
        abpm = len(peaks_a)/len(y_atrium) * fps * 60
        bad_atrium=True

    try:
        n, m = hp.process(np.array(y_ventricle), sample_rate = fps)
        peaks_v = n['peaklist']
        vbpm = m['bpm']
        bad_ventricle=False
    except Warning:
        peaks_v = signal.find_peaks_cwt(y_ventricle,np.arange(1,15))
        vbpm = len(peaks_v)/len(y_ventricle) * fps * 60
        bad_ventricle=True

    peaks_a = signal.find_peaks_cwt(y_atrium,np.arange(1,30))

    return y_atrium,peaks_a,abpm,bad_atrium,y_ventricle,peaks_v,vbpm,bad_ventricle


def convert_16_to_8(img):
    imin = img.min()
    imax = img.max()

    a = 255 / (imax - imin)
    b = 255 - a * imax
    new_img = (a * img + b).astype(np.uint8)

    return new_img


def scale_signals(v, a):
    div = max(max(v),max(a))
    return v/(div), a/div #(div-div+1)

def ecg(atrium, ventricle,path, save=False):
    ventricle, atrium = scale_signals(ventricle, atrium)

    # Invert ventricle signal
    inv_atrium = -atrium-max(-atrium)+max(atrium)

    # Plot
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)
    ax.plot(ventricle, 'cornflowerblue', label = 'Ventricle')
    ax.plot(inv_atrium, 'indianred', label = 'Atrium')
    ax.set(ylim=(min(ventricle)-0.03, max(ventricle)+0.02))
    ax.legend(loc='lower right', shadow=True, ncol=2)
    if save:
        mpld3.save_html(fig, path)
        plt.close(fig)

def save_csv(dict,path,debug=False):
    csv_columns = ['video','fps','active_contours','bad_atrium_signal','bad_ventricle_signal','a_beating','v_beating','a_bpm','v_bpm','longest_a','longest_v','shortest_a','shortest_v','ef_a','ef_v','qt_mean','arrhythmia_1','arrhythmia_2']
    name=dict['video']
    csv_file = path + ('output' if name =='unknown' else name)+".csv"
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for key in csv_columns:
                if key!='arrhythmia_2':
                    csvfile.write("%s,"%(dict[key]))
                else:
                    csvfile.write("%s"%(dict[key]))

        return csv_file

    except IOError:
        if debug:
            print("I/O error for the CSV creation")


def generate_video(video_frames, masks_a, masks_v, fps, video_name, debug=False):

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


def process_video(input_video, base_it=120, update_it=4, skip=1, memory_it=1, border_removal=20, filter_signal=True, lowcut=(10), highcut=(350), fps=76, p_index=2, p_out_shape=(482,408), gen_video=False, video_name='output.webm', p_store=False, p_out_dir='output', debug=False):
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
    # --> fps: fps of the raw video
    # --> p_index: index of the video in the LIF file (can be checked via ImageJ)
    # --> p_out_shape: shape (n x n) of the output images, if 'original' mantain shape
    # --> mode: 'nn', 'ac' or 'both'. If nn it uses nn, ac uses Active Contours, both uses nn for atrium and Active Contours for ventricle.
    # --> batch_size: batch size for nn masks input
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
            video = read(input_video,debug)
    else:
        video = input_video

    #METRIC EXTRACTION
    frame_a = [None]*len(video)
    frame_v = [None]*len(video)
    masks_a,masks_v=ac_masks(video,update_it=update_it,base_it=base_it,skip=skip,memory_it=memory_it,debug=debug)

    for i in range(len(video)):
        #CROP THE EDGES OF THE MASK AND IMAGE
        reduced_v = masks_v[i][border_removal:-border_removal, border_removal:-border_removal]
        reduced_a = masks_a[i][border_removal:-border_removal, border_removal:-border_removal]

        img = video[i][border_removal:-border_removal, border_removal:-border_removal]

        _, counts_v = np.unique(reduced_v, return_counts=True)
        _, counts_a = np.unique(reduced_a, return_counts=True)

        frame_v[i] = min(counts_v)
        frame_a[i] = min(counts_a)


    frame_a,peaks_a,abpm,bad_atrium,frame_v,peaks_v,vbpm,bad_ventricle=extract_metrics(frame_a, frame_v, filter_signal, lowcut/60, highcut/60, fps)

    metrics=dict_metrics(frame_a,peaks_a,abpm,frame_v,peaks_v,vbpm,fps=fps,bad_atrium=bad_atrium,bad_ventricle=bad_ventricle,debug=debug)

    video_data = {}
    video_data['video']=((input_video.split('/')[-1]).split('.')[0] if isinstance(input_video,str) else 'unknown')
    video_data['fps']=fps
    video_data['active_contours']=str(base_it)+' '+str(update_it)+' '+str(skip)

    final_metrics = {**metrics, **video_data}

    video_path=None
    if gen_video:
        video_path = generate_video(video,masks_a=masks_a,masks_v=masks_v,video_name=video_name,fps=fps,debug=debug)

    #FREE MEMORY#
    del video
    gc.collect()
    #############

    return masks_a, masks_v, frame_a, frame_v, final_metrics, video_path


def process_multiple(input_multiple, base_it=120, update_it=4, skip=1, memory_it=1, border_removal=20, filter_signal=False, lowcut=10, highcut=350, fps=76, p_index=2, p_out_shape=(482,408), gen_video=False, video_name='.webm', p_store=False, p_out_dir='output', debug=False):

    if isinstance(input_multiple, str):
        path_files = [f.split('.')[0] for f in listdir(input_multiple) if isfile(join(input_multiple, f))]
        video_arrays = lifpreprocess(input_multiple,out_dir=p_out_dir,index_of_interest=p_index,out_shape=p_out_shape,store=p_store,debug=debug)

    else:
        video_arrays = input_multiple
        path_files= ['video_'+str(i) for i in range(len(input_multiple))]

    if(debug):
        print("Number of read videos: "+str(len(video_arrays)))

    masks_a, masks_v = [None] * len(video_arrays), [None] * len(video_arrays)
    frame_a, frame_v = [None] * len(video_arrays), [None] * len(video_arrays)
    metrics = [None] * len(video_arrays)
    video_path = [None] * len(video_arrays)

    for i in range(len(video_arrays)):
        if (debug):
            print("\nProcessing video ",i+1," of ",len(video_arrays))
        masks_a[i], masks_v[i], frame_a[i], frame_v[i], metrics[i], video_path[i]=process_video(input_video=video_arrays[i],base_it=base_it,update_it=update_it,skip=skip,memory_it=memory_it,border_removal=border_removal,filter_signal=filter_signal, lowcut=lowcut, highcut=highcut, fps=fps, p_index=p_index, p_out_shape=p_out_shape,gen_video=gen_video,video_name=path_files[i]+video_name,p_store=p_store,p_out_dir=p_out_dir,debug=debug)

    return masks_a,masks_v,frame_a,frame_v,metrics,video_path


#########
#EXECUTION:
#import process
# output = processdir('path' -> Directory with subfolders for each video, debug = T or F)
#path = '/Users/marcfuon/Desktop/LIFS/20170102_SME_085.lif'
#frames = lifpreprocess(path, out_shape = (482,408), debug=True)
#masks_a,masks_v,a,v,dict,_ =process_video(frames[:500],fps=76,debug=True,filter_signal=False,gen_video=False)
#########
#EXAMPLE:
#masks_a,masks_v,a,v,_=process_dir("../../LIFS",update_it=4,skip=1,debug=True,mode='ac',gen_video=True)
#print(dict['a_bpm'],dict['v_bpm'],dict['ef_a'],dict['ef_v'])
#plt.imshow(masks_v[0], cmap='gray')
#plt.show()
#########
