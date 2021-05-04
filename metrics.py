#
# Version 1.2.4.7.2J
#
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

## Cardiac Arrest # v_signal = ventricle[3] = c[3]
def cardiac_arrest(v_signal):
    return  np.std(v_signal)<0.000376003903472389


# Longest time without a beat
def longest_time_wo_beat(peaks,fps):
    max_dist = 0
    x1 = 0
    x2 = peaks[0]
    for i in range(1,len(peaks)):
        x1=x2
        x2=peaks[i]
        if x2-x1>max_dist:
            max_dist = x2-x1

    return max_dist/fps

# Shortest time without a beat
def shortest_time_wo_beat(peaks,fs):
    min_dist = 3000
    x1 = 0
    x2 = peaks[0]
    for i in range(1,len(peaks)):
        x1=x2
        x2=peaks[i]
        if x2-x1<min_dist:
            min_dist = x2-x1

    return min_dist/fs


# per ejection fraction
def min_between_beats(a,b,area):
    target = np.min(area[a:b])
    for i in range(a,b+1):
        if area[i] == target: return i
        else: i = i+1


# ejection fraction
#INPUT: llista de pics, array area/frame b[4] o c[4]
def ejection_fraction(peaks, area_per_frame):
    ej_frac = 0

    for i in range(1,len(peaks)):
        a = peaks[i-1] #left max peak
        b = peaks[i] #right max peak
        c = min_between_beats(a, b, area_per_frame)

        percent_ej_frac = area_per_frame[c]*100/area_per_frame[a]

        ej_frac = ej_frac + percent_ej_frac

    return ej_frac/len(peaks)

# qt mean
def qt (atrpeaks, venpeaks):
    diff = np.array(venpeaks) - np.array(atrpeaks)
    return np.mean(diff)

# Arithmia estimation by longest/shortest
def arithmia_by_longest_shortest(longest,shortest):
    arithmia_coefficient = longest/shortest
    arithmia = False
    treshold = 3 # cal preguntar a la sylvia
    if arithmia_coefficient > treshold:
        arithmia = True
    return arithmia

# Arithmia estimation by distance distribution
def arithmia_by_distance_distribution(peaks,fps):
    x1 = 0
    x2 = peaks[0]
    dist = [None]*(len(peaks)-1)
    for i in range(1,len(peaks)):
        x1=x2
        x2=peaks[i]
        dist[i-1] = x2-x1

    arithmia = False
    mean = np.mean(dist)
    percentage = 0.2 # PARAMETER
    for d in dist:
        if np.abs(d-mean)>percentage*mean:
            arithmia = True
    return arithmia

################## Function calls ##################

def metrics(atrium,ventricle,fps):
    peaks_a = atrium[1]['peaklist']
    peaks_v = ventricle[1]['peaklist']

    metrics_dict = {}

    metrics_dict['cardiac_arrest'] = bool(cardiac_arrest(ventricle[0])) # ventricle[3]

    metrics_dict['longest_v'] = longest_time_wo_beat(peaks_v,fps)
    metrics_dict['longest_a'] = longest_time_wo_beat(peaks_a,fps)
    # print("Longest time without a beat:",longest,"seconds")

    metrics_dict['shortest_v'] = shortest_time_wo_beat(peaks_v,fps)
    metrics_dict['shortest_a'] = shortest_time_wo_beat(peaks_a,fps)
    #print("Shortest time without a beat:",shortest,"seconds")

    metrics_dict['ef_v'] = ejection_fraction(peaks_v,ventricle[0])
    metrics_dict['ef_a'] = ejection_fraction(peaks_a,atrium[0])

    metrics_dict['qt_mean'] = qt(peaks_a, peaks_v)

    metrics_dict['arithmia_1']  = bool(arithmia_by_longest_shortest(metrics_dict['longest_v'],metrics_dict['longest_a']))
    metrics_dict['arithmia_2']  = bool(arithmia_by_distance_distribution(peaks_v,fps))

    with open('static/dict/metrics.pckl', 'wb') as handle:
         pickle.dump(metrics_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


###################################################
