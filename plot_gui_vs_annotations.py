# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 12:18:11 2017

@author: Fruit Flies
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 23:25:11 2017

@author: Fruit Flies
"""
import sys

from os import listdir
from os.path import isfile, join,  splitext

import numpy as np
from scipy import signal, interpolate
from matplotlib import pyplot as plt
from matplotlib import colors 

import csv

from load_hdf5_data import load_hdf5
from bout_analysis_func import bout_analysis
from expresso_gui_params import analysisParams
#------------------------------------------------------------------------------
#def plot_comp(ind, data_dir, annotations_dir):

ind = 20 #ind + 1 is the name of the data file

data_dir = 'F:\\Dropbox\\Sam\\eBrewerQ\\annotationsdatafiles_hdf5\\'
annotations_dir = 'F:\\Dropbox\\Sam\\eBrewerQ\\observer_corrected\\'

if sys.version_info[0] < 3:
    filekeyname = unicode('XP02') 
    groupkeyname = unicode('channel_2') 
else:
    filekeyname = 'XP02' 
    groupkeyname = 'channel_2' 

min_bout_duration = analysisParams['min_bout_duration']
#min_bout_volume = analysisParams['min_bout_volume']

data_filenames = [f for f in listdir(data_dir) if 
                    isfile(join(data_dir, f)) and f.endswith('.hdf5')]
annotation_filenames = [f for f in listdir(annotations_dir) if 
                    isfile(join(annotations_dir, f)) and f.endswith('.csv')]
                    
data_filenames_int = np.empty(shape=(len(data_filenames),), dtype=int)
for kth in np.arange(len(data_filenames)):
    fname = data_filenames[kth]    
    fname_split = splitext(fname)
    fname_intstr = fname_split[0]
    data_filenames_int[kth] = int(fname_intstr)

filename_sort_ind = np.argsort(data_filenames_int)
data_filenames_sorted = [data_filenames[sort_ind] for sort_ind in filename_sort_ind]
annotation_filenames_sorted = [annotation_filenames[sort_ind] \
                                for sort_ind in filename_sort_ind]

#------------------------------------------------------------------------------

#load and analyze data 
data_file = join(data_dir,data_filenames_sorted[ind])     
dset, t = load_hdf5(data_file,filekeyname,groupkeyname)
    
dset_check = (dset != -1)
if (np.sum(dset_check) == 0):
    messagestr = "Bad dataset: " + data_file
    print(messagestr)

dset_size = dset.size     
frames = np.arange(0,dset_size)

dset = dset[dset_check]
frames = frames[np.squeeze(dset_check)]
t = t[dset_check]

new_frames = np.arange(0,np.max(frames)+1)
sp_raw = interpolate.InterpolatedUnivariateSpline(frames, dset)
sp_t = interpolate.InterpolatedUnivariateSpline(frames, t)
dset = sp_raw(new_frames)
t = sp_t(new_frames)
frames = new_frames
    
dset_smooth, bouts_data, _ = bout_analysis(dset,frames)

data_binary_array = np.zeros(dset_size)
for ith in np.arange(0,bouts_data.shape[1]):
    data_binary_array[bouts_data[0,ith]:bouts_data[1,ith]] = 1 
    
#get annotations
annotations_file = join(annotations_dir,annotation_filenames_sorted[ind])
csv_rows = [] 
with open(annotations_file, 'rb') as csvfile:
    annotations_reader = csv.reader(csvfile)
    for row in annotations_reader:
        csv_rows.append(row)

#clugy, need to fix
if csv_rows[1][3] == ' ' :
    bouts_annotation = np.empty(shape=(2L, 0L), dtype=int)
    bouts_oldcode = np.empty(shape=(2L, 0L), dtype=int)

    annotation_binary_array = np.zeros(dset_size)
    oldcode_binary_array = np.zeros(dset_size)   
else:    
    bouts_annotation = np.ndarray(shape=(2,len(csv_rows)-1), dtype=int)
    bouts_oldcode = np.ndarray(shape=(2,len(csv_rows)-1), dtype=int)
    
    annotation_binary_array = np.zeros(dset_size)
    oldcode_binary_array = np.zeros(dset_size)       
    for row_ind in np.arange(1,len(csv_rows)):
        row_curr = csv_rows[row_ind]
        
        #user annotated bout timing        
        bout_start_ann_t = int(row_curr[3])
        bout_end_ann_t = int(row_curr[4])
        bout_ann_duration = int(row_curr[5])
        
        bout_start_ann = np.searchsorted(t,bout_start_ann_t,side='right')
        bout_end_ann = np.searchsorted(t,bout_end_ann_t,side='right')
        #old code annotated bout timing
        bout_start_oc_t = int(float(row_curr[6]))
        bout_end_oc_t = int(float(row_curr[7]))
        bout_oc_duration = int(float(row_curr[8]))
        
        bout_start_oc = np.searchsorted(t,bout_start_oc_t,side='right')
        bout_end_oc = np.searchsorted(t,bout_end_oc_t,side='right')
        #remove bouts with too short duration and concatenate
        if bout_ann_duration < min_bout_duration:
            bouts_annotation[:,row_ind-1] = np.full((2,),np.nan)
        else:    
            bout_ann = np.vstack((bout_start_ann, bout_end_ann))
            bouts_annotation[:,row_ind-1] = np.squeeze(bout_ann)
            annotation_binary_array[bout_start_ann:bout_end_ann] = 1
            
        if bout_oc_duration < min_bout_duration:
            bouts_oldcode[:,row_ind-1] = np.full((2,),np.nan)
        else:    
            bout_oc = np.vstack((bout_start_oc, bout_end_oc))
            bouts_oldcode[:,row_ind-1] = np.squeeze(bout_oc)
            oldcode_binary_array[bout_start_oc:bout_end_oc] = 1    
            
    bouts_annotation = bouts_annotation[:,~np.isnan(bouts_annotation[1,:])]
    bouts_oldcode = bouts_oldcode[:,~np.isnan(bouts_oldcode[1,:])]

#analyze differences. Want to look at:
#   -normalized hamming distance
#   -bout edges
#   -number of bouts detected
bout_num_diff = bouts_data.shape[1] - bouts_oldcode.shape[1] 
num_bouts_comp = np.abs(bout_num_diff)

hamming_dist = np.sum(np.abs(data_binary_array - oldcode_binary_array))
norm_hamm_dist = hamming_dist/dset_size

#bout_edges_comp = hamming_dist

#------------------------------------------------------------------------------      

#------------------------------------------------------------------------------  
# print summary
#------------------------------------------------------------------------------  
print('Difference in number of bouts detected:')
print(num_bouts_comp)
print('\n')
print('Normalized Hamming distance:')
print(norm_hamm_dist)

#------------------------------------------------------------------------------          
# make plots to display results 
#------------------------------------------------------------------------------  
fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(17, 7))
        
ax1.set_ylabel('Liquid [nL]')
ax2.set_ylabel('Liquid [nL]')
#ax2.set_xlabel('Time [s]')
ax1.set_title(data_filenames_sorted[ind],fontsize=20)
ax2.set_title('Smoothed Data')

ax1.plot(frames,dset,'k-')
ax2.plot(frames, dset_smooth,'k-')
for i in np.arange(bouts_data.shape[1]):
    ax1.plot(frames[bouts_data[0,i]:bouts_data[1,i]], 
             dset[bouts_data[0,i]:bouts_data[1,i]],'r-')
    ax2.plot(frames[bouts_data[0,i]:bouts_data[1,i]], 
             dset_smooth[bouts_data[0,i]:bouts_data[1,i]],'r-')
for j in np.arange(bouts_oldcode.shape[1]):             
    ax2.axvspan(frames[bouts_oldcode[0,j]],frames[bouts_oldcode[1,j]], 
                     facecolor='green', edgecolor='blue', alpha=0.3)
    ax1.axvspan(frames[bouts_oldcode[0,j]],frames[bouts_oldcode[1,j]], 
                     facecolor='green', edgecolor='blue', alpha=0.3)
    
ax1.set_xlim([frames[0],frames[-1]])
ax1.set_ylim([np.amin(dset),np.amax(dset)])

#fig_raster, ax_raster = plt.subplots()
cmap = colors.ListedColormap(['white','red','blue','grey'])
bounds = [0, 1, 2, 3, 4]
norm = colors.BoundaryNorm(bounds, cmap.N)

raster_im = 2*oldcode_binary_array + data_binary_array
raster_im = np.expand_dims(raster_im,axis=0)
ax3.imshow(raster_im, aspect='auto',cmap=cmap,norm=norm,interpolation='none')

ax3.set_xlabel("Time [Idx]")
ax3.set_yticklabels([])
ax3.set_title("Feeding Bout Comparison")    
