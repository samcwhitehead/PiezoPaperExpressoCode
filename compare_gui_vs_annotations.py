# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 23:25:11 2017

@author: Fruit Flies
"""
import sys

from os import listdir
from os.path import isfile, join, splitext

import numpy as np
from scipy import signal, interpolate
from matplotlib import pyplot as plt

import csv

from load_hdf5_data import load_hdf5
from bout_analysis_func import bout_analysis
from expresso_gui_params import analysisParams
#------------------------------------------------------------------------------
data_dir = 'F:\\Dropbox\\Sam\\eBrewerQ\\annotationsdatafiles_hdf5\\'
annotations_dir = 'F:\\Dropbox\\Sam\\eBrewerQ\\observer_corrected\\'

if sys.version_info[0] < 3:
    filekeyname = unicode('XP02') 
    groupkeyname = unicode('channel_2') 
else:
    filekeyname = 'XP02' 
    groupkeyname = 'channel_2' 

min_bout_duration = analysisParams['min_bout_duration']
min_bout_volume = analysisParams['min_bout_volume']

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
# initialize list for data storage
bouts_list_data = [] 
bouts_list_annotations = []
bouts_list_oldcode = [] 
dset_size_list = []

# initialize arrays for comparison metrics. First column will compare my code
#   and the user annotations. Second will compare my code and old code
norm_hamm_dist = np.ndarray(shape=(len(data_filenames),2))
num_bouts_comp = np.ndarray(shape=(len(data_filenames),2))
bout_edges_comp = np.ndarray(shape=(len(data_filenames),2))
                    
for ind in np.arange(0,len(data_filenames_sorted)):
    
    #load and analyze data 
    data_file = join(data_dir,data_filenames_sorted[ind])     
    dset, t = load_hdf5(data_file,filekeyname,groupkeyname)
        
    dset_check = (dset != -1)
    if (np.sum(dset_check) == 0):
        messagestr = "Bad dataset: " + data_file
        print(messagestr)
        continue 
    
    dset_size = dset.size     
    frames = np.arange(0,dset_size)
    dset_size_list.append(dset_size) 
    
    dset = dset[dset_check]
    frames = frames[np.squeeze(dset_check)]
    t = t[dset_check]
    
    new_frames = np.arange(0,np.max(frames)+1)
    sp_raw = interpolate.InterpolatedUnivariateSpline(frames, dset)
    sp_t = interpolate.InterpolatedUnivariateSpline(frames, t)
    dset = sp_raw(new_frames)
    t = sp_t(new_frames)
    frames = new_frames
        
    _, bouts_data, _ = bout_analysis(dset,frames)
    bouts_list_data.append(bouts_data)
    
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
            bout_start_ann_t = int(row_curr[3]) - 1
            bout_end_ann_t = int(row_curr[4]) - 1
            bout_ann_duration = int(row_curr[5])
            
            bout_start_ann = np.searchsorted(t,bout_start_ann_t,side='right')
            bout_end_ann = np.searchsorted(t,bout_end_ann_t,side='right')
            
            #old code annotated bout timing
            bout_start_oc = int(float(row_curr[6])) - 1
            bout_end_oc = int(float(row_curr[7])) - 1 
            bout_oc_duration = int(float(row_curr[8]))
            
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
    
    #add to lists    
    bouts_list_annotations.append(bouts_annotation)
    bouts_list_oldcode.append(bouts_oldcode)

    #analyze differences. Want to look at:
    #   -normalized hamming distance
    #   -bout edges
    #   -number of bouts detected
    bout_num_diff_1 = bouts_data.shape[1] - bouts_annotation.shape[1] 
    bout_num_diff_2 = bouts_data.shape[1] - bouts_oldcode.shape[1]         
    num_bouts_comp[ind,0] = bout_num_diff_1
    num_bouts_comp[ind,1] = bout_num_diff_2
    
    hamming_dist_1 = np.sum(np.abs(data_binary_array - annotation_binary_array))
    hamming_dist_2 = np.sum(np.abs(data_binary_array - oldcode_binary_array))    
    norm_hamm_dist[ind,0] = hamming_dist_1/dset_size
    norm_hamm_dist[ind,1] = hamming_dist_2/dset_size
    
    bout_edges_comp[ind,0] = hamming_dist_1 
    bout_edges_comp[ind,1] = hamming_dist_2
    
#------------------------------------------------------------------------------
#print summary
    
avg_num_bout_diff_1 = np.abs(np.mean(num_bouts_comp[:,0]))
avg_num_bout_diff_2 = np.abs(np.mean(num_bouts_comp[:,1]))

print('Average difference in number of bouts detected:')
print('User vs new code:')
print(avg_num_bout_diff_1)

print('Old code vs new code:')
print(avg_num_bout_diff_2)

avg_norm_hamm_dist_1 = np.mean(norm_hamm_dist[:,0])
avg_norm_hamm_dist_2 = np.mean(norm_hamm_dist[:,1])

print('\n')
print('Average difference in normalized hamming distance:')
print('User vs new code:')
print(avg_norm_hamm_dist_1)

print('Old code vs new code:')
print(avg_norm_hamm_dist_2)

#------------------------------------------------------------------------------    
    
            
# make plots to display results 
#------------------------------------------------------------------------------  
bin_centers = np.arange(0,len(data_filenames_sorted)) + 1
width = 0.35 

#------------------------------------------------------------------------------    
fig_numbouts, ax_numbouts  = plt.subplots(figsize=(17,4.5)) 

rects_numbouts_1 = ax_numbouts.bar(bin_centers-width, num_bouts_comp[:,0], 
                                   width, color='r')
rects_numbouts_2 = ax_numbouts.bar(bin_centers, num_bouts_comp[:,1], 
                                   width, color='b')

ax_numbouts.set_ylabel('Diff. in # of Bouts')
ax_numbouts.set_xlabel('Data File')    
ax_numbouts.set_title('Difference in Number of Bouts')
plt.legend(('new code - user annotations', 'new code - old code'))

ax_numbouts.set_xlim([0,len(data_filenames_sorted)+1]) 
ax_numbouts.set_xticks(np.arange(len(data_filenames_sorted)+1))
ax_numbouts.set_xticklabels(np.arange(len(data_filenames_sorted)+1))
fig_numbouts.set_tight_layout(True)    

#------------------------------------------------------------------------------    
fig_normhamm, ax_normhamm  = plt.subplots(figsize=(17,4.5)) 

rects_normhamm_1 = ax_normhamm.bar(bin_centers-width, norm_hamm_dist[:,0], 
                                   width, color='r')
rects_normhamm_2 = ax_normhamm.bar(bin_centers, norm_hamm_dist[:,1], 
                                   width, color='b')

ax_normhamm.set_ylabel('Norm. Hamming Dist. [Idx]')
ax_normhamm.set_xlabel('Data File')    
ax_normhamm.set_title('Normalized Hamming Distance')
plt.legend(('user annotations v. new code', 'old code v. new code'))   

ax_normhamm.set_xlim([0,len(data_filenames_sorted)+1]) 
ax_normhamm.set_xticks(np.arange(len(data_filenames_sorted)+1))
ax_normhamm.set_xticklabels(np.arange(len(data_filenames_sorted)+1))
fig_normhamm.set_tight_layout(True)    
                   
#------------------------------------------------------------------------------   
"""
fig_edgecomp, ax_edgecomp  = plt.subplots() 

rects_edgecomp_1 = ax_edgecomp.bar(bin_centers-width/2, bout_edges_comp[:,0], 
                                   width, color='r')
rects_edgecomp_2 = ax_edgecomp.bar(bin_centers+width/2, bout_edges_comp[:,1], 
                                   width, color='b')

ax_edgecomp.set_ylabel('Bout Edge Comp. [Idx]')
ax_edgecomp.set_xlabel('Data File')    
ax_edgecomp.set_title('Bout Edge Comparison')
plt.legend(('user annotations v. new code', 'old code v. new code'))                               

ax_edgecomp.set_xlim([0,len(data_filenames_sorted)+1]) 
fig_edgecomp.set_tight_layout(True)    
"""
            