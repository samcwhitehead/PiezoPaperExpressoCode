# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 17:55:40 2017

@author: Fruit Flies
"""
import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')

import os 
import sys 

import numpy as np
from scipy import interpolate

from load_hdf5_data import load_hdf5
from bout_analysis_func import bout_analysis

from openpyxl import Workbook
#------------------------------------------------------------------------------
tmin = 0
tmax = 500 
tbin_size = 5 

saveFlag = True
save_name = 'test.xlsx'

file_list = ["G:\\Expresso GUI\\for_testing\\fewbigmealsevents\\1.hdf5",
             "G:\\Expresso GUI\\for_testing\\fewbigmealsevents\\4.hdf5",
             "G:\\Expresso GUI\\for_testing\\fewbigmealsevents\\6.hdf5",
             "G:\\Expresso GUI\\for_testing\\fewbigmealsevents\\7.hdf5",
             "G:\\Expresso GUI\\for_testing\\fewbigmealsevents\\11.hdf5",
             "G:\\Expresso GUI\\for_testing\\fewbigmealsevents\\12.hdf5",
             "G:\\Expresso GUI\\for_testing\\fewbigmealsevents\\13.hdf5",
             "G:\\Expresso GUI\\for_testing\\manysmallmeals\\2.hdf5",
             "G:\\Expresso GUI\\for_testing\\manysmallmeals\\20.hdf5",
             "G:\\Expresso GUI\\for_testing\\manysmallmeals\\21.hdf5",
             "G:\\Expresso GUI\\for_testing\\manysmallmeals\\23.hdf5",
             "G:\\Expresso GUI\\for_testing\\manysmallmeals\\36.hdf5",
             "G:\\Expresso GUI\\for_testing\\manysmallmeals\\38.hdf5"]
grpnum = 0 
dsetnum = 1

#------------------------------------------------------------------------------
# loop through files and detect bouts. store data in lists             
bouts_list = []
dset_smooth_list = [] 
volumes_list = []
name_list = []

for filename in file_list:

    dset = load_hdf5(filename,grpnum,dsetnum)        
    
    dset_check = (dset != -1)
    if (np.sum(dset_check) == 0):
        dset = np.array([])
        frames = np.array([])
        sys.exit("Bad dataset; try different group or dataset number")        
        
    frames = np.arange(0,dset.size)
    
    dset = dset[dset_check]
    frames = frames[np.squeeze(dset_check)]
    
    new_frames = np.arange(0,np.max(frames)+1)
    sp_raw = interpolate.InterpolatedUnivariateSpline(frames, dset)
    dset = sp_raw(new_frames)
    frames = new_frames
    
    dset_smooth, bouts, volumes = bout_analysis(dset,frames)
    
    bouts_list.append(bouts)
    dset_smooth_list.append(dset_smooth)
    volumes_list.append(volumes)
    
    _, name = os.path.split(filename)
    name_list.append(name[0:-5])
    
# calculate raster array 
raster_im = np.zeros([len(bouts_list),tmax-tmin])
for ith, bouts_curr in enumerate(bouts_list):
    bouts_curr = bouts_curr[:,bouts_curr[0,:]<tmax]
    for jth in np.arange(0,bouts_curr.shape[1]):
        raster_im[ith,bouts_curr[0,jth]:bouts_curr[1,jth]] = 1 

# find consumption per unit time
diff_list = []
for dset in dset_smooth_list:
    dset_diff = np.diff(dset)
    dset_diff_length = dset_diff.size
    max_ind_curr = np.amin(np.array([dset_diff_length, tmax-1]))
    dset_diff = dset_diff[0:max_ind_curr]
    dset_diff = np.insert(dset_diff,0,0)
    diff_list.append(dset_diff)

diff_mat = np.vstack(diff_list)
diff_mat = -1.0*diff_mat     


# get summaries for time bin and fly 
bins = np.arange(tmin, tmax, tbin_size)
total_consumption = np.sum(raster_im*diff_mat,axis=0)

consumption_hist = [np.sum(total_consumption[t:t+tbin_size]) for t in bins]
consumption_per_fly = np.sum(raster_im*diff_mat,axis=1)
duration_per_fly = np.sum(raster_im,axis=1)
latency_per_fly = [bouts_temp[0,0] for bouts_temp in bouts_list]

latency_sort_ind = np.argsort(latency_per_fly)
name_list_sorted = [name_list[sort_ind] for sort_ind in latency_sort_ind]        
#------------------------------------------------------------------------------
# make raster plot of bouts             

fig_raster, ax_raster = plt.subplots()
ax_raster.imshow(raster_im[latency_sort_ind,:], aspect='auto',cmap='Greys',interpolation='none')

ax_raster.set_yticks(np.arange(0,len(bouts_list)))
ax_raster.set_yticklabels(name_list_sorted)
ax_raster.set_xlabel("Time [units?]")
ax_raster.set_ylabel("Data File")
ax_raster.set_title("Feeding Bouts")    
    
#------------------------------------------------------------------------------
# make histogram of food consumption

fig_hist, ax_hist = plt.subplots()
ax_hist.fill_between(bins, 0, consumption_hist, facecolor='green', alpha=0.5)

ax_hist.set_xlabel("Time [units?]")
ax_hist.set_ylabel("Food consumed [nL]")
ax_hist.set_title("Total Consumption Histogram")
#------------------------------------------------------------------------------    
# make xlsx file for batch 

wb = Workbook()    

ws_summary = wb.active
ws_summary.title = "Summary"
ws_events = wb.create_sheet("Events")

summary_heading = ["Filename", "XP", "Channel", "Number of Events", "Total Volume [nL]", "Total Duration", "Latency" ]
ws_summary.append(summary_heading)

for ith, bouts_curr in enumerate(bouts_list):
    name_curr = name_list[ith]
    xp_curr = 'XP02'
    channel_curr = 'channel_1'
    event_num_curr = bouts_curr.shape[1]
    total_vol_curr = consumption_per_fly[ith]
    duration_curr = duration_per_fly[ith]
    latency_curr = latency_per_fly[ith]
    
    row_curr = [name_curr, xp_curr, channel_curr, event_num_curr, total_vol_curr, duration_curr, latency_curr]
    ws_summary.append(row_curr)

events_heading = ["Filename", "XP", "Channel", "StartIdx", "EndIdx", "DurationIdx", "Volume [nL]"]
ws_events.append(events_heading)

for ith, bouts_curr in enumerate(bouts_list):
    name_curr = name_list[ith]
    xp_curr = 'XP02'
    channel_curr = 'channel_1'
    ws_events.append([name_curr, xp_curr, channel_curr])
    
    volumes_curr = volumes_list[ith]
    for jth in np.arange(0,bouts_curr.shape[1]):
        bout_start_curr = bouts_curr[0,jth] 
        bout_end_curr = bouts_curr[1,jth]
        volume_curr = volumes_curr[jth]
        duration_curr = bout_end_curr - bout_start_curr 
        row_curr = [" "," "," ",bout_start_curr,bout_end_curr,duration_curr,volume_curr]
        ws_events.append(row_curr) 
        
if saveFlag:
    wb.save(save_name)
  