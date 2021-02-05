# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 11:28:03 2017

@author: Fruit Flies
"""

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
from scipy import interpolate
from matplotlib import pyplot as plt
from matplotlib.widgets import MultiCursor

import csv

from load_hdf5_data import load_hdf5
from bout_analysis_func import bout_analysis
from expresso_gui_params import analysisParams
#------------------------------------------------------------------------------
#def plot_comp(ind, data_dir, annotations_dir):

DATA_FILE_NUM = 21
ind = DATA_FILE_NUM - 1 #ind + 1 is the name of the data file
#------------------------------------------------------------------------------

data_dir = 'F:\\Dropbox\\Sam\\eBrewerQ\\annotationsdatafiles_hdf5\\'
save_dir = 'F:\\Dropbox\\Sam\\eBrewerQ\\sam_corrected\\'

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
                    
data_filenames_int = np.empty(shape=(len(data_filenames),), dtype=int)
for kth in np.arange(len(data_filenames)):
    fname = data_filenames[kth]    
    fname_split = splitext(fname)
    fname_intstr = fname_split[0]
    data_filenames_int[kth] = int(fname_intstr)

filename_sort_ind = np.argsort(data_filenames_int)
data_filenames_sorted = [data_filenames[sort_ind] for sort_ind in filename_sort_ind]
data_filenames_int = data_filenames_int[filename_sort_ind]
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

#------------------------------------------------------------------------------          
# make plots to display results 
#------------------------------------------------------------------------------  
bout_start_list = [] 
bout_end_list = [] 

bout_start_ind_list = [] 
bout_end_ind_list = [] 

fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True, figsize=(17, 7))
        
ax1.set_ylabel('Liquid [nL]')
ax2.set_ylabel('Liquid [nL]')
ax2.set_xlabel('Time [s]')
ax1.set_title(data_filenames_sorted[ind],fontsize=20)
ax2.set_title('Smoothed Data')

ax1.plot(t,dset,'k-')
ax2.plot(t, dset_smooth,'k-')
    
ax1.set_xlim([t[0],t[-1]])
ax1.set_ylim([np.amin(dset),np.amax(dset)])

ax1.grid(True)
ax2.grid(True)

multi = MultiCursor(fig.canvas, (ax1, ax2), color='grey', lw=.5, horizOn=True, 
                    vertOn=True)
                    
def onclick(event):
    if event.dblclick:
        t_pick = event.xdata 
        t_closest_ind = np.searchsorted(t,t_pick,side='right')
        t_closest = t[t_closest_ind]
        
        # DOUBLE LEFT CLICK TO SELECT BOUT START IND
        if event.button == 1: 
            bout_start_list.append(t_closest)
            bout_start_ind_list.append(t_closest_ind)
            print('Selected bout start:')
            print(t_closest)
            ax1.axvline(x=t_closest,color='g')
            ax2.axvline(x=t_closest,color='g')
            
            fig.canvas.draw()
        # DOUBLE rIGHT CLICK TO SELECT BOUT END IND    
        elif event.button == 3: 
            bout_end_list.append(t_closest)
            bout_end_ind_list.appbend(t_closest_ind)
            print('Selected bout end:')
            print(t_closest)
            ax1.axvline(x=t_closest,color='r')
            ax2.axvline(x=t_closest,color='r')
            fig.canvas.draw()
        
        else:
            print(event.button)    
          
def onpress(event):
    # Z KEY TO DELETE PREVIOUS START IND SELECTION
    if event.key.lower() == 'z' and len(bout_start_list) > 0 :
        #num_bout_starts = len(bout_start_list)
        del(bout_start_list[-1])
        del(bout_start_ind_list[-1])
        if len(ax1.lines) % 2 == 0: 
            ax1.lines[-1].remove()
            ax2.lines[-1].remove()
        else:
            ax1.lines[-2].remove()  
            ax2.lines[-2].remove()
        fig.canvas.draw() 
    # X KEY TO DELETE PREVIOUS END IND SELECTION    
    elif event.key.lower() == 'x' and len(bout_end_list) > 0 :
        del(bout_end_list[-1])
        del(bout_end_ind_list[-1])
        if len(ax1.lines) % 2 == 0: 
            ax1.lines[-2].remove()
            ax2.lines[-2].remove()
        else:
            ax1.lines[-1].remove()  
            ax2.lines[-1].remove()
        fig.canvas.draw()
    # B KEY TO SAVE RESULTS TO FILE    
    elif event.key.lower() == 'b':
        bout_start_array = np.asarray(bout_start_list)
        bout_end_array = np.asarray(bout_end_list)
        
        bout_start_ind_array = np.asarray(bout_start_ind_list)
        bout_end_ind_array = np.asarray(bout_end_ind_list)
        
        bouts_t = np.transpose(np.vstack((bout_start_array,bout_end_array)))
        bouts_ind = np.transpose(np.vstack((bout_start_ind_array,bout_end_ind_array)))
        row_mat = np.hstack((bouts_ind, bouts_t))
        save_filename = save_dir + str(data_filenames_int[ind]) + '.csv'
        if sys.version_info[0] < 3:
            save_file = open(save_filename, 'wb')
        else:
            save_file = open(save_filename, 'w', newline='')
        save_writer = csv.writer(save_file)
            
        save_writer.writerow([data_filenames_sorted[ind]])
        save_writer.writerow(['Bout Number'] + ['Bout Start [idx]'] + \
            ['Bout End [idx]'] + ['Bout Start [s]'] + ['Bout End [s]'])
        cc = 1            
        for row in row_mat:
            new_row = np.insert(row,0,cc)
            save_writer.writerow(new_row)
            cc += 1
        
        
cid = fig.canvas.mpl_connect('button_press_event', onclick)  
pid = fig.canvas.mpl_connect('key_press_event', onpress)                  