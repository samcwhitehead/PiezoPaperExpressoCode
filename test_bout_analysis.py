# -*- coding: utf-8 -*-
"""
Created on Wed Dec 07 10:39:45 2016

@author: Fruit Flies
"""
from __future__ import division
import sys
#sys.path.append('C:\Users\Fruit Flies\Documents\Python Scripts\BayesChangePt')
from os import listdir
from os.path import isfile, join

from load_hdf5_data import load_hdf5
from my_wavelet_denoise import wavelet_denoise
import numpy as np

from scipy import signal, interpolate

from expresso_data_folders import dataFolders
#import bayesian_changepoint_detection.offline_changepoint_detection as offcd
#from functools import partial
import seaborn

from changepy import pelt
from changepy.costs import normal_mean

import matplotlib.pyplot as plt

from expresso_gui_params import analysisParams

#from bayesian_changepoint_detection import offline_changepoint_detectionas offcd
#from functools import partial
#---------------------------------------------------------------------------------------

folder_ind = 6  #0 = few big, 1 = few small, 2 = many small, 3 = no drinking, 4 = annotated files
datapath = dataFolders[folder_ind] 

filenames = [f for f in listdir(datapath) if isfile(join(datapath, f))]

file_ind = 0 
grpnum = 0 
dsetnum = 3
wlevel = analysisParams['wlevel'] 
wtype = analysisParams['wtype']
medfilt_window = analysisParams['medfilt_window']
#var_scale = .5
#var_scale_testdata = 2*(1.3e-4)
mad_thresh = analysisParams['mad_thresh']
var_user = analysisParams['var_user']

#---------------------------------------------------------------------------------------
filename = join(datapath, filenames[file_ind])
print(filename)

dset, t = load_hdf5(filename,grpnum,dsetnum)
dset_check = (dset != -1)
if (np.sum(dset_check) == 0):
    sys.exit("Bad dataset; try different group or dataset number")    
    
frames = np.arange(0,dset.size)

dset = dset[dset_check]
frames = frames[np.squeeze(dset_check)]

new_frames = np.arange(0,np.max(frames)+1)
sp_raw = interpolate.InterpolatedUnivariateSpline(frames, dset)
dset = sp_raw(new_frames)
frames = new_frames

dset_denoised = wavelet_denoise(dset, wtype, wlevel) 

#---------------------------------------------------------------------------------------

#sp_signal = interpolate.UnivariateSpline(frames, np.squeeze(dset_denoised))
#sp_der = sp_signal.derivative(n=1)

#dset_der = sp_der(frames)

dset_denoised_med = signal.medfilt(dset_denoised,medfilt_window)

sp_dset = interpolate.InterpolatedUnivariateSpline(frames, np.squeeze(dset_denoised_med))
sp_der = sp_dset.derivative(n=1)

dset_der = sp_der(frames)
#---------------------------------------------------------------------------------------

#Q, P, Pcp = offcd.offline_changepoint_detection(dset_der, partial(offcd.const_prior, 
#                l=(len(dset_der)+1)), offcd.gaussian_obs_log_likelihood, truncate=-50)


#---------------------------------------------------------------------------------------

dset_var = np.var(dset_der)
#Q, P, Pcp = offcd.offline_changepoint_detection(dset_var, \
#     partial(offcd.const_prior, l=(len(dset_var)+1)), \
#     offcd.gaussian_obs_log_likelihood, truncate=-40)
changepts = pelt(normal_mean(dset_der,var_user),len(dset_der)) #var_scale*dset_var #var_scale_testdata*len(dset_der)
#changepts = pelt(normal_meanvar(dset_der),len(dset_der))
N = len(dset_der) - 1 

if 0 not in changepts:
    changepts.insert(0,0)
#if len(dset_der) not in changepts:
#    changepts.append(len(dset_der))
if N not in changepts:
    changepts.append(N)
        
#print(changepts)

#---------------------------------------------------------------------------------------

piecewise_fits = np.empty(len(changepts)-1)
piecewise_fit_dist = np.empty_like(dset_der)

for i in range(0,len(changepts)-1):
    ipt1 = changepts[i]
    ipt2 = changepts[i+1] + 1
    fit_temp = np.median(dset_der[ipt1:ipt2])
    piecewise_fits[i] = fit_temp
    piecewise_fit_dist[ipt1:ipt2] =  fit_temp*np.ones_like(dset_der[ipt1:ipt2])


mean_pw_slope = np.mean(piecewise_fit_dist)
std_pw_slope = np.std(piecewise_fit_dist)
mad_slope = np.median(np.abs(np.median(dset_der)-dset_der))

piecewise_fits_dev = (piecewise_fits - np.median(dset_der)) / mad_slope
bout_ind = (piecewise_fits_dev < mad_thresh) #~z score of 1 #(mean_pw_slope - std_pw_slope)
bout_ind = bout_ind.astype(int)
bout_ind_diff = np.diff(bout_ind)

#plt.figure()
#plt.plot(bout_ind)

bouts_start_ind = np.where(bout_ind_diff == 1)[0] + 1 
bouts_end_ind = np.where(bout_ind_diff == -1)[0] + 1

#print(bouts_start_ind)
#print(bouts_end_ind)

if len(bouts_start_ind) != len(bouts_end_ind):
    minLength = np.min([len(bouts_start_ind), len(bouts_end_ind)])
    bouts_start_ind = bouts_start_ind[0:minLength]
    bouts_end_ind = bouts_end_ind[0:minLength]
    
#print(bouts_start_ind)
#print(bouts_end_ind)

changepts_array = np.asarray(changepts)
bouts_start = changepts_array[bouts_start_ind]
bouts_end = changepts_array[bouts_end_ind]

bouts = np.vstack((bouts_start, bouts_end))
print(bouts)
#print(changepts[bouts_start_ind])
#print(changepts[bouts_end_ind])
#bouts = changepts[bouts]

#---------------------------------------------------------------------------------------

plt.figure()
plt.plot(dset_denoised_med)
plt.plot(changepts[1:-1], dset_denoised_med[changepts[1:-1]], 'go')


f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)

ax1.plot(frames,dset)
for i in np.arange(bouts.shape[1]):
    ax1.plot(frames[bouts[0,i]:bouts[1,i]], dset[bouts[0,i]:bouts[1,i]],'r-')

ax2.plot(frames, dset_denoised_med)
for i in np.arange(bouts.shape[1]):
    ax2.plot(frames[bouts[0,i]:bouts[1,i]], dset_denoised_med[bouts[0,i]:bouts[1,i]],'r-')
    
    
#plt.plot(sp_signal(frames))
#plt.plot(changepts, dset_denoised[changepts], 'go')



