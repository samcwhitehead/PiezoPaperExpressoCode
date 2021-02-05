# -*- coding: utf-8 -*-
"""
Created on Mon Oct 02 12:22:52 2017

@author: Fruit Flies

Load and process Visual Expresso data
"""
#------------------------------------------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import cv2
import h5py

from scipy import signal, interpolate
from expresso_image_lib_mk3 import nan_helper
from expresso_image_lib_mk3 import idx_by_thresh
#------------------------------------------------------------------------------

#----------------------------------------------
# set path information for loading/saving data
#----------------------------------------------
DATA_PATH="F:\\Expresso GUI\\Imaging\\example_videos_expresso\\"
DATA_FILENAME="VExpressoTest2FixVod-Sam_testing_no_auto_exposure_TRACKING.hdf5"

SAVE_PATH = DATA_PATH 

#---------------------------------------------------------------
# set analysis parameters
#---------------------------------------------------------------
FLYNUM_RANGE = [0]      #number of flies that you want to analyze
SMOOTHING_FACTOR = 1    # degree of smoothing for interpolant spline [0, Inf)
MEDFILT_WINDOW = 31     # window for median filter, in units of frame number
#SAV_GOL_WINDOW = 101     # window of savitzky-golay filter
#SAV_GOL_ORDER = 3       # order of savitzky-golay filter

LABEL_FONTSIZE = 14     # for any plots that come up in the script
TITLE_FONTSIZE = 16

#----------------------------------------------
# flags for various debugging/analysis options
#----------------------------------------------
SAVE_DATA_FLAG = False  #save center of mass data to file
DEBUG_FLAG = False      #after analysis, play movie with track pts overlaid
PLOT_VEL_FLAG = True    #plot all velocity traces
PLOT_CM_FLAG = True     #plot x and y center of mass

#----------------------------------------------
# initialize storage for outputs (kind of)
#----------------------------------------------
xcm_list = [] 
ycm_list = []
xcm_smoothed_list = []
ycm_smoothed_list = [] 
xcm_vel_list = []
ycm_vel_list = []  

#------------------------------------------------------------------------------
# interpolate through nan values (missing track points)
def interp_nans(y,min_length=5):
    z = y.copy()
    nans, x = nan_helper(z)
    #eliminate small data chunks (likely spurious)
    not_nan_idx = idx_by_thresh(~nans)
    for nidx in not_nan_idx:    
        if len(nidx) < min_length:
            #print(nidx)
            z[nidx] = np.nan
            nans[nidx] = np.nan
    #interpolate through remaining points
    z[nans] = np.interp(x(nans),x(~nans),z[~nans])
    return z

#------------------------------------------------------------------------------

#=======================================
# load tracking data 
#=======================================

filename = os.path.join(DATA_PATH, DATA_FILENAME)

with h5py.File(filename,'r') as f:
    t = f['Time']['t'].value
    for flynum in FLYNUM_RANGE:
        xcm = f['BodyCM']['xcm_%02d'%(flynum)].value 
        ycm = f['BodyCM']['ycm_%02d'%(flynum)].value 
        
        xcm_list.append(xcm)
        ycm_list.append(ycm)
    
#=======================================
# Interpolate, filter, and smooth
#=======================================

for ith in np.arange(len(FLYNUM_RANGE)):
    xcm_curr = xcm_list[ith]
    ycm_curr = ycm_list[ith]
    
    # interpolate through nan values with a spline
    xcm_interp = interp_nans(xcm_curr)
    ycm_interp = interp_nans(ycm_curr)
    
    # apply median filter to data
    xcm_filt = signal.medfilt(xcm_interp,MEDFILT_WINDOW)
    ycm_filt = signal.medfilt(ycm_interp,MEDFILT_WINDOW)
    #xcm_filt = signal.savgol_filter(xcm_interp,SAV_GOL_WINDOW, SAV_GOL_ORDER)
    #ycm_filt = signal.savgol_filter(ycm_interp,SAV_GOL_WINDOW, SAV_GOL_ORDER)
    
    # fit smoothing spline to calculate derivative
    sp_xcm = interpolate.UnivariateSpline(t,xcm_filt,s=SMOOTHING_FACTOR)
    sp_ycm = interpolate.UnivariateSpline(t,ycm_filt,s=SMOOTHING_FACTOR)
    
    sp_xcm_vel = sp_xcm.derivative(n=1)  
    sp_ycm_vel = sp_ycm.derivative(n=1)
    
    # append to lists
    xcm_smooth = sp_xcm(t)
    ycm_smooth = sp_ycm(t)
    xcm_vel = sp_xcm_vel(t)
    ycm_vel = sp_ycm_vel(t)
    
    xcm_smoothed_list.append(xcm_smooth)
    ycm_smoothed_list.append(ycm_smooth)
    xcm_vel_list.append(xcm_vel)
    ycm_vel_list.append(ycm_vel)
    
    # check if over/under- smoothing
    if DEBUG_FLAG:
        fig_vs_t, (ax0,ax1) = plt.subplots(2,1,sharex=True,figsize=(12,7))
        
        #raw
        ax0.plot(t,xcm_curr,'k.',label='raw')
        ax1.plot(t,ycm_curr,'k.',label='raw')
        
        #smoothed
        ax0.plot(t,xcm_smooth,'r',label='smoothed')
        ax1.plot(t,ycm_smooth,'r',label='smoothed')
        
        #interpolated
        #ax0.plot(t,xcm_interp,'b:',label='interpolated')
        #ax1.plot(t,ycm_interp,'b:',label='interpolated')
        
        #filtered
        ax0.plot(t,xcm_filt,'g:',label='filtered')
        ax1.plot(t,ycm_filt,'g:',label='filtered')
        
        #ax0.set_xlabel('Time [s]',fontsize=LABEL_FONTSIZE)
        ax1.set_xlabel('Time [s]',fontsize=LABEL_FONTSIZE)
        ax0.set_ylabel('X [cm]',fontsize=LABEL_FONTSIZE)
        ax1.set_ylabel('Y [cm]',fontsize=LABEL_FONTSIZE)
        ax0.set_title('Fly Number {:02d}'.format(ith))
        
        legend0 = ax0.legend(loc='upper right')
        legend1 = ax1.legend(loc='upper right')
        
        ax0.set_xlim([np.amin(t),np.amax(t)])
        plt.tight_layout()

#=======================================
# Plot results (?)
#======================================= 
if PLOT_CM_FLAG:
    cmap = cm.get_cmap('Set1')
    cnorm = colors.Normalize(vmin=0.0, vmax=float(len(FLYNUM_RANGE))-1.0)
    for mth in np.arange(len(FLYNUM_RANGE)):
        
        rgb_vec = cmap(cnorm(FLYNUM_RANGE[mth]))[:3]
        rgb_vec = tuple([x for x in rgb_vec])
        
        # X vs t and Y vs t
        fig_vs_t, (ax0,ax1) = plt.subplots(2,1,sharex=True,figsize=(12,7))
        ax0.plot(t,xcm_list[mth],'k.',markersize=2,label='raw')
        ax1.plot(t,ycm_list[mth],'k.',markersize=2,label='raw')
        
        ax0.plot(t,xcm_smoothed_list[mth],color=rgb_vec,label='smoothed')
        ax1.plot(t,ycm_smoothed_list[mth],color=rgb_vec,label='smoothed')
    
        ax1.set_xlabel('Time [s]',fontsize=LABEL_FONTSIZE)
        ax0.set_ylabel('X [cm]',fontsize=LABEL_FONTSIZE)
        ax1.set_ylabel('Y [cm]',fontsize=LABEL_FONTSIZE)
        ax0.set_title('Fly Number {:02d}'.format(FLYNUM_RANGE[mth]))
        
        ax0.set_xlim([np.amin(t),np.amax(t)])
        legend0 = ax0.legend(loc='upper right')
        legend1 = ax1.legend(loc='upper right')
        plt.tight_layout()
        
        # Y vs X
        fig_spatial, ax = plt.subplots(figsize=(8.0,3.6))
        ax.plot(xcm_smoothed_list[mth],ycm_smoothed_list[mth],color=rgb_vec)
        ax.set_xlabel('X [cm]',fontsize=LABEL_FONTSIZE)
        ax.set_ylabel('Y [cm]',fontsize=LABEL_FONTSIZE)
        ax.set_title('Fly Number {:02d}'.format(FLYNUM_RANGE[mth]))
        plt.tight_layout()
        plt.axis('equal')
  
if PLOT_VEL_FLAG:    
    cmap = cm.get_cmap('Set1')
    cnorm = colors.Normalize(vmin=0.0, vmax=float(len(FLYNUM_RANGE))-1.0)
    for mth in np.arange(len(FLYNUM_RANGE)):
        
        rgb_vec = cmap(cnorm(FLYNUM_RANGE[mth]))[:3]
        rgb_vec = tuple([x for x in rgb_vec])
        
        # X vs t and Y vs t
        fig_vs_t, (ax0,ax1) = plt.subplots(2,1,sharex=True,figsize=(12,7))
        ax0.plot(t,xcm_vel_list[mth],color=rgb_vec)
        ax1.plot(t,ycm_vel_list[mth],color=rgb_vec)
        
        #ax0.set_xlabel('Time [s]',fontsize=LABEL_FONTSIZE)
        ax1.set_xlabel('Time [s]',fontsize=LABEL_FONTSIZE)
        ax0.set_ylabel('X Velocity [cm/s]',fontsize=LABEL_FONTSIZE)
        ax1.set_ylabel('Y Velocity [cm/s]',fontsize=LABEL_FONTSIZE)
        ax0.set_title('Fly Number {:02d}'.format(FLYNUM_RANGE[mth]))
        
        ax0.set_xlim([np.amin(t),np.amax(t)])
        plt.tight_layout()
    
         
    





