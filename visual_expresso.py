# -*- coding: utf-8 -*-
"""
Created on Sun Oct 01 20:59:03 2017

@author: Fruit Flies

Script to run analysis on Expresso videos
"""
#------------------------------------------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import cv2
import h5py
from expresso_image_lib_mk3 import get_pixel2cm
from expresso_image_lib_mk3 import get_cap_tip
from expresso_image_lib_mk3 import get_bg
from expresso_image_lib_mk3 import get_cm
from expresso_image_lib_mk3 import get_roi
from expresso_image_lib_mk3 import get_cropped_im
from expresso_image_lib_mk3 import undistort_im

#------------------------------------------------------------------------------

#----------------------------------------------
# set path information for loading/saving data
#----------------------------------------------
DATA_PATH = "F:\\Expresso GUI\\Imaging\\example_videos_expresso\\"
DATA_FILENAME = "VExpressoTest2FixVod-Sam_testing_no_auto_exposure.avi"
CALIB_COEFF_PATH = os.path.join("C:\\Users\\Fruit Flies\\Documents\\",
                  "Python Scripts\\Expresso GUI\\CalibImages\\calib_coeff.hdf5")  
SAVE_PATH = DATA_PATH 

#---------------------------------------------------------------
# select number of vials to analyze and set analysis parameters
#---------------------------------------------------------------
NUM_ROI = 1 
FLY_SIZE_RANGE = [20, 100] #still in pixels
PIX2CM =  0.02 # fill in if known, otherwise estimated by script
T_OFFSET = 0 # to be filled later--there's a delay between video and Expresso

LABEL_FONTSIZE = 14 # for any plots that come up in the script
TITLE_FONTSIZE = 16

#----------------------------------------------
# flags for various debugging/analysis options
#----------------------------------------------
DEBUG_BG_FLAG = True       #show image and estimated foreground during find_bg
DEBUG_CM_FLAG = True       #show images during get_cm
UNDISTORT_FLAG = False      #undistort images using calibration coefficients
SAVE_DATA_FLAG = False      #save center of mass data to file
SHOW_RESULTS_FLAG = False   #after analysis, play movie with track pts overlaid
PLOT_BG_FLAG = False        #plot all ROI backgrounds
PLOT_CM_FLAG = False        #plot x and y center of mass

#----------------------------------------------
# initialize storage for outputs (kind of)
#----------------------------------------------
roi_list = []
cap_tip_list = []
bg_list = []
xcm_list = []
ycm_list = [] 
mean_intensity_list = [] 

#------------------------------------------------------------------------------

#=======================================
# load video and get proper units/ROIs
#=======================================
filename = os.path.join(DATA_PATH, DATA_FILENAME)
cap = cv2.VideoCapture(filename)
N_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
FPS = cap.get(cv2.CAP_PROP_FPS)

#get base image from which to select ROIs
ret, frame = cap.read(1)
im0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
if UNDISTORT_FLAG:
    #under construction 
    with h5py.File(CALIB_COEFF_PATH,'r') as f:
        mtx = f['.']['mtx'].value 
        dist = f['.']['dist'].value 
    im0 = undistort_im(im0,mtx,dist)

im_for_roi = im0.copy()    
im_for_meas = im0.copy()

# determine pixel to centimeter conversion 
if not PIX2CM:
    PIX2CM = get_pixel2cm(im_for_meas)
    
# get ROI and capillary tip location for each fly    
for ith in np.arange(NUM_ROI):
    r = get_roi(im0)
    roi_list.append(r)
    im0[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] = 0  #black out
    
    im_roi = get_cropped_im(1,cap,r) 
    cap_tip = get_cap_tip(im_roi)
    cap_tip_list.append(cap_tip)

#=======================================
# Estimate static background
#=======================================
for jth in np.arange(NUM_ROI):
        bg, _, _,mean_intensity = get_bg(filename,roi_list[jth],
                     fly_size_range=FLY_SIZE_RANGE, debugFlag = DEBUG_BG_FLAG)
        bg_list.append(bg)
        mean_intensity_list.append(np.mean(mean_intensity))

# if you want to plot all backgrounds
if PLOT_BG_FLAG:
    fig, ax = plt.subplots(5,2)
    for kth in np.arange(NUM_ROI):
        ax_curr = ax.ravel()[kth]
        ax_curr.imshow(bg_list[kth])
        ax_curr.set_title('Fly Number {:02d}'.format(kth))
    plt.tight_layout()    
    
#=======================================
# Get center of mass coordinates
#=======================================
for kth in np.arange(NUM_ROI):
    mean_intensity_curr = mean_intensity_list[kth]
    xcm, ycm,_,_,_ = get_cm(filename, bg_list[kth],roi_list[kth], 
                            mean_intensity=mean_intensity_curr,
                            debugFlag=DEBUG_CM_FLAG)
    
    xcm_transformed = (PIX2CM*(xcm - cap_tip[0]))[T_OFFSET:]
    ycm_transformed = (PIX2CM*(ycm - cap_tip[1]))[T_OFFSET:]                         
    
    xcm_list.append(xcm_transformed)
    ycm_list.append(ycm_transformed)

# make arrays for frame number and real time
frame_nums = np.arange(N_FRAMES-1)
frame_nums= frame_nums[T_OFFSET:] #once T_OFFSET is defined
t = frame_nums/FPS


if PLOT_CM_FLAG:
    cmap = cm.get_cmap('Set1')
    cnorm = colors.Normalize(vmin=0.0, vmax=float(NUM_ROI)-1.0)
    for mth in np.arange(NUM_ROI):
        
        rgb_vec = cmap(cnorm(mth))[:3]
        rgb_vec = tuple([x for x in rgb_vec])
        
        # X vs t and Y vs t
        fig_vs_t, (ax0,ax1) = plt.subplots(2,1,sharex=True,figsize=(12,7))
        ax0.plot(t,xcm_list[mth],color=rgb_vec)
        ax1.plot(t,ycm_list[mth],color=rgb_vec)
        
        #ax0.set_xlabel('Time [s]',fontsize=LABEL_FONTSIZE)
        ax1.set_xlabel('Time [s]',fontsize=LABEL_FONTSIZE)
        ax0.set_ylabel('X [cm]',fontsize=LABEL_FONTSIZE)
        ax1.set_ylabel('Y [cm]',fontsize=LABEL_FONTSIZE)
        ax0.set_title('Fly Number {:02d}'.format(mth))
        
        ax0.set_xlim([np.amin(t),np.amax(t)])
        plt.tight_layout()
        
        # X vs Y
        fig_spatial, ax = plt.subplots(figsize=(8.0,3.6))
        ax.plot(xcm_list[mth],ycm_list[mth],color=rgb_vec)
        ax.set_xlabel('X [cm]',fontsize=LABEL_FONTSIZE)
        ax.set_ylabel('Y [cm]',fontsize=LABEL_FONTSIZE)
        ax.set_title('Fly Number {:02d}'.format(mth))
        plt.tight_layout()
        plt.axis('equal')
#=======================================
# Save results 
#=======================================
if SAVE_DATA_FLAG:
    savename_prefix = os.path.splitext(DATA_FILENAME)[0]
    save_filename = os.path.join(SAVE_PATH,savename_prefix + "_TRACKING.hdf5")
    
    with h5py.File(save_filename,'w') as f:
        f.create_dataset('Time/t', data=t)
        f.create_dataset('Time/frame_num', data=frame_nums)
        f.create_dataset('Units/pix2cm', data=PIX2CM)
        for dset_num in np.arange(NUM_ROI):
            f.create_dataset('BodyCM/xcm_%02d'%(dset_num), 
                             data=xcm_list[dset_num])
            f.create_dataset('BodyCM/ycm_%02d'%(dset_num),
                             data=ycm_list[dset_num])
            f.create_dataset('ROI/roi_%02d'%(dset_num),
                             data=roi_list[dset_num])                 
            f.create_dataset('BG/bg_%02d'%(dset_num),
                             data=bg_list[dset_num])
            f.create_dataset('CAP_TIP/cap_tip_%02d'%(dset_num),
                             data=cap_tip_list[dset_num])

#=======================================
# Showing tracking results on frame
#=======================================
if SHOW_RESULTS_FLAG:
    cmap = cm.get_cmap('Set1')
    cnorm = colors.Normalize(vmin=0.0, vmax=float(NUM_ROI)-1.0)
    
    cv2.namedWindow('Tracking results')
       
    cc = 0
    while(1):
        cap.set(1,cc)
        _, frame = cap.read()
       
        for roi_num in np.arange(NUM_ROI):
            r_curr = roi_list[roi_num]
            
            bgr_vec = cmap(cnorm(roi_num))[:3]
            bgr_vec = tuple([255*x for x in bgr_vec])
            bgr_vec = bgr_vec[::-1]
            
            for xcm_curr, ycm_curr in zip(xcm_list[roi_num][cc],ycm_list[roi_num][cc]):
                try:
                    xcm_curr_int = int(xcm_curr)
                    ycm_curr_int = int(ycm_curr)
                    cv2.circle(frame,(xcm_curr_int+int(r_curr[0]),
                                     ycm_curr_int+int(r_curr[1])),1,bgr_vec,-1)
                except ValueError:
                    continue
        
        cv2.imshow('Tracking results',frame)
        #cv2.imwrite(os.path.join(SAVE_PATH,'im_{:04d}'.format(cc)),frame)
        key = cv2.waitKey(10) & 0xFF
        if key == 27 :
            break
        cc+=1
        
#=======================================
# Close windows and release cv2 object
#=======================================
cap.release()    
cv2.destroyAllWindows()
    
    


