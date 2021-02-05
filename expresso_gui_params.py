# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 12:18:21 2017

@author: Fruit Flies
"""
#------------------------------------------------------------------------------
#
#   wlevel = wavelet decomposition threshold level (e.g. 2, 3 ,4)
#   wtype = wavelet type for denoising (e.g. 'haar', 'db4', sym5')
#   medfilt_window = window size for median filter used on wavelet-denoised 
#       data (e.g. 11, 13, ...)
#   mad_thresh = threshold for the median absolute deviation of slopes in 
#       segmented dataset. slopes below this value are considered possible bouts
#   var_user = user set variation fed into the PELT change point detector. 
#
#------------------------------------------------------------------------------
analysisParams = {'wlevel' : 2 ,
             'wtype' : 'db4' ,
             'medfilt_window' : 11 ,
             'mad_thresh' : -8 ,
             'var_user' : 0.5 ,
             'min_bout_duration': 2 ,
             'min_bout_volume': 6}
             
guiParams = {'bgcolor' : '#3e3d42' ,
             'listbgcolor': '#222222' , 
             'textcolor' : '#ffffff' ,
             'buttontextcolor': '#fff7bc' ,
             'buttonbgcolor': '#222222' ,
             'plotbgcolor' : '#3e3d42' , 
             'plottextcolor' : '#c994c7' }