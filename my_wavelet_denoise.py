# -*- coding: utf-8 -*-
"""
Created on Mon Jan 09 11:33:19 2017

@author: Fruit Flies
"""
import numpy as np
import pywt
from pywt import threshold as pywtthresh
from statsmodels.robust import mad
import matplotlib.pyplot as plt

def wavelet_denoise(data,wtype='db4',wlevel=2,plotFlag=False):
    
    coeffs = pywt.wavedec(np.squeeze(data),wtype, level=wlevel)
    sigma = mad(coeffs[-1])
    uthresh = sigma*np.sqrt(2*np.log(data.size))
    
    denoised = coeffs[:]
    denoised[1:] = ( pywtthresh(i, value=uthresh, mode='soft') for i in denoised[1:])
    
    data_denoised = pywt.waverec(denoised, wtype)
    
    if plotFlag:
        plt.figure()
        plt.plot(data)
        plt.plot(data_denoised,'r')
    
    if (data.size != data_denoised.size):
        data_denoised = data_denoised[0:data.size]
    return data_denoised