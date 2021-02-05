# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 22:42:57 2017

@author: samcw
"""

import numpy as np
import cv2
from scipy.spatial.distance import pdist, squareform
import h5py

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# return a cropped, grayscale image specified by an ROI=r
def get_cropped_im(framenum,cap,r):
    
    cap.set(1,framenum)
    _, frame = cap.read()
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    return imCrop

#-----------------------------------------------------------------------------
# plot cropped image with cm marked on it
def plot_im_and_cm(framenum,x_cm,y_cm,cap,r):
    imCrop = get_cropped_im(framenum,cap,r)
    imsize_row, imsize_col = imCrop.shape
    fig, ax = plt.subplots()
    ax.imshow(imCrop)
    ax.plot(y_cm[framenum],x_cm[framenum],'rx')

#-----------------------------------------------------------------------------
# tool for pulling out indices of an array with elements above thresh value
def idx_by_thresh(signal,thresh = 0.1):
    import numpy as np
    idxs = np.squeeze(np.argwhere(signal > thresh))
    try:
        split_idxs = np.squeeze(np.argwhere(np.diff(idxs) > 1))
    except IndexError:
        #print 'IndexError'
        return None
    #split_idxs = [split_idxs]
    if split_idxs.ndim == 0:
        split_idxs = np.array([split_idxs])
    #print split_idxs
    try:
        idx_list = np.split(idxs,split_idxs)
    except ValueError:
        #print 'value error'
        np.split(idxs,split_idxs)
        return None
    idx_list = [x[1:] for x in idx_list]
    idx_list = [x for x in idx_list if len(x)>0]
    return idx_list

#-----------------------------------------------------------------------------
# handle nans
def nan_helper(y):
    return np.isnan(y),lambda z: z.nonzero()[0]

#-----------------------------------------------------------------------------
# interpolate over nan values if the sequence of nans has length < min_length
def fill_nan(y,min_diff=5):
    z = y.copy()
    nans, x = nan_helper(z)
    nan_idx = idx_by_thresh(nans)
    for nidx in nan_idx:    
        if len(nidx) < min_diff:
            nan_log_ind = np.zeros(nans.shape,dtype='bool')
            nan_log_ind[nidx] = True
            z[nan_log_ind] = np.interp(x(nan_log_ind),x(~nans),z[~nans])
    #y[nans] = np.interp(x(nans),x(~nans),y[~nans])
    return z

#-----------------------------------------------------------------------------
# user defined ROI
def get_roi(img):
    #cap = cv2.VideoCapture(filename)

    #ret, frame = cap.read(1)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #cv2.imshow('frame',gray)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
    fromCenter = False
    showCrosshair = False
    r = cv2.selectROI("ROI",img,fromCenter,showCrosshair)
    #cv2.waitKey(0)
    cv2.destroyAllWindows()
    return r

#-----------------------------------------------------------------------------
# callback function to return x,y coordinates of a mouse DOUBLE LEFT CLICK 
def get_xy(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print((x,y))
        mouseX,mouseY = x,y
        #param = (x,y)
        #cv2.circle(img,(mouseX,mouseY),4,(255,0,0),1)
        #cv2.imshow('get capillary tip',img)
        
#-----------------------------------------------------------------------------
# get capillary tip manually        
def get_cap_tip(img):
    clone = img.copy()
    cv2.namedWindow('get capillary tip')
    cv2.setMouseCallback('get capillary tip',get_xy) 
    cv2.imshow('get capillary tip',img)
    
    while True:
        try:
            cv2.circle(img,(mouseX,mouseY),4,(255,0,0),-1)
        except NameError:
            pass
        cv2.imshow('get capillary tip',img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            img = clone
        elif key == ord("c"):
            break
       
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return (mouseX,mouseY)

#-----------------------------------------------------------------------------
# original version of find static background    
def get_bg(filename,r,thresh_val=80):
    #is it necessary to loop through all the images?
    cap = cv2.VideoCapture(filename)
    #r = get_roi(filename)
    
    #get video parameters
    N_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #fps = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    #im_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #im_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    #initialize cm list. note that x and y are in image coordinates
    x_cm = [] 
    y_cm = [] 
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    
    for ith in range(0,N_frames): #,int(round(N_frames/100))):
        #print(ith)
        imCrop = get_cropped_im(ith,cap,r)
        _,th = cv2.threshold(imCrop,thresh_val,255,cv2.THRESH_BINARY_INV)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
        if np.max(th) < 255:
            x_cm.append(np.nan)  
            y_cm.append(np.nan)   
        else:
            fly_ind = np.where(th==255)
            fly_ind_cm = np.mean(fly_ind,axis=1)
            x_cm.append(fly_ind_cm[0])
            y_cm.append(fly_ind_cm[1])
        
    x_cm = np.asarray(x_cm)
    y_cm = np.asarray(y_cm)
    
    #find two frames with most distant centers of mass
    fly_cm = np.transpose(np.vstack((x_cm,y_cm)))
    D = pdist(fly_cm)
    D = squareform(D);
    max_dist = np.nanmax(D)
    ind = np.where(D==max_dist)[0]
    t1 = ind[0]
    t2 = ind[1]
    
    #get background from combination of frames with distant cm
    dx = np.abs(x_cm[t2] - x_cm[t1])
    dy = np.abs(y_cm[t2] - y_cm[t1])
    
    xmid = int((x_cm[t2] + x_cm[t1])/2)
    ymid = int((y_cm[t2] + y_cm[t1])/2)
    
    imt1 = get_cropped_im(t1,cap,r)
    imt2 = get_cropped_im(t2,cap,r)
    
    bg = imt1
    if dx >= dy:
        if x_cm[t1] < xmid:
            bg[:xmid,:] = imt2[:xmid,:]
        else:
            bg[xmid:,:] = imt2[xmid:,:]
    else:
        if y_cm[t1] < ymid:
            bg[:,:ymid] = imt2[:,:ymid]
        else:
            bg[:,ymid:] = imt2[:,ymid:]
    
    cap.release()      
    return (bg, x_cm, y_cm)

#-----------------------------------------------------------------------------
# beter version of find background
def get_bg_alt(filename,r,fly_size_range=[20,100],min_dist=70,morphSize=3,
               debugFlag=True):
    cap = cv2.VideoCapture(filename)
    N_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=100, \
                                              detectShadows=False)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(morphSize,morphSize))
    
    x_cm = [] 
    y_cm = [] 
    framenum_list = []
    #min_dist = 70
    delta_cm = 0 
    cc = 0
    while (delta_cm < min_dist) and (cc < N_frames):
        ret, frame = cap.read()
    
        frame = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2]),:]
        fgmask = fgbg.apply(frame)
        
        #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        
        if debugFlag:
            cv2.imshow('fgmask',frame)
            cv2.imshow('frame',fgmask)
        
            
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        if (np.sum(fgmask) > 255*fly_size_range[0]) and (np.sum(fgmask) < 255*fly_size_range[1]):
            #print(np.sum(fgmask))
            framenum_list.append(cc)
            
            fly_ind = np.where(fgmask==255)
            fly_ind_cm = np.mean(fly_ind,axis=1)
            x_cm.append(fly_ind_cm[0])
            y_cm.append(fly_ind_cm[1])
            
            if len(x_cm) > 1:
                fly_cm = np.transpose(np.vstack((np.asarray(x_cm),np.asarray(y_cm))))
                D = pdist(fly_cm)
                #D = squareform(D);
                delta_cm = np.nanmax(D)
            
        cc+=1
    
    D = squareform(D)    
    ind = np.where(D==delta_cm)[0]
    t1 = ind[0]
    t2 = ind[1]
    
    #get background from combination of frames with distant cm
    dx = np.abs(x_cm[t2] - x_cm[t1])
    dy = np.abs(y_cm[t2] - y_cm[t1])
    
    xmid = int((x_cm[t2] + x_cm[t1])/2)
    ymid = int((y_cm[t2] + y_cm[t1])/2)
    
    imt1 = get_cropped_im(framenum_list[t1],cap,r)
    imt2 = get_cropped_im(framenum_list[t2],cap,r)
    
    bg = imt1
    if dx >= dy:
        if x_cm[t1] < xmid:
            bg[:xmid,:] = imt2[:xmid,:]
        else:
            bg[xmid:,:] = imt2[xmid:,:]
    else:
        if y_cm[t1] < ymid:
            bg[:,:ymid] = imt2[:,:ymid]
        else:
            bg[:,ymid:] = imt2[:,ymid:]
    cap.release()
    cv2.namedWindow('Background', cv2.WINDOW_NORMAL)
    cv2.imshow('Background',bg)
    return (bg, x_cm, y_cm)

#-----------------------------------------------------------------------------
# get center of mass of fly from image ROI        
def get_cm(filename,bg,r,fly_size_range=[20,100],morphSize=5,min_thresh=10, 
           debugFlag=True):
               
    cap = cv2.VideoCapture(filename)
    N_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    connectivity = 4 
    
    nan_count = 0 
    
    #min_thresh = 10
    
    x_cm = [np.nan]
    y_cm = [np.nan] 
    thresh_list = []
    x_ind = []
    y_ind = []
    #.namedWindow('image')
    #cv2.namedWindow('thresh')
    for ith in range(1,N_frames-1):
        #print(ith)
        #im0 = get_cropped_im(ith-1,cap,r)
        im1 = get_cropped_im(ith,cap,r)
        #im2 = get_cropped_im(ith+1,cap,r)
        #if isinstance(im, tuple):
        #    print(ith)
        #mask = np.zeros(im.shape,dtype = bool)
        #im_minus_bg0 = cv2.subtract(bg,im0)
        im_minus_bg = cv2.absdiff(bg,im1)
        #im_minus_bg2 = cv2.subtract(bg,im2)
        
        #im_minus_bg = cv2.add(im_minus_bg0,cv2.add(im_minus_bg1,im_minus_bg2))
        #im_minus_bg2 = np.zeros(im_minus_bg.shape, dtype='uint8')
        
        
        #im_minus_bg2[x1:x2,y1:y2] = im_minus_bg[x1:x2,y1:y2])
        #morphologically open image?
        #kernel_rad = 2
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_rad,kernel_rad))
        #im_minus_bg = cv2.morphologyEx(im_minus_bg, cv2.MORPH_OPEN, kernel)
        im_minus_bg = cv2.GaussianBlur(im_minus_bg,(morphSize,morphSize),0)
        
        otsu_thresh, _ = cv2.threshold(im_minus_bg.astype('uint8'), \
                                0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        _, th_otsu = cv2.threshold(im_minus_bg.astype('uint8'),np.max((otsu_thresh,min_thresh)), \
                                255, cv2.THRESH_BINARY)
        #otsu_thresh, th_otsu = cv2.threshold(im_minus_bg.astype('uint8'), \
        #                        80, 255, cv2.THRESH_BINARY)
        """
        if ~np.isnan(x_cm_curr):
            x1 = int(np.max((y_cm_curr-W,0)))
            x2 = int(np.min((y_cm_curr+W,im_minus_bg.shape[0])))
            y1 = int(np.max((x_cm_curr-W,0)))
            y2 = int(np.min((x_cm_curr+W,im_minus_bg.shape[1])))
            print(x1)
            print(x2)
            print(y1)
            print(y2)
            mask = np.zeros(im_minus_bg.shape,dtype='uint8')
            mask[x1:x2,y1:y2] = 255
            cv2.imshow('mask',mask)
            th_otsu = cv2.bitwise_and(mask,th_otsu)
        """
        #th_otsu = cv2.medianBlur(th_otsu,5)
        
        
        num_labels, labels, stats, centroids = \
            cv2.connectedComponentsWithStats(th_otsu,connectivity)
        
        if num_labels < 2:
            x_cm.append(np.nan)
            y_cm.append(np.nan)
            x_ind.append(np.nan)
            y_ind.append(np.nan)
            thresh_list.append(None)
            
            nan_count += 1
            
        else:
            cc_areas = [(idx,a) for idx,a in list(enumerate(stats[:,cv2.CC_STAT_AREA])) \
                        if a > fly_size_range[0] and a < fly_size_range[1]]
            
            if len(cc_areas) == 1 :
                x_cm.append(centroids[cc_areas[0][0],0])
                y_cm.append(centroids[cc_areas[0][0],1])
                
                fly_ind = np.where(labels == cc_areas[0][0])
                x_ind.append(fly_ind[0])
                y_ind.append(fly_ind[1])
                
                thresh_list.append(otsu_thresh)
                
                x_cm_curr = centroids[cc_areas[0][0],0]
                y_cm_curr = centroids[cc_areas[0][0],1]
                nan_count = 0 
            else:
                x_cm.append(np.nan)
                y_cm.append(np.nan)
                
                x_ind.append(np.nan)
                y_ind.append(np.nan)
                
                thresh_list.append(np.nan)
                nan_count += 1

        if nan_count > 3 :
            x_cm_curr = np.nan
            y_cm_curr = np.nan
            
        if debugFlag:    
            try:
                cv2.circle(im1,(int(x_cm[ith]),int(y_cm[ith])),2,(255,0,0),-1)
                cv2.circle(th_otsu,(int(x_cm[ith]),int(y_cm[ith])),2,(255,0,0),-1)
            except ValueError:
                print(ith)
            cv2.imshow('image',im1)
            cv2.imshow('thresh',th_otsu)
            cv2.imshow('im_minus_bg',im_minus_bg)
            
            k = cv2.waitKey(20) & 0xff
            if k == 27:
                break
               
    x_cm = np.array(x_cm,dtype=np.float)
    y_cm = np.array(y_cm,dtype=np.float)
    
    cap.release()
    return (x_cm,y_cm,x_ind,y_ind,thresh_list)

#------------------------------------------------------------------------------
if __name__ == "__main__":
    num_ROI = 1
    #filename = \
    #    "C:\\Users\\samcw\\Desktop\\Expresso Imaging\\SampleForSam-06092017132717-0000.avi"
    cmap = cm.get_cmap('Set1')
    cnorm = colors.Normalize(vmin=0.0, vmax=float(num_ROI)-1.0)
    filename = \
        "F:\\Expresso GUI\\Imaging\\VExpressoTest3Vod-09012017163301-0000.avi"
    
    cm_savename = "F:\\Expresso GUI\\Imaging\\test output\\cm_data.hdf5"
    roi_savename = "F:\\Expresso GUI\\Imaging\\test output\\cm_data.hdf5"
    
    cap = cv2.VideoCapture(filename)
    
    #get base image from which to select ROIs
    ret, frame = cap.read(1)
    im0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    r_list = []
    
    
    for ith in np.arange(num_ROI):
        r = get_roi(im0)
        r_list.append(r)
        im0[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] = 0 
    
    bg_list = []
    for jth in np.arange(num_ROI):
        (bg, x_cm_guess, y_cm_guess) = get_bg_alt(filename,r_list[jth])
        bg_list.append(bg)
    
    if False:
        fig, ax = plt.subplots(5,2)
        for kth in np.arange(num_ROI):
            ax_curr = ax.ravel()[kth]
            ax_curr.imshow(bg_list[kth])
        
    x_cm_list = []
    y_cm_list = []
    for qth in np.arange(num_ROI):
        x_cm,y_cm,_,_,_ = get_cm(filename,bg_list[qth],r_list[qth])
        x_cm_list.append(x_cm)
        y_cm_list.append(y_cm)
    
    
    # save data
    with h5py.File(cm_savename,'w') as f:
        for dset_num in np.arange(num_ROI):
            f.create_dataset('xcm_%02d'%(dset_num), data=x_cm_list[dset_num])
            f.create_dataset('ycm_%02d'%(dset_num), data=y_cm_list[dset_num])
    with h5py.File(roi_savename, 'w') as g:
        for roi_num in np.arange(num_ROI):
            g.create_dataset('roi_%02d'%(ith),data=r)
                          
    x_cm_list_interp = []
    y_cm_list_interp = []
    for rth in np.arange(num_ROI):
        x_cm_temp = x_cm_list[rth]
        y_cm_temp = y_cm_list[rth]
        
        x_cm_temp = fill_nan(x_cm_temp)
        y_cm_temp = fill_nan(y_cm_temp)
        
        x_cm_list_interp.append(x_cm_temp)
        y_cm_list_interp.append(y_cm_temp)
        
        #fig, ax = plt.subplots()
        #plt.plot(x_cm,y_cm,'k.')
        #plt.plot(x_cm_list[rth], y_cm_list[rth],'b.')
        
       
    #mov_name = 'C:\\Users\\samcw\\Desktop\\Expresso Imaging\\test_track_2\\test_track_%03d.png'
    #N_mov_frames = 500
    #fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    #out = cv2.VideoWriter(mov_name,fourcc, 30.0, im0.shape)
    
    cv2.namedWindow('test movie')
    #bad_frame = []
    #for frm_num in np.arange(N_mov_frames):
    #    cap.set(1,frm_num)
    #    _, frame = cap.read()
    cc = 0
    while(1):
        cap.set(1,cc)
        _, frame = cap.read()
        #frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
       
        for roi_num in np.arange(num_ROI):
            r_curr = r_list[roi_num]
            
            bgr_vec = cmap(cnorm(roi_num))[:3]
            bgr_vec = tuple([255*x for x in bgr_vec])
            bgr_vec = bgr_vec[::-1]
            for xcm_curr, ycm_curr in zip(x_cm_list_interp[roi_num][:cc],y_cm_list_interp[roi_num][:cc]):
                #xcm_curr = int(x_cm_list[roi_num][cc])
                #ycm_curr = int(y_cm_list[roi_num][cc])
                try:
                    xcm_curr_int = int(xcm_curr)
                    ycm_curr_int = int(ycm_curr)
                    cv2.circle(frame,(xcm_curr_int+int(r_curr[0]),ycm_curr_int+int(r_curr[1])),1,bgr_vec,-1)
                except ValueError:
                    print("dropped frame: {}".format(np.max(cc)))
        #out.write(frame)
        cv2.imshow('test movie',frame)
        #cv2.imwrite('F:\\Expresso GUI\\Imaging\\test output\\test_im.png',frame)
        key = cv2.waitKey(10) & 0xFF
        if key == 27 :
            break
        cc+=1
    #out.release()   
        
    cap.release()    
    cv2.destroyAllWindows()
        #x_cm,y_cm,_,_,_ = get_cm(filename,bg,r_list[jth])
        
        