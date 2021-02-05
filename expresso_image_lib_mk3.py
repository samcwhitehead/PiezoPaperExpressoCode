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
# TO DO:
#   -undistort
#   -make sure bg subtract works when fly doesn't move much
#   -real distance (pix to cm)
#   -kalman filter and/or interpolation
#   -enhance contrast for background subtracted images?
#   -incorporate capillary tip into coordinates
#   -move execution of function to separate script
#   -incorporate time
#-----------------------------------------------------------------------------
# use camera calibration parameters to undistort image 
def undistort_im(img, mtx,dist,alpha=1):
    # get image dimensions    
    h, w = img.shape[:2]
    
    # new camera matrix    
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h),
                                                      centerPrincipalPoint=True)
    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    
    # crop the image
    #x,y,w,h = roi
    #dst = dst[y:y+h, x:x+w]
    return dst
    
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
    #idx_list = [x[1:] for x in idx_list]
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
    
    cv2.namedWindow('Select ROI (press enter when finished)', cv2.WINDOW_NORMAL)
    fromCenter = False
    showCrosshair = False
    r = cv2.selectROI('Select ROI (press enter when finished)',img,fromCenter,showCrosshair)
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
    win_name = 'get capillary tip (double click, enter when finished)'
    cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name,get_xy) 
    cv2.imshow(win_name,img)
    
    while True:
        cv2.imshow(win_name,img)
        try:
            img = clone.copy()
            cv2.circle(img,(mouseX,mouseY),4,(255,0,0),-1)
        except NameError:
            pass
        
        k = cv2.waitKey(1) & 0xFF
        if k == 13:
            break
        
    img = clone.copy()
    cv2.destroyAllWindows()
    
    return (mouseX,mouseY)

#-----------------------------------------------------------------------------
# draw a line with mouse
def draw_line(img,window_name):
    
    clone = img.copy()
    
    class LinePoints:
        line_points = []
        
    def select_line_pts(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            LinePoints.line_points = [] 
            LinePoints.line_points.append((x,y))
            cv2.circle(img,(x,y),2,(255,0,0),-1)
            
        elif event == cv2.EVENT_LBUTTONUP:
            LinePoints.line_points.append((x,y))
            cv2.line(img,LinePoints.line_points[-2],(x,y),(255,0,0))
    
    cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name,select_line_pts)
    while(1):
        cv2.imshow(window_name,img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            img = clone.copy()
        elif k == 13:
            break
    
    img = clone.copy()
    cv2.destroyAllWindows()
    #out_pts = np.asarray(LinePoints.line_points)
    return LinePoints.line_points
    
#-----------------------------------------------------------------------------
# define pixel to centimeter conversion
def get_pixel2cm(img, vial_length_cm=4.42, vial_width_cm=1.22):
    print('Draw line indicating vial length; press enter after completion')
    vial_length_pts = draw_line(img,'Vial length')
    vial_length_px = cv2.norm(vial_length_pts[0],vial_length_pts[1])
    print('Draw line indicating vial width; press enter after completion')
    vial_width_pts = draw_line(img,'Vial width')     
    vial_width_px = cv2.norm(vial_width_pts[0],vial_width_pts[1])
    
    print("vial length conversion: ", vial_length_cm/vial_length_px)
    print("vial width conversion: ", vial_width_cm/vial_width_px)
    
    pix2cm = np.mean([vial_length_cm/vial_length_px,vial_width_cm/vial_width_px])
    return pix2cm
#-----------------------------------------------------------------------------
# beter version of find background
def get_bg(filename,r,fly_size_range=[20,100],min_dist=80,morphSize=3,
               debugFlag=True, verbose=True):
                   
    if verbose:
        print("Finding background...")
               
    cap = cv2.VideoCapture(filename)
    N_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=100, 
                                              detectShadows=False)
    
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(morphSize,morphSize))
    
    x_cm = [] 
    y_cm = [] 
    framenum_list = []
    mean_intensity = [] 
    #min_dist = 70
    delta_cm = 0 
    cc = 0
    while (delta_cm < min_dist) and (cc < N_frames):
        ret, frame = cap.read()
    
        frame = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2]),:]
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_intensity.append(np.mean(frame_gray))
        fgmask = fgbg.apply(frame)
        
        #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        
        if debugFlag:
            cv2.imshow('frame',frame)
            cv2.imshow('foreground mask',fgmask)
        
            
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        
        if verbose and (np.mod(cc,50) == 0):
            print("Find BG: " + str(cc) + "/" +  str(N_frames) + " completed")
            
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
    
    #imt1 = cv2.normalize(imt1,imt1, 0, 255, cv2.NORM_MINMAX)
    #imt2 = cv2.normalize(imt2,imt2, 0, 255, cv2.NORM_MINMAX)
    imt1 = cv2.subtract(imt1,(np.mean(imt1)-130.0))
    imt2 = cv2.subtract(imt2,(np.mean(imt2)-130.0))
    
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
    return (bg, x_cm, y_cm, mean_intensity)

#-----------------------------------------------------------------------------
# get center of mass of fly from image ROI        
def get_cm(filename,bg,r,fly_size_range=[20,100],morphSize=5,min_thresh=10, 
           mean_intensity=130.0, debugFlag=True, verbose=True):
    
    if verbose:
        print('Finding center of mass coordinates...')           
    cap = cv2.VideoCapture(filename)
    N_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    connectivity = 4 
    
    #nan_count = 0 
    
    #min_thresh = 10
    
    x_cm = [np.nan]
    y_cm = [np.nan] 
    thresh_list = []
    x_ind = []
    y_ind = []
    #mean_intensity = [] 
    #std_intensity = [] 
    #.namedWindow('image')
    #cv2.namedWindow('thresh')
    for ith in range(1,N_frames-1):
        
        if verbose and (np.mod(ith,50) == 0):
           print("Find CM: " + str(ith) + "/" +  str(N_frames) + " completed")
        #print(ith)
        #im0 = get_cropped_im(ith-1,cap,r)
        im1 = get_cropped_im(ith,cap,r)
        #im1 = cv2.normalize(im1,im1, 0, 255, cv2.NORM_MINMAX)
        im1 = cv2.subtract(im1,(np.mean(im1)-130.0))
        
        #mean_intensity.append(np.mean(im1))
        #std_intensity.append(np.std(im1))
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
        #th_otsu = cv2.medianBlur(th_otsu,5)
        
        
        num_labels, labels, stats, centroids = \
            cv2.connectedComponentsWithStats(th_otsu,connectivity)
        
        if num_labels < 2:
            x_cm.append(np.nan)
            y_cm.append(np.nan)
            x_ind.append(np.nan)
            y_ind.append(np.nan)
            thresh_list.append(None)
            
            #nan_count += 1
            
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
                
                #x_cm_curr = centroids[cc_areas[0][0],0]
                #y_cm_curr = centroids[cc_areas[0][0],1]
                #nan_count = 0 
            else:
                x_cm.append(np.nan)
                y_cm.append(np.nan)
                
                x_ind.append(np.nan)
                y_ind.append(np.nan)
                
                thresh_list.append(np.nan)
                #nan_count += 1

#        if nan_count > 3 :
#            x_cm_curr = np.nan
#            y_cm_curr = np.nan
            
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
#if __name__ == "__main__":
#    num_ROI = 1
#    #filename = \
#    #    "C:\\Users\\samcw\\Desktop\\Expresso Imaging\\SampleForSam-06092017132717-0000.avi"
#    cmap = cm.get_cmap('Set1')
#    cnorm = colors.Normalize(vmin=0.0, vmax=float(num_ROI)-1.0)
#    filename = \
#        "F:\\Expresso GUI\\Imaging\\example_videos_expresso\\VExpressoTest2FixVod-Sam_testing_no_auto_exposure.avi"
#    calib_coeff_path = "C:\\Users\\Fruit Flies\\Documents\\Python Scripts\\Expresso GUI\\CalibImages\\calib_coeff.hdf5"    
#    
#    cm_savename = "F:\\Expresso GUI\\Imaging\\example_videos_expresso\\cm_data.hdf5"
#    roi_savename = "F:\\Expresso GUI\\Imaging\\example_videos_expresso\\roi_data.hdf5"
#    
#    # main vidoe cap
#    cap = cv2.VideoCapture(filename)
#    
#    # get undistortion parameters
#    with h5py.File(calib_coeff_path,'r') as f:
#        mtx = f['.']['mtx'].value 
#        dist = f['.']['dist'].value 
#    
#    #get base image from which to select ROIs
#    ret, frame = cap.read(1)
#    im0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    r_list = []
#    
#    #im0_undistort = undistort_im(im0, mtx,dist)
#    #cv2.imshow('distorted image', im0)
#    #cv2.imshow('undistorted image', im0_undistort)
#    #cv2.waitKey(0) 
#    
#    for ith in np.arange(num_ROI):
#        r = get_roi(im0)
#        r_list.append(r)
#        im0[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] = 0 
#    
#    bg_list = []
#    for jth in np.arange(num_ROI):
#        (bg, x_cm_guess, y_cm_guess,mean_intensity) = get_bg(filename,r_list[jth])
#        bg_list.append(bg)
#    
#    if False:
#        fig, ax = plt.subplots(5,2)
#        for kth in np.arange(num_ROI):
#            ax_curr = ax.ravel()[kth]
#            ax_curr.imshow(bg_list[kth])
#        
#    x_cm_list = []
#    y_cm_list = []
#    for qth in np.arange(num_ROI):
#        x_cm,y_cm,_,_,_,mean_intensity2,std_intensity2 = get_cm(filename,
#            bg_list[qth], r_list[qth],mean_intensity=np.median(mean_intensity))
#        x_cm_list.append(x_cm)
#        y_cm_list.append(y_cm)
#    
#    
#    # save data
##    with h5py.File(cm_savename,'w') as f:
##        for dset_num in np.arange(num_ROI):
##            f.create_dataset('xcm_%02d'%(dset_num), data=x_cm_list[dset_num])
##            f.create_dataset('ycm_%02d'%(dset_num), data=y_cm_list[dset_num])
##    with h5py.File(roi_savename, 'w') as g:
##        for roi_num in np.arange(num_ROI):
##            g.create_dataset('roi_%02d'%(ith),data=r)
#                          
#    x_cm_list_interp = []
#    y_cm_list_interp = []
#    for rth in np.arange(num_ROI):
#        x_cm_temp = x_cm_list[rth]
#        y_cm_temp = y_cm_list[rth]
#        
#        x_cm_temp = fill_nan(x_cm_temp)
#        y_cm_temp = fill_nan(y_cm_temp)
#        
#        x_cm_list_interp.append(x_cm_temp)
#        y_cm_list_interp.append(y_cm_temp)
#        
#        #fig, ax = plt.subplots()
#        #plt.plot(x_cm,y_cm,'k.')
#        #plt.plot(x_cm_list[rth], y_cm_list[rth],'b.')
#        
#       
#    #mov_name = 'C:\\Users\\samcw\\Desktop\\Expresso Imaging\\test_track_2\\test_track_%03d.png'
#    #N_mov_frames = 500
#    #fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#    #out = cv2.VideoWriter(mov_name,fourcc, 30.0, im0.shape)
#    
#    cv2.namedWindow('test movie')
#    #bad_frame = []
#    #for frm_num in np.arange(N_mov_frames):
#    #    cap.set(1,frm_num)
#    #    _, frame = cap.read()
#    cc = 0
#    while(1):
#        cap.set(1,cc)
#        _, frame = cap.read()
#        #frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
#       
#        for roi_num in np.arange(num_ROI):
#            r_curr = r_list[roi_num]
#            
#            bgr_vec = cmap(cnorm(roi_num))[:3]
#            bgr_vec = tuple([255*x for x in bgr_vec])
#            bgr_vec = bgr_vec[::-1]
#            for xcm_curr, ycm_curr in zip(x_cm_list_interp[roi_num][:cc],y_cm_list_interp[roi_num][:cc]):
#                #xcm_curr = int(x_cm_list[roi_num][cc])
#                #ycm_curr = int(y_cm_list[roi_num][cc])
#                try:
#                    xcm_curr_int = int(xcm_curr)
#                    ycm_curr_int = int(ycm_curr)
#                    cv2.circle(frame,(xcm_curr_int+int(r_curr[0]),ycm_curr_int+int(r_curr[1])),1,bgr_vec,-1)
#                except ValueError:
#                    print("dropped frame: {}".format(np.max(cc)))
#        #out.write(frame)
#        cv2.imshow('test movie',frame)
#        #cv2.imwrite('F:\\Expresso GUI\\Imaging\\test output\\test_im.png',frame)
#        key = cv2.waitKey(10) & 0xFF
#        if key == 27 :
#            break
#        cc+=1
#    #out.release()   
#        
#    cap.release()    
#    cv2.destroyAllWindows()
#        #x_cm,y_cm,_,_,_ = get_cm(filename,bg,r_list[jth])
#        
#        