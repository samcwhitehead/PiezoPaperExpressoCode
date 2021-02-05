# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 11:00:30 2017

@author: Saumya
"""

import cv2
import numpy as np
import glob

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((7*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2)

objpoints = [] 
imgpoints = [] 

images = glob.glob('C:\Users\Saumya\Google Drive\Yapici Lab\Camera Calibration\CalibImages\*.jpg')

for imagefile in images:
    img = cv2.imread(imagefile)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,7),None)
    
   
    if ret == True:
        objpoints.append(objp)
    
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)
        
        cv2.drawChessboardCorners(gray, (9,7), corners,ret)
        cv2.imshow('img',gray)
        cv2.waitKey(100)
        
cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

#reprojection error
mean_error = 0

for i in xrange(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print "total error: ", mean_error/len(objpoints)


#undistorting

#undistortIm = cv2.imread('C:\\Users\\Saumya\\Google Drive\\Yapici Lab\\Camera Calibration\\CalibImages\\toTest.jpg')
#h,  w = undistortIm.shape[:2]
#newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

#dst = cv2.undistort(undistortIm, mtx, dist, None, newcameramtx)
#cv2.imwrite('calibresult_toTest.png',dst)    