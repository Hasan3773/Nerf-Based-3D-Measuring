import numpy as np
import cv2
import glob

# used OpenCV camera calibration docs as guide

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*10,3), np.float32) # creates an empty array which will store checkerboard coordinates
objp[:,:2] = np.mgrid[0:7,0:10].T.reshape(-1,2) # populates x,y,z coords of each square into array 

# find corners of checkerboard
images = glob.glob('chessboards/*.jpg') # loads jpg into image variable

obj_points = [] # 3d point in real world space
img_points = [] # 2d points in image plane


for fname in images: 
    img = cv2.imread(fname)

    # finding chessboard corners in 2D space
    found, out_corners = cv2.findChessboardCorners(img, (7, 10))

    if found: 
        img_points.append(out_corners)        
        obj_points.append(objp)

    # draw and display corners
    cv2.drawChessboardCorners(img, (7, 10), out_corners, found)
    cv2.imshow('img', img)

    cv2.waitKey(0)

    # TODO OPTIONAL: cornerSubPix

print(np.array(img_points).shape)
print(img.shape)
# takes in list of obj points(3d), img points(2d), img size, runs calibration algo to find rvec, tvec, dist 
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img.shape[:2], None, None) # does pose est of checkerboard 

print(ret)
print(mtx)
print(dist)
print(rvecs)