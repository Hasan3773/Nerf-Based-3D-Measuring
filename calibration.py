import numpy as np
import cv2
import glob

# used OpenCV camera calibration docs as guide

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*10,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:10].T.reshape(-1,2)

# find corners of checkerboard
images = glob.glob('chessboards/chessboard*.jpg')

obj_points = [] # 3d point in real world space
img_points = [] # 2d points in image plane


for fname in images: 
    img = cv2.imread(fname)
    # grey = cv2.cvtColor(img, cv.COLOR_BGR2GRAY)

    # finding chessboard corners
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
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img.shape[:2], None, None)

print(ret)
print(mtx)
print(dist)
print(rvecs)