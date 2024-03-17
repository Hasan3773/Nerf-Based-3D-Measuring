import numpy as np
import cv2 as cv
import glob
 
# Dummy camera matrix (3x3)
camera_matrix = np.array([[7.93414938e+02, 0.00000000e+00, 5.26000909e+02],
 [0.00000000e+00, 3.62409090e+03, 9.97845449e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

# Dummy distortion coefficients (1x5)
dist_coeffs = np.array([0.1, -0.2, 0.05, 0, 0])

# Convert the matrix to floating point
mtx = camera_matrix.astype(np.float32)
dist = dist_coeffs.astype(np.float32)

"""
# Save the arrays to a NPZ file 
np.savez('camera_data.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

# Load previously saved data
with np.load('camera_data.npz') as X:
 mtx, dist = [X[i] for i in ('camera_matrix','dist_coeffs')]
"""

# Initialize some variables - mabye come back and understand them more
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((10*7,3), np.float32)
objp[:,:2] = np.mgrid[0:10,0:7].T.reshape(-1,2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

# ------------------------------------

# Function to draw lines onto the img 
def draw(img, corners, imgpts):
 corner = np.array(tuple(corners[0].ravel())).astype(np.int32)

 img = cv.line(img, corner, np.array(tuple(imgpts[0].ravel())).astype(np.int32), (255,0,0), 5)
 img = cv.line(img, corner, np.array(tuple(imgpts[1].ravel())).astype(np.int32), (0,255,0), 5)
 img = cv.line(img, corner, np.array(tuple(imgpts[2].ravel())).astype(np.int32), (0,0,255), 5)
 return img

# Draws the axis onto the image 
for fname in glob.glob('chessboards\chessboard1.jpg'):
 print(fname)
 img = cv.imread(fname)
 gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
 ret, corners = cv.findChessboardCorners(gray, (10,7),None)
 
 if ret == True:
    corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
 
 # Find the rotation and translation vectors.
 ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
 
 # project 3D points to image plane
 imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
 
 img = draw(img,corners2,imgpts)
 cv.imshow('img',img)
 k = cv.waitKey(0) & 0xFF
 if k == ord('s'):
    cv.imwrite(fname[:6]+'.png', img)
 
cv.destroyAllWindows()



