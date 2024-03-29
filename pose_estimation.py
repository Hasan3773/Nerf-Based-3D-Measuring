import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation as R

def estimate_pose(img, gray, detector, camera_matrix, dist_coeffs):
   # Detect markers
   corners, ids, _ = detector.detectMarkers(gray) # define & finds vars for corners and id's of the 2 tags

   labeled = img.copy() # creates copy of img
   t_matrix = np.eye(4,4) # creates identity matrix 
   success = ids is not None and (len(ids.flatten()) > 1) # makes sure there are 2 id's 
   print(success) 
   # If detected
   if success:
      # Detect aruco pose
      rvec, tvec ,_ = cv.aruco.estimatePoseSingleMarkers(corners, 0.053/2, camera_matrix, dist_coeffs) # 0.053 is ratio to prev aruco tag
      
      # Get first rvec and tvec (hopefully there's only one)
      rvec = rvec[0]
      tvec1 = tvec[0] # add entry in the dict for image and transform
      tvec2 = tvec[1] # add entry in the dict for image and transform
      tvec = (tvec1 + tvec2)/2.0 # finds the midpoint of the aruco tags to use as origin 
      # append to the frames array in the dict
      r_matrix = cv.Rodrigues(rvec)[0]

      t_matrix[:3,:3] = r_matrix
      t_matrix[:3,3] = tvec
      t_matrix = np.linalg.inv(t_matrix) # invert transforms from camera to world

      # rotate camera to fix flipped camera issue 
      r1 = R.from_euler('y', 180, degrees=True).as_matrix()
      r2 = R.from_euler('z', 180, degrees=True).as_matrix()
      t_matrix[:3,:3] = t_matrix[:3,:3] @ r1 @ r2 # multiple t matrix by the two rotations to apply them 
      # t_matrix[1,3] = -t_matrix[1,3]
      
      # Draw axis for the aruco markers
      cv.drawFrameAxes(labeled, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

   return t_matrix, labeled, success
      







