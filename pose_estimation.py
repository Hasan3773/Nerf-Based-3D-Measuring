import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation as R

def estimate_pose(img, gray, detector, camera_matrix, dist_coeffs):
   # Detect markers
   corners, ids, _ = detector.detectMarkers(gray)

   labeled = img.copy()
   t_matrix = np.eye(4,4)
   success = ids is not None and (len(ids.flatten()) > 1)
   print(success)
   # If detected
   if success:
      # Detect aruco pose
      rvec, tvec ,_ = cv.aruco.estimatePoseSingleMarkers(corners, 0.053, camera_matrix, dist_coeffs)
      
      # Get first rvec and tvec (hopefully there's only one)
      rvec = rvec[0]
      tvec1 = tvec[0] # add entry in the dict for image and transform
      tvec2 = tvec[1] # add entry in the dict for image and transform
      tvec = (tvec1 + tvec2)/2.0
      # append to the frames array in the dict
      r_matrix = cv.Rodrigues(rvec)[0]

      t_matrix[:3,:3] = r_matrix
      t_matrix[:3,3] = tvec
      t_matrix = np.linalg.inv(t_matrix)

      r1 = R.from_euler('y', 180, degrees=True).as_matrix()
      r2 = R.from_euler('z', 180, degrees=True).as_matrix()
      t_matrix[:3,:3] = t_matrix[:3,:3] @ r1 @ r2
      # t_matrix[1,3] = -t_matrix[1,3]
      
      # Draw axis for the aruco markers
      cv.drawFrameAxes(labeled, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

   return t_matrix, labeled, success
      







