import numpy as np
import cv2 as cv
import glob
 
# Calibrated camera matrix (3x3)
camera_matrix = np.array([[1.12132240e+03, 0.00000000e+00, 9.37579395e+02],
                          [0.00000000e+00, 1.11927460e+03, 5.40821641e+02],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

# Calibrated distortion coefficients (1x5)
dist_coeffs = np.array([[ 0.13312033, -0.49735709, -0.00141145,  0.00096862,  0.59645158]])

# initiliase json (create dick)
# Read images 
for fname in glob.glob('chessboards/tag*.jpg'):
   img = cv.imread(fname)
   gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

   # Aruco Tag Parameters
   dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
   parameters =  cv.aruco.DetectorParameters()

   # Create aruco tag detector
   detector = cv.aruco.ArucoDetector(dictionary, parameters)

   # Detect markers
   corners, ids, rej = detector.detectMarkers(gray)

   # If detected
   if len(ids) > 0:
        # Detect aruco pose
        rvec, tvec ,_ = cv.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)
        
        # Get first rvec and tvec (hopefully there's only one)
        rvec = rvec[0]
        tvec = tvec[0] # add entry in the dick for image and transform
        # append to the frames array in the dick (json)
   
        r_matrix = cv.Rodrigues(rvec)[0]
        t_matrix = np.eye(4,4)
        t_matrix[:3,:3] = r_matrix
        t_matrix[:3,3] = tvec
        print(t_matrix)

        # Draw axis for the aruco markers
        cv.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

   cv.imshow('img',img)
   k = cv.waitKey(0)

cv.destroyAllWindows()

# ------------ Generate Transformation Json ---------------------------

# so i need to take the transformation matrix from the image after i take the pose 






