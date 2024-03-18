import numpy as np
import cv2 as cv
import glob
import json
import os
from pathlib import Path
from pose_estimation import estimate_pose
import sys

use_video = True

# Calibrated camera matrix (3x3)
camera_matrix = np.array([[1.12132240e+03, 0.00000000e+00, 9.37579395e+02],
                          [0.00000000e+00, 1.11927460e+03, 5.40821641e+02],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

# Calibrated distortion coefficients (1x5)
dist_coeffs = np.array([[ 0.13312033, -0.49735709, -0.00141145,  0.00096862,  0.59645158]])

RED_MAX = np.array([131, 255, 255])
RED_MIN = np.array([109, 105, 103])

# Aruco Tag Parameters
dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
parameters =  cv.aruco.DetectorParameters()

# Create aruco tag detector
detector = cv.aruco.ArucoDetector(dictionary, parameters)

# initiliase json (create dick)
transforms = {}
transforms["fl_x"] = camera_matrix[0, 0]
transforms["fl_y"] = camera_matrix[1, 1]
transforms["cx"] = camera_matrix[0, 2]
transforms["cy"] = camera_matrix[1, 2]
transforms["aabb_scale"] = 1
transforms["scale"] = 15
transforms["frames"] = []


def process_image(image, fname):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    masked = cv.inRange(cv.cvtColor(image, cv.COLOR_RGB2HSV), RED_MIN, RED_MAX)
    masked_path = dataset_path / "images" / f"masked_{Path(fname).stem}.png"
    masked_rgb =  cv.bitwise_and(image, image, mask=masked)
    masked_rgb = np.concatenate([masked_rgb, masked[..., None]], axis=2)
    # masked_rgb = cv.flip(masked_rgb, 0) 
    # masked_rgb = cv.flip(masked_rgb, 1) 
    cv.imwrite(str(masked_path), masked_rgb)

    # Dynamically update W and H of images (ensure all images are same dimensions)
    transforms["w"] = image.shape[1]
    transforms["h"] = image.shape[0]

    # Detect markers
    corners, ids, rej = detector.detectMarkers(gray)
    
    t_matrix, labeled, success = estimate_pose(image, gray, detector, camera_matrix, dist_coeffs)
    print(success)
    if success:
        frame = {}
        frame["file_path"] = str(Path("images") / f"masked_{Path(fname).stem}.png")
        frame["sharpness"] = 100
        frame["transform_matrix"] = [list(t) for t in t_matrix]
        transforms["frames"].append(frame)
    return labeled

# Dataset paths
dataset_path = Path('datasets/c0_0')
images_glob = 'WIN_2024031*.jpg'

if not use_video:
    for fname in glob.glob(str(dataset_path / "images" / images_glob)):
        image = cv.imread(fname)
        labeled = process_image(image, fname)
        cv.imshow("labeled", labeled)
        cv.waitKey(0)
else:
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    i = 0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        fname = f"image{i}.png"
        labeled = process_image(frame, fname)
        cv.imshow("labeled", labeled)
    
        if cv.waitKey(1) == ord('q'):
          break

with open(str(dataset_path/"transforms.json"), "w") as outfile: 
    json.dump(transforms, outfile, indent=2)

cv.destroyAllWindows()
print(f"/mnt/c/Users/owenl/Documents/Instant-NGP-for-RTX-3000-and-4000/Instant-NGP-for-RTX-3000-and-4000/instant-ngp.exe /home/owen/Nerf-Based-3D-Measuring/{str(dataset_path)}")
os.system(f"/mnt/c/Users/owenl/Documents/Instant-NGP-for-RTX-3000-and-4000/Instant-NGP-for-RTX-3000-and-4000/instant-ngp.exe {str(dataset_path)}")  

