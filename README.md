# Nerf-Based-3D-Measurement-System

## Step 1: Calibration & Pose Estimation 
- Calculated the camera intrinsic matrix & distortion coefficient
- Used arocu tags known dimensions as a reference 
- Detect corners of aruco tags in view using OpenCV
- Triangulate the corners of aruco tags to get 3D pose
- Triangulates origin of object based on the center of the two tags 
- Returns tvec & rvec

## Step 2: Color Filtering 
- Converts live streamed image to HSV (Hue, Saturation, Value) color space
- Define min & max HSV threshold, tuned to the color chosen & environment
- Creates alpha mask to eliminate colors outside of range

## Step 3: Integration with Instant NGP
- Extract camera intrinsics from calibration
- Extract tvec & rvec from pose estimation
- Created json including the camera matrix & image transforms 
- Fed json into Nvidia’s Instant NGP model
- Outputs flattened 3D matrix of densities

## Step 4: Volumetric Bounding Box
- Reconstructs 3D matrix from spliced version 
- Fits a 2D bounding box to the summed topological splice 
- Outputs length & width 


