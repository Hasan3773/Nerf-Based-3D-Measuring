import cv2 
import numpy as np
import json

dataset = "datasets/rivian-R3X"
# To read image 
img = cv2.imread(f"{dataset}.density_slices_256x256x256.png", cv2.IMREAD_COLOR) 

# convert to gray scale to turn into a 2D matrix of densities corelating to grayness 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
# ret, thresh = cv2.threshold(gray, 127, 255, 0) 

# Defines gray threshold, so if less than 200 do not include 
threshold = 200

# List that stores all the slices 
reshaped = []

# iterate through every set of 256 pixels in both i & j
for i in range(0, 16*256,256):
    for j in range(0, 16*256,256):
        # seperate giant image into 256x256 sections
        slice = gray[i:i+256, j:j+256]
        # add each section to the reshaped list 
        reshaped.append(slice)

# stack all the sliced into a 3D matrix
reshaped = np.stack(reshaped)

# define a top down map to iterate and sum through 
top_down_map = np.zeros((256, 256))
for i in range(0, reshaped.shape[1]):
    top_down_map = np.maximum(top_down_map, reshaped[:, i, :]) # finds the largest area covered in all slices 

# apply threshold to 2D matrix 
top_down_map[top_down_map < threshold] = 0
top_down_map = top_down_map.astype(np.uint8) # convert to a 0-255 number for opencv requirements 

# creates a contour map from the top down map - creates outline 
ctrs = cv2.findContours(top_down_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

width = 0
height = 0
max_i = 0
# Loop through all the contours and find the largest one to elimenate any artifacts 
for i, ctr in enumerate(ctrs):
    rect = cv2.minAreaRect(ctr) # finds the smallest possible bounds to close the contour in 

    (x, y), (w, h), angle = rect # retrives variable from tuple 
    if(w > width):
        max_i = i
    # assigns width & height to the max value 
    width = max(width, w)
    height = max(height, h)

with open(f'{dataset}/transforms.json') as f:
    scale = json.load(f)['scale'] # divide by scale factor to get real dimensions 

# converts from nerf units to mm's
width = width / scale
height = height / scale
calibrated_width = 5.23*width+0.553
calibrated_height = 5.23*height+0.553


print("Raw Dimension 1:", width, "Raw Dimension 2:", height)
print("Calibrated Dimension 1:", calibrated_width, "Calibrated Dimension 2:", calibrated_height)

# drew the rectange and display it on the screen
map_rgb = cv2.cvtColor(top_down_map,cv2.COLOR_GRAY2RGB)

rect = cv2.minAreaRect(ctrs[max_i]) # draw the biggest contour
box = cv2.boxPoints(rect)
box = np.intp(box)
cv2.drawContours(map_rgb,[box],0,(0,0,255),2)

cv2.imshow("Top Down Map", map_rgb)
cv2.waitKey(0)
