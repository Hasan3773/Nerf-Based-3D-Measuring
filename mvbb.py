import cv2 
import numpy as np
import numpy as np
import json

dataset = "datasets/lego_thin_test"
# To read image 
img = cv2.imread(f"{dataset}.density_slices_256x256x256.png", cv2.IMREAD_COLOR) 

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
# ret, thresh = cv2.threshold(gray, 127, 255, 0) 

cv2.imwrite('opncv_sample.png', gray)

threshold = 200

reshaped = []

for i in range(0, 16*256,256):
    for j in range(0, 16*256,256):
        slice = gray[i:i+256, j:j+256]
        reshaped.append(slice)

reshaped = np.stack(reshaped)

top_down_map = np.zeros((256, 256))
for i in range(0, reshaped.shape[1]):
    top_down_map = np.maximum(top_down_map, reshaped[:, i, :])

top_down_map[top_down_map < threshold] = 0
top_down_map = top_down_map.astype(np.uint8)

ctrs = cv2.findContours(top_down_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

width = 0
height = 0
max_i = 0
for i, ctr in enumerate(ctrs):
    rect = cv2.minAreaRect(ctr)

    (x, y), (w, h), angle = rect
    if(w > width):
        max_i = i
    width = max(width, w)
    height = max(height, h)

with open(f'{dataset}/transforms.json') as f:
    scale = json.load(f)['scale']

width = width / scale
height = height / scale
calibrated_width = 5.23*width+0.553
calibrated_height = 5.23*height+0.553


print("Raw Dimension 1:", width, "Raw Dimension 2:", height)
print("Calibrated Dimension 1:", calibrated_width, "Calibrated Dimension 2:", calibrated_height)

map_rgb = cv2.cvtColor(top_down_map,cv2.COLOR_GRAY2RGB)

rect = cv2.minAreaRect(ctrs[max_i])
box = cv2.boxPoints(rect)
box = np.intp(box)
cv2.drawContours(map_rgb,[box],0,(0,0,255),2)

cv2.imshow("Top Down Map", map_rgb)
cv2.waitKey(0)
