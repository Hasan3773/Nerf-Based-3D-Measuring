import cv2
import numpy as np

# default max/min values
RED_MAX = np.array([180, 255, 255])
RED_MIN = np.array([106, 141, 63])

print(RED_MAX)
print(RED_MIN)

# trackbar
def nothing(x):
    pass

cv2.namedWindow('filtered')

# trackbar for max HSV threshold
cv2.createTrackbar('H1', 'filtered', 0, 180, nothing)
cv2.createTrackbar('S1', 'filtered', 0, 255, nothing)
cv2.createTrackbar('V1', 'filtered', 0, 255, nothing)

# trackbar for min HSV threshold
cv2.createTrackbar('H2', 'filtered', 0, 180, nothing)
cv2.createTrackbar('S2', 'filtered', 0, 255, nothing)
cv2.createTrackbar('V2', 'filtered', 0, 255, nothing)

use_camera = True

if use_camera:        
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
else: 
    frame = cv2.imread("datasets/WIN_20240318_17_10_33_Pro.jpg")

while(1):
    print('t')
    if use_camera:
        ret, frame = cap.read()
        print('t')
    
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # get current positions of four trackbars
    h1 = cv2.getTrackbarPos('H1','filtered')
    s1 = cv2.getTrackbarPos('S1','filtered')
    v1 = cv2.getTrackbarPos('V1','filtered')

    h2 = cv2.getTrackbarPos('H2','filtered')
    s2 = cv2.getTrackbarPos('S2','filtered')
    v2 = cv2.getTrackbarPos('V2','filtered')

    # update min and max values
    RED_MAX = np.array([h1, s1, v1])
    RED_MIN = np.array([h2, s2, v2])

    # recreate the mask
    filtered_mask = cv2.inRange(image, RED_MIN, RED_MAX)
    masked_rgb = cv2.bitwise_and(frame, frame, mask=filtered_mask)
    masked_rgb = np.concatenate([masked_rgb, filtered_mask[..., None]], axis=2)
    cv2.imshow('filtered', masked_rgb)
    cv2.waitKey(1)