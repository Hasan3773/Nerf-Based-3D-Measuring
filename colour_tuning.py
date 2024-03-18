import cv2
import numpy as np

# default max/min values
RED_MAX = np.array([131, 255, 255])
RED_MIN = np.array([109, 105, 103])

print(RED_MAX)
print(RED_MIN)

src = cv2.imread("dataset6/images/WIN_20240318_13_42_27_Pro (3).jpg")

image = cv2.cvtColor(src, cv2.COLOR_RGB2HSV)
filtered_mask = cv2.inRange(image, RED_MIN, RED_MAX)

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

while(1):
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

    cv2.imshow('filtered', filtered_mask)
    cv2.waitKey(1)