import cv2
import numpy as np
corner = cv2.imread("images/cameraman2.jpg")
print(corner.shape[0])
print(corner.shape[1])
pic2 = np.zeros((int(1400/3), int(1400/3), 3), np.uint8)
pic1 = corner[0:1400,0:1400]
cv2.imshow("image1", pic1)
m = int(1400/3)
for i in range(m-2):
    for j in range(m-2):
        row = 1 + (3 * i)
        col = 1 + (3 * j)
        pic2[i,j] = pic1[row,col]
cv2.imshow("image2", pic2)
cv2.imwrite("images/zoom.jpg", pic2)
cv2.waitKey(0)