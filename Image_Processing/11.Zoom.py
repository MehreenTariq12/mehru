import cv2
import numpy as np
pic1 = cv2.imread("images/1.jpg")
pic2 = np.zeros((675, 675, 3), np.uint8)
cv2.imshow("image1", pic1)
for i in range(225):
    for j in range(225):
        row = 1 + (3 * i)
        col = 1 + (3 * j)
        pic2[row, col] = pic1[i, j]
        pic2[row-1, col-1] = pic1[i, j]
        pic2[row-1, col] = pic1[i, j]
        pic2[row-1, col+1] = pic1[i, j]
        pic2[row, col-1] = pic1[i, j]
        pic2[row, col+1] = pic1[i, j]
        pic2[row+1, col-1] = pic1[i, j]
        pic2[row+1, col] = pic1[i, j]
        pic2[row+1, col+1] = pic1[i, j]
cv2.imshow("image2", pic2)
cv2.imwrite("images/zoom.jpg", pic2)
cv2.waitKey(0)