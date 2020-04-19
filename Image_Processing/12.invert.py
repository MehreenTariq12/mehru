import cv2
import numpy as np
pic1 = cv2.imread("images/1.jpg")
pic2 = np.zeros((225, 225, 3), np.uint8)
pic3 = np.zeros((225, 225, 3), np.uint8)
pic4 = np.zeros((225, 225, 3), np.uint8)
pic5 = np.zeros((225, 225, 3), np.uint8)
pic6 = np.zeros((225, 225, 3), np.uint8)
cv2.imshow("image1", pic1)
for i in range(225):
    for j in range(225):
        pic2[i, 224-j] = pic1[i, j]
        pic3[j, i] = pic1[i, j]
        pic4[j, 224-i] = pic1[i, j]
        pic5[224 - i, 224 - j] = pic1[i, j]
        pic6[224 - i, j] = pic1[i, j]
cv2.imshow("Result1", pic2)
cv2.imshow("Result2", pic3)
cv2.imshow("Result3", pic4)
cv2.imshow("Result4", pic5)
cv2.imshow("Result5", pic6)
cv2.waitKey(0)