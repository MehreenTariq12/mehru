import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("images/lena.png")
#img=img6[100:150,100:150]
cv2.imshow("original image", img)
rows=img.shape[0]
cols=img.shape[1]
img2 = np.zeros((rows,cols,3),np.uint8)
img3 = np.zeros((rows,cols,3),np.uint8)
img4 = np.zeros((rows,cols,3),np.uint8)
img5 = np.zeros((rows,cols,3),np.uint8)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img2[i,j] = 16*img[i,j]**(1/2)
        img3[i,j] = 255*(img[i,j]/255)**(2)
        img4[i, j] = (np.log(img[i, j]+1))*46
        img5[i,j] = (np.exp(img[i,j]/46))

cv2.imshow("nth root", img2)
cv2.imshow("nth power", img3)
cv2.imshow("log", img4)
cv2.imshow("anti-log", img5)
cv2.waitKey(0)

