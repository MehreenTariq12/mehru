import cv2
import numpy as np
image = cv2.imread("images/hut.jpg")
rows = image.shape[0]
cols = image.shape[1]
CMY = np.zeros((rows,cols,3),np.float)
CMY1 = np.zeros((rows,cols,3),np.float)
CMY2 = np.zeros((rows,cols,3),np.float)
CMY3 = np.zeros((rows,cols,3),np.float)
RGB = np.zeros((rows,cols,3),np.float)
RGB1 = np.zeros((rows,cols,3),np.float)
RGB2 = np.zeros((rows,cols,3),np.float)
RGB3 = np.zeros((rows,cols,3),np.float)
for i in range(rows):
    for j in range(cols):
        (b1,g1,r1) = image[i,j]
        b = b1/255
        g = g1/255
        r = r1/255
        (b2,g2,r2) = (1-b,1-g,1-r)
        CMY[i,j] = (r2,g2,b2)
        CMY1[i, j] = (r2,1,1)
        CMY2[i, j] = (1, g2,1)
        CMY3[i, j] = (1,1,b2)
        RGB1[i, j] = (1-r2, 0, 0)
        RGB2[i, j] = (0, 1-g2, 0)
        RGB3[i, j] = (0, 0, 1-b2)
        RGB[i, j] = (1-b2, 1-g2, 1 - r2)
cv2.imshow("Original",image)
cv2.imshow("CMY",CMY)
cv2.imshow("Yellow",CMY1)
cv2.imshow("Megentta",CMY2)
cv2.imshow("Cyan",CMY3)
cv2.imshow("Blue=1-yellow",RGB1)
cv2.imshow("Green=1-Magenta",RGB2)
cv2.imshow("Red=1-Cyan",RGB3)
cv2.imshow("BGR",RGB)
cv2.waitKey(0)