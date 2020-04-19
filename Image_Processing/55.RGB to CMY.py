import cv2
import numpy as np
image = cv2.imread("images/hut.jpg")
rows = image.shape[0]
cols = image.shape[1]
CMY = np.zeros((rows,cols,3),np.uint8)
CMY1 = np.zeros((rows,cols,3),np.uint8)
CMY2 = np.zeros((rows,cols,3),np.uint8)
CMY3 = np.zeros((rows,cols,3),np.uint8)
for i in range(rows):
    for j in range(cols):
        (b1,g1,r1) = image[i,j]
        b = b1/255
        g = g1/255
        r = r1/255
        (b2,g2,r2) = (1-b,1-g,1-r)
        CMY[i,j] = (r2*255,g2*255,b2*255)
        CMY1[i, j] = (r2 *255,255,  255)
        CMY2[i, j] = (255, g2 * 255,255)
        CMY3[i, j] = (255,255,b2 * 255)
cv2.imshow("BGR",image)
cv2.imshow("CMY",CMY)
cv2.imshow("Yellow",CMY1)
cv2.imshow("Megentta",CMY2)
cv2.imshow("Cyan",CMY3)
cv2.waitKey(0)