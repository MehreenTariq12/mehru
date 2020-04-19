import cv2
import numpy as np
image = cv2.imread("images/hut.jpg")
rows = image.shape[0]
cols = image.shape[1]
CMY = np.zeros((rows,cols,3),np.float)
CMY1 = np.zeros((rows,cols,3),np.float)
CMY2 = np.zeros((rows,cols,3),np.float)
CMY3 = np.zeros((rows,cols,3),np.float)
for i in range(rows):
    for j in range(cols):
        (b1,g1,r1) = image[i,j]
        b = b1/255
        g = g1/255
        r = r1/255
        (Y,M,C) = (1-b,1-g,1-r)
        if C<M and C<Y:
            K = C
        elif M<C and M<Y:
            K = M
        else:
            K = Y
        if K == 1:
            C = 0
            M = 0
            Y = 0
        else:
            C = (C-K)/(1-K)
            M = (M-K)/(1-K)
            Y = (Y - K) / (1 - K)


        CMY[i,j] = (C,M,Y)
        CMY1[i, j] = (C, 1, 1)
        CMY2[i, j] = (1, M, 1)
        CMY3[i, j] = (1, 1, Y)

cv2.imshow("BGR",image)
cv2.imshow("CMYK",CMY)
cv2.imshow("Yellow",CMY1)
cv2.imshow("Magentta",CMY2)
cv2.imshow("Cyan",CMY3)
cv2.waitKey(0)