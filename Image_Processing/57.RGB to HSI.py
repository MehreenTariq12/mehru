import cv2
import numpy as np
import math
image = cv2.imread("images/vege.png")
cv2.imshow("Original",image)
rows = image.shape[0]
cols = image.shape[1]
HSI = np.zeros((rows,cols,3),np.float)
for i in range(rows):
    for j in range(cols):
        (b1,g1,r1) = image[i,j]
        b = b1/255
        g = g1/255
        r = r1/255

        I = (r+g+b)/3

        if b < g and b < r:
            min = b
        elif g < b and g < r:
            min = g
        else:
            min = r
        num5 = (min/(r+g+b+0.00000001))*3
        S = 1 - num5





        num = 0.5 *((r-g)+(r-b))/math.sqrt((r-g)**2+((r-b)*(g-b))+0.00000001)
        theta = math.acos(num)

        if b <= g:
            H = theta
        elif b > g:
            H = 360 - theta
        HSI[i, j] = (I, S, H)

cv2.imshow("HSI",HSI)
#cv2.imshow('H Channel', HSI[:, :, 0])
#cv2.imshow('S Channel', HSI[:, :, 1])
#cv2.imshow('I Channel', HSI[:, :, 2])
cv2.imwrite("images/HSI.jpeg",HSI)
cv2.waitKey(0)

