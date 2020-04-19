import cv2
import numpy as np
img = cv2.imread("images/green.jpg")
cv2.imshow("Original",img)
rows = img.shape[0]
cols = img.shape[1]
img1 = np.zeros((rows,cols,3),np.uint8)
img2 = np.zeros((rows,cols,3),np.uint8)
img3 = np.zeros((rows,cols,3),np.uint8)
for i in range(2,rows-2):
    for j in range(2,cols-2):
        bsum = gsum = rsum = 0
        for x in range(i-2, i+2,1):
            for y in range(j-2, j+2,1):
                (b,g,r) = img[x,y]
                bsum = bsum + b
                gsum = gsum + g
                rsum = rsum + r
        bsum = bsum / 25
        gsum = gsum / 25
        rsum = rsum / 25
        img2[i,j] = (bsum,gsum,rsum)
for i in range(3,rows-3):
    for j in range(3,cols-3):
        bsum = gsum = rsum = 0
        for x in range(i-3, i+3,1):
            for y in range(j-3, j+3,1):
                (b,g,r) = img[x,y]
                bsum = bsum + b
                gsum = gsum + g
                rsum = rsum + r
        bsum = bsum / 49
        gsum = gsum / 49
        rsum = rsum / 49
        img3[i,j] = (bsum,gsum,rsum)
for i in range(1,rows-1):
    for j in range(1,cols-1):
        bsum = gsum = rsum = 0
        for x in range(i-1, i+1,1):
            for y in range(j-1, j+1,1):
                (b,g,r) = img[x,y]
                bsum = bsum + b
                gsum = gsum + g
                rsum = rsum + r
        bsum = bsum / 9
        gsum = gsum / 9
        rsum = rsum / 9
        img1[i,j] = (bsum,gsum,rsum)
cv2.imshow("3*3 neighbourhood",img1)
cv2.imshow("5*5 neighbourhood",img2)
cv2.imshow("7*7 neighbourhood",img3)
cv2.waitKey(0)
