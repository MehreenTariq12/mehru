import cv2
import numpy as np
image = cv2.imread("images/lena.png")
img=image
cv2.imshow("Original",img)
rows = img.shape[0]
cols = img.shape[1]
img1 = np.zeros((rows,cols,3),np.uint8)
new = np.zeros((rows,cols,3),np.uint8)
w = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]

for i in range(1,rows-1):
    for j in range(1,cols-1):
        bsum = gsum = rsum = 0
        for x in range(-1,2,1):
            for y in range(-1,2,1):
                (b,g,r) = img[i+x, j+y]
                bsum = bsum + (b * w[1+x][1+y])
                gsum = gsum + (g * w[1 + x][1 + y])
                rsum = rsum + (r * w[1 + x][1 + y])
        if bsum > 255:
            bsum=255
        elif bsum < 0:
            bsum = 0
        if gsum > 255:
            gsum=255
        elif gsum < 0:
            gsum = 0
        if rsum > 255:
            rsum=255
        elif rsum < 0:
            rsum = 0
        img1[i, j,0] = bsum
        img1[i, j, 1] = gsum
        img1[i, j, 2] = rsum
cv2.imshow("Convoluted image",img1)
for i in range(rows):
    for j in range(cols):
        new[i,j] = cv2.add(np.uint8([img[i,j]]), np.uint8([img1[i,j]]))

cv2.imshow("result",new)
cv2.waitKey(0)
