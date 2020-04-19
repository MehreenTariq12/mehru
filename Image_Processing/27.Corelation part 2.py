import cv2
import numpy as np
img = cv2.imread("images/green.jpg")
cv2.imshow("Original",img)
rows = img.shape[0]
cols = img.shape[1]
img1 = np.zeros((rows,cols,3),np.uint8)
w = [[0.3679, 0.6065, 0.3679], [0.6065, 1.0000, 0.6065], [0.3679, 0.6065, 0.3679]]
for i in range(1,rows-1):
    for j in range(1,cols-1):
        bsum = gsum = rsum = 0
        for x in range(-1,2,1):
            for y in range(-1,2,1):
                (b, g, r) = img[i+x, j+y]
                bsum = bsum + (b * w[1+x][1+y])
                gsum = gsum + (g * w[1 + x][1 + y])
                rsum = rsum + (r * w[1 + x][1 + y])
        bsum = bsum / 4.8976
        gsum = gsum / 4.8976
        rsum = rsum / 4.8976
        img1[i, j] = (bsum, gsum, rsum)
cv2.imshow("Cor-related image",img1)
cv2.imwrite("images/Corelated guassian.jpg",img1)
cv2.waitKey(0)
