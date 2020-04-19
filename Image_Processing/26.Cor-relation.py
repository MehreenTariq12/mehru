import cv2
import numpy as np
img = cv2.imread("images/green.jpg")
cv2.imshow("Original",img)
rows = img.shape[0]
cols = img.shape[1]
img1 = np.zeros((rows,cols,3),np.uint8)
img2 = np.zeros((rows,cols,3),np.uint8)
img3 = np.zeros((rows,cols,3),np.uint8)
img4 = np.zeros((rows,cols,3),np.uint8)
img5 = np.zeros((rows,cols,3),np.uint8)
img6 = np.zeros((rows,cols,3),np.uint8)
img7 = np.zeros((rows,cols,3),np.uint8)
img8 = np.zeros((rows,cols,3),np.uint8)
img9 = np.zeros((rows,cols,3),np.uint8)
w = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
for i in range(1,rows-1):
    for j in range(1,cols-1):
        bsum = gsum = rsum = 0
        for x in range(-1,2,1):
            for y in range(-1,2,1):
                (b, g, r) = img[i+x, j+y]
                bsum = bsum + (b * w[1+x][1+y])
                gsum = gsum + (g * w[1 + x][1 + y])
                rsum = rsum + (r * w[1 + x][1 + y])
        bsum = bsum / 9
        gsum = gsum / 9
        rsum = rsum / 9
        img1[i, j] = (bsum, gsum, rsum)
        img2[i, j] = (bsum, bsum, rsum)
        img3[i, j] = (gsum, gsum, rsum)
        img4[i, j] = (bsum, rsum, rsum)
        img5[i, j] = (rsum, gsum, rsum)
        img6[i, j] = (bsum, gsum, bsum)
        img7[i, j] = (bsum, gsum, gsum)
        img8[i, j] = (rsum, gsum, bsum)
cv2.imshow("Cor-related image",img1)
cv2.imshow("bbr",img2)
cv2.imshow("ggr",img3)
cv2.imshow("brr",img4)
cv2.imshow("rgr",img5)
cv2.imshow("bgb",img6)
cv2.imshow("bgg",img7)
cv2.imshow("rgb",img8)
cv2.imwrite("images/Corelated box.jpg",img1)
cv2.waitKey(0)
