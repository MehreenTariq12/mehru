import cv2
import numpy as np
image = cv2.imread("images/morphology.png")
img=image
cv2.imshow("Original",img)
rows = img.shape[0]
cols = img.shape[1]
img1 = np.zeros((rows,cols,1),np.uint8)
img2 = np.zeros((rows,cols,1),np.uint8)
SE = [[1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1]]
m1=[]
for i in range(2,rows-2):
    for j in range(2,cols-2):
        pix = img[i,j,0]
        m1 = []
        for m in range(-2,3,1):
            for n in range(-2, 3, 1):
                m1.append(img[i + m, j + n, 0]*SE[2+m][2+n])
        img1[i,j,0] = min(m1)
        img2[i, j, 0] = max(m1)

cv2.imshow("Erosion",img1)
cv2.imshow("Dilution",img2)
cv2.waitKey(0)