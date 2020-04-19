import cv2
import numpy as np
img = cv2.imread("images/1.jpg")
cv2.imshow("Image",img)
rows1 = img.shape[0]
cols1 = img.shape[1]
img2 = np.zeros((rows1*2,cols1*2,3),np.uint8)
rows2 = rows1 * 2
cols2 = cols1 * 2
rd = rows2 / rows1
cd = cols2 / cols1
for x in range(rows2):
    for y in range(cols2):
        a = x / rd
        b = y / cd
        img2[x, y] = img[int(a), int(b)]
cv2.imshow("Result", img2)
cv2.waitKey(0)


