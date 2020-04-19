import cv2
import numpy as np
img = cv2.imread("images/cameraman.jpg")
rows=img.shape[0]
cols=img.shape[1]
cv2.imshow("picture", img)
img2 = np.zeros((rows,cols,3),np.uint8)
for i in range(rows):
    for j in range(cols):
        (b,g,r)=img[i,j]
        if b >= 80 or g >= 80 or r >= 80:
            img2[i,j] = img[i,j] + 20
        elif b <=80 or g <=80 or b <=80:
            img2[i,j]= img[i,j] - 20
cv2.imshow("threshhold picture", img2)
cv2.waitKey(0)