import cv2
import numpy as np
img = cv2.imread("images/cameraman.jpg")
cv2.imshow("original picture", img)
rows=img.shape[0]
cols=img.shape[1]
img2 = np.zeros((rows,cols,3),np.uint8)
for i in range(rows):
    for j in range(cols):
        (b,g,r)=img[i,j]
        if b > 50 or g >50 or r>50:
            img2[i,j] =100
        elif b <=50 or g <=50 or b <=50:
            img2[i,j]= 200
cv2.imshow("threshhold picture", img2)
cv2.waitKey(0)