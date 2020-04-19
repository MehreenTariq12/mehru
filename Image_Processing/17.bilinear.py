import cv2
import numpy as np
img = cv2.imread("images/2.jpg")
cv2.imshow("Image",img)
rows1 = img.shape[0]
cols1 = img.shape[1]
img2 = np.zeros((rows1*2,cols1*2,1),np.uint8)
rows2 = rows1 * 2
cols2 = cols1 * 2
rd = rows2 / rows1
cd = cols2 / cols1
for x in range(rows2-4):
    for y in range(cols2-4):

        p=int(x/2)
        q=int(y/2)
        a = (x / rd)-p
        b = (y / cd)-q

        img2[x,y,0] = img[p,q,0]*(1-a)*(1-b)+(b)*(1-a)*img[p,q+1,0] + (1-b)*(a)*img[p+1,q,0] + (b)*(a)*img[p+1,q+1,0]

cv2.imshow("Result", img2)
cv2.waitKey(0)


