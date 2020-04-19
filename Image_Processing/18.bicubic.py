import cv2
import numpy as np
img1 = cv2.imread("images/2.jpg")
rows1 = img1.shape[0]
cols1 = img1.shape[1]
img = np.zeros((rows1, cols1, 1), np.uint8)
for i in range(rows1):
    for j in range(cols1):
        img[i,j] = img1[i,j,0]
cv2.imshow("Image",img)

img2 = np.zeros((rows1*2,cols1*2,1),np.uint8)
rows2 = img2.shape[0]
cols2 = img2.shape[1]
for x in range(rows2):
    for y in range(cols2):
        sum1=0
        secret_sum=0
        (p1,q1) = (int(x/2), int(y/2))
        (u,v) = (float(x/2),(y/2))
        for i in range(-1,3,1):
            for j in range(-1,3,1):
                (p2,q2) = (abs(p1+i), abs(q1+j))
                (xd1,yd1) = (abs(p2-u),abs(q2-v))
                if xd1 < 1:
                    w1 = (3/2*(xd1)*(3)) - ((5/2)*(xd1)*(2))+ 1
                elif xd1 < 2:
                    w1 = (-(1/2)*(xd1)*(3))+((5/2)*(xd1)*2)-(4*(xd1))+2
                else:
                    w1 = 0
                if yd1 < 1:
                    w2 = (3/2*(yd1)*(3)) - ((5/2)*(yd1)*(2))+ 1
                elif yd1 < 2:
                    w2 = (-(1/2)*(yd1)*(3))+((5/2)*(yd1)*2)-(4*(yd1))+2
                else:
                    w2 = 0
                sum1 = sum1 + w1 * w2 * img[p1,q1,0]
                secret_sum = secret_sum + (w1)*(w2)
        img2[x,y,0] = int(sum1 / secret_sum)
cv2.imshow("Result",img2)
cv2.waitKey(0)
