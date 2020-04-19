import cv2
import numpy as np
img1 = cv2.imread("images/green.jpg")
cv2.imshow("Original Image",img1)
img=[]
l=1
k=2
m=0
while k<=256:
    img.append(np.zeros((img1.shape[0], img1.shape[1], 3), np.uint8))
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            (b,g,r)=img1[i,j]
            b1=(b%k) - (b%l)
            g1 = (g % k) - (g % l)
            r1 = (r % k) - (r % l)
            img[m][i,j] = (b1,g1,r1)
    l = k
    k = k * 2
    cv2.imshow("img" + str(m),img[m])
    m=m+1
img2 = np.zeros((img1.shape[0],img1.shape[1],3),np.uint8)
img2= img[0] + img[1] + img[2] + img[3] + img[4] + img[5] + img[6] + img[7]
cv2.imshow("bit slices added",img2)
cv2.waitKey(0)