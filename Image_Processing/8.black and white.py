import cv2
import numpy
img=cv2.imread("images/try.jpg")
cv2.imshow("real image", img)
for x in range(0,img.shape[0]):
    for y in range(0,img.shape[1]):
        m=img[x,y,0]
        img[x,y]=m
        #img.itemset((x,y,0),0)
        #img.itemset((x,y,1),0)
cv2.imshow(" black and white image", img)
cv2.imwrite("images/black&white.jpg",img)
cv2.waitKey(0)