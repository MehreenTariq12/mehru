import cv2
import numpy as np
pic1 = cv2.imread("images/try.jpg")
pic2 = cv2.imread("images/cameraman.jpg")
corner = pic1[100:325,100:325]
cv2.imshow("Image1",corner)
cv2.imshow("Image2",pic2)
result = pic2
for x in range(0,225):
    for y in range(0, 225):
        (b1,g1,r1) = pic2[x,y]
        (b2, g2, r2) = corner[x, y]
        result[x,y] = (b1+b2,g1+g2,r1+r2)
#result=pic2 + corner
cv2.imshow("Result",result)
numpy_horizontal = np.hstack((corner,result))
cv2.imshow('Numpy Horizontal', numpy_horizontal)
numpy_horizontal_concat = np.concatenate((numpy_horizontal, numpy_horizontal))
cv2.imshow('Numpy Horizontal Concat', numpy_horizontal_concat)
cv2.imwrite("images/image addition.jpg",result)
cv2.waitKey(0)