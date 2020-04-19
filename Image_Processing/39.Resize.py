import cv2
import numpy as np
image = cv2.imread("images/1.jpg")
cv2.imshow("Original",image)
r= 550/image.shape[0]
dim = (int(image.shape[1]*r),550)
resized = cv2.resize(image,dim,interpolation=cv2.INTER_AREA)
cv2.imshow("Resized",resized)
cv2.waitKey(0)