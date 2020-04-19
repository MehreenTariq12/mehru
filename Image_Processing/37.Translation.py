import cv2
import numpy as np
image = cv2.imread("images/5.jpg")
M=np.float32([[1,0,25],[0,1,50]])
shifted = cv2.warpAffine(image,M,(image.shape[1],image.shape[0]))
cv2.imshow("Image1",shifted)
M=np.float32([[1,0,-50],[0,1,-90]])
shifted2 = cv2.warpAffine(image,M,(image.shape[1],image.shape[0]))
cv2.imshow("Image2",shifted2)
M=np.float32([[1,0,0],[0,1,100]])
shifted3 = cv2.warpAffine(image,M,(image.shape[1],image.shape[0]))
cv2.imshow("Image3",shifted3)
cv2.waitKey(0)