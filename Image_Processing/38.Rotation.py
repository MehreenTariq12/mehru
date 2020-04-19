import cv2
import numpy as np
image = cv2.imread("images/1.jpg")
cv2.imshow("Original",image)
(h,w)=image.shape[:2]
center=(w//2,h//2)
M = cv2.getRotationMatrix2D(center,45,1.0)
rotated = cv2.warpAffine(image,M,(h,w))
cv2.imshow("45 Rotated",rotated)
M = cv2.getRotationMatrix2D(center,-90,1.0)
rotated2 = cv2.warpAffine(image,M,(h,w))
cv2.imshow("-90 Rotated",rotated2)
cv2.waitKey(0)