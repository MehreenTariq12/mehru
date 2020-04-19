import numpy as np
import random
from matplotlib import pyplot as plt
import cv2
image = cv2.imread("images/gaus.jpg",0)
cv2.imshow('Original', image)
salt_and_pepper = np.zeros((image.shape[0],image.shape[1],1),np.uint8)
p=0.05
q=1-0.05
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        n = random.random()
        if n<p:
            salt_and_pepper[i,j]=0
        elif n>q:
            salt_and_pepper[i, j] = 255
        else:
            salt_and_pepper[i,j]=image[i,j]
plt.hist(salt_and_pepper.ravel(),256,[0,256]); plt.show()
cv2.imshow('salt_and_pepper', salt_and_pepper)
cv2.imwrite("images/Sap.jpg",salt_and_pepper)
cv2.waitKey(0)
