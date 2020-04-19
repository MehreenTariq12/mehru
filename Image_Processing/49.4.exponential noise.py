import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
img1 = cv2.imread("images/gaus.jpg")


img = np.zeros((img1.shape[0],img1.shape[1],1), np.float32)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img[i,j,0]=img1[i,j,0]

exponential = np.random.exponential(scale=10.0,size = (img.shape[0],img.shape[1],1))
noisy = np.zeros(img.shape, np.uint8)
noisy = img + exponential
cv2.normalize(noisy, noisy, 0, 255, cv2.NORM_MINMAX, dtype=-1)
noisy = noisy.astype(np.uint8)
img = img.astype(np.uint8)
plt.hist(noisy.ravel(),256,[0,256]); plt.show()
xarray = []
mean = np.mean(noisy)
std = np.std(noisy)
a= 1/mean
for i in range(256):
    z=i
    if z<a:
        pz=0
    elif z>=0:
        po = -(a*z)
        pz = a * math.exp(po)
    xarray.append(pz)
cv2.imshow("original",img)
cv2.imshow("exponential",exponential)
cv2.imshow("noisy",noisy)

plt.plot(xarray)
plt.xlabel("Intensity Levels")
plt.ylabel("Normalized Probability")
plt.title("Histogram")
plt.show()


cv2.waitKey(0)