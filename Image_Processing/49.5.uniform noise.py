import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
img1 = cv2.imread("images/lena.png")
img = np.zeros((img1.shape[0],img1.shape[1],1), np.float32)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img[i,j,0]=img1[i,j,0]
uniform = np.random.uniform(low=0.0,high=20.0,size = (img.shape[0],img.shape[1],1))
noisy = np.zeros(img.shape, np.uint8)
noisy = img + uniform
cv2.normalize(noisy, noisy, 0, 255, cv2.NORM_MINMAX, dtype=-1)
noisy = noisy.astype(np.uint8)
img = img.astype(np.uint8)
plt.hist(noisy.ravel(),256,[0,256]); plt.show()

xarray = []
mean = np.mean(noisy)
std = np.std(noisy)
a = mean - math.sqrt(3 * (std)**2)
b = mean + math.sqrt(3 * (std)**2)
for i in range(256):
    z=i
    if a<=z and z<=b:
        pz = 1/(b-a)
    else:
        pz = 0
    xarray.append(pz)
cv2.imshow("original",img)
cv2.imshow("uniform",uniform)
cv2.imshow("noisy",noisy)

plt.plot(xarray)
plt.xlabel("Intensity Levels")
plt.ylabel("Normalized Probability")
plt.title("Histogram")
plt.show()

cv2.waitKey(0)