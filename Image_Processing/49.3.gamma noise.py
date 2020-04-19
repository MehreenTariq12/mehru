import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
img1 = cv2.imread("images/lena.png")


img = np.zeros((img1.shape[0],img1.shape[1],1), np.float32)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img[i,j,0]=img1[i,j,0]

gamma = np.random.gamma(shape=2,scale=5.0,size = (img.shape[0],img.shape[1],1))
noisy = np.zeros(img.shape, np.uint8)
noisy = img + gamma

cv2.normalize(noisy, noisy, 0, 255, cv2.NORM_MINMAX, dtype=-1)
noisy = noisy.astype(np.uint8)
img = img.astype(np.uint8)
cv2.imshow("original",img)
cv2.imshow("gamma",gamma)
cv2.imshow("noisy",noisy)

plt.hist(gamma.ravel(),256,[0,256]); plt.show()

mean = np.mean(noisy)
std = np.std(noisy)
a = (mean/(std)**2)
b = ((mean)**2/(std)**2)

xarray = []
for i in range(256):
    z = i
    if z<a:
        pz = 0
    elif z>=0:
        n1 = (a**b) * (z**(b-1))
        n2 = -(a*z)
        po = math.exp(n2)
        pz = (n1 * po)/math.factorial(np.uint8(b-1))
    xarray.append(pz)

plt.plot(xarray)
plt.xlabel("Intensity Levels")
plt.ylabel("Normalized Probability")
plt.title("Histogram")
plt.show()




cv2.waitKey(0)