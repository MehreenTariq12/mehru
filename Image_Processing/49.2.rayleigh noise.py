import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
img1 = cv2.imread("images/lena.png")

img = np.zeros((img1.shape[0],img1.shape[1],1), np.float32)
for i in range(img1.shape[0]):
    for j in range(img1.shape[1]):
        img[i,j,0]=img1[i,j,0]

rayleigh = np.random.rayleigh(scale=10.0,size = (img.shape[0],img.shape[1],1))
noisy = np.zeros(img.shape, np.uint8)
noisy = img + rayleigh
cv2.normalize(noisy, noisy, 0, 255, cv2.NORM_MINMAX, dtype=-1)
noisy = noisy.astype(np.uint8)
plt.hist(rayleigh.ravel(),256,[0,256]); plt.show()
img = img.astype(np.uint8)

cv2.imshow("original",img)
cv2.imshow("rayleigh",rayleigh)
cv2.imshow("noisy",noisy)
std=np.var(noisy)
mean=np.mean(noisy)

a = (mean - (math.sqrt((3.14*(std))/(4-3.14))))
b = ((4*(std))/(4-3.14))
xarray=[]
new = []
for i in range(256):
    z=i
    if z<a:
        pz=0
    elif z>=a:
        pow = -(((z-a)**2)/b)
        ex = math.exp(pow)
        pz = (2*(z-a)*ex)/b
    xarray.append(pz)
    new.append(i)
plt.plot(xarray)
plt.xlabel("Intensity Levels")
plt.ylabel("Normalized Probability")
plt.title("Histogram")
plt.show()



cv2.waitKey(0)