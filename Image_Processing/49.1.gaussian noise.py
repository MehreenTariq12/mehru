import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
img1 = cv2.imread("images/lena.png")
img = np.zeros((img1.shape[0],img1.shape[1],1), np.uint8)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img[i,j,0]=img1[i,j,0]

cv2.imshow("Original",img)
mean = 5
var = 20
sigma = var ** 0.5
gaussian = np.random.normal(mean, sigma, (img.shape[1], img.shape[1],1)) #  np.zeros((224, 224), np.float32)
noisy = np.zeros(img.shape, np.uint8)
noisy = img + gaussian
#noisy_image= noisy_image % 255
cv2.normalize(noisy, noisy, 0, 255, cv2.NORM_MINMAX, dtype=-1)
noisy = noisy.astype(np.uint8)
plt.hist(gaussian.ravel(),256,[0,256]); plt.show()
std=np.std(noisy)
mean=np.mean(noisy)
pz=0
xarray=[]
new=[]
for i in range(256):
    z = i
    power = (-((z - mean) ** 2) / (2 * (std) ** 2))
    a = math.exp(power)
    pz = a / (math.sqrt(2 * 3.14) * std)
    xarray.append(pz)
    new.append(i)
print(xarray)
cv2.imshow("gaussian", gaussian)
cv2.imshow("noisy", noisy)
plt.plot(xarray)
plt.xlabel("Intensity Levels")
plt.ylabel("Normalized Probability")
plt.title("Histogram")
plt.show()

cv2.waitKey(0)
