import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("images/lowcon.png")
rows = img.shape[0]
cols = img.shape[1]
total = rows * cols
newimg = np.zeros((rows,cols,1),np.uint8)
equalizedimg = np.zeros((rows,cols,1),np.uint8)
for i in range(rows):
    for j in range(cols):
        newimg[i,j,0]=img[i,j,0]
cv2.imshow("original",newimg)
r = []
pr = []
x = []
ps = []
for i in range(256):
    r.append(0)
    pr.append(0)
    x.append(0)
    ps.append(0)
for i in range(rows):
    for j in range(cols):
        intensity = newimg[i,j,0]
        r[intensity] += 1
for i in range(256):
    pr[i]=r[i]/total
    x[i] = i

s= []
for i in range(256):
    sum=0
    for j in range(i):
        sum = sum + pr[j]
    sum = np.round(sum * 255)
    s.append(sum)
for i in range(256):
    ps[i]=s[i]/total


plt.bar(x,pr)
plt.xlabel("Intensity Levels")
plt.ylabel("Normalized Probability")
plt.title("Histogram")
plt.show()

plt.bar(x,ps)
#plt.bar(x,pr)
plt.xlabel("Intensity Levels")
plt.ylabel("Normalized Probability")
plt.title("Histogram")
plt.show()

for x in range(rows):
    for y in range(cols):
        m = newimg[x,y,0]
        equalizedimg[x,y,0] = s[m]
cv2.imshow("equalizes",equalizedimg)

cv2.waitKey(0)
