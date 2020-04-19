import cv2
import numpy as np
import matplotlib.pyplot as plt
img1 = cv2.imread("images/lowcon.png")
img2 = cv2.imread("images/download.jpg")
rows1 = img1.shape[0]
cols1 = img1.shape[1]
total1 = rows1 * cols1
rows2 = img2.shape[0]
cols2 = img2.shape[1]
total2 = rows2 * cols2
newimg1 = np.zeros((rows1,cols1,1),np.uint8)
newimg2 = np.zeros((rows2,cols2,1),np.uint8)
matchedimg = np.zeros((rows1,cols1,1),np.uint8)
for i in range(rows1):
    for j in range(cols1):
        newimg1[i,j,0]=img1[i,j,0]
for i in range(rows2):
    for j in range(cols2):
        newimg2[i,j,0]=img2[i,j,0]
cv2.imshow("original",newimg1)
cv2.imshow("Specified",newimg2)
r1=[]
r2=[]
pr1=[]
pr2=[]
s=[]
t=[]
new=[]
x=[]
for i in range(256):
    r1.append(0)
    r2.append(0)
    pr1.append(0)
    pr2.append(0)
    s.append(0)
    t.append(0)
    x.append(0)
    new.append(0)
for i in range(rows1):
    for j in range(cols1):
        intensity1 = newimg1[i,j,0]
        r1[intensity1] += 1
for i in range(256):
    pr1[i]=r1[i]/total1
    x[i] = i
plt.bar(x,pr1)
plt.xlabel("Intensity Levels")
plt.ylabel("Normalized Probability")
plt.title("Histogram")
plt.show()

for i in range(rows2):
    for j in range(cols2):
        intensity2 = newimg2[i,j,0]
        r2[intensity2] += 1
for i in range(256):
    pr2[i]=r2[i]/total2

plt.bar(x,pr2)
plt.xlabel("Intensity Levels")
plt.ylabel("Normalized Probability")
plt.title("Histogram")
plt.show()

for i in range(256):
    sum1 = 0
    sum2 =0
    for j in range(i):
        sum1 = sum1 + pr1[j]
        sum2 = sum2 + pr2[j]
    sum1 = round(sum1 * 255)
    sum2 = round(sum2 * 255)
    s[i] = sum1
    t[i] = sum2
for z in range(256):
    pivot = s[z]
    for y in range(256):
        if t[y] == pivot:
            new[z] = y
            break
        elif t[y] > pivot:
            prev = np.abs(pivot - t[y-1])
            next = np.abs(pivot - t[y])
            if prev < next:
                new[z] = y-1
                break
            else:
                new[z] = y
                break
plt.bar(x,new)
plt.xlabel("Intensity Levels")
plt.ylabel("Normalized Probability")
plt.title("Histogram")
plt.show()
for i in range(rows1):
    for j in range(cols1):
        m = newimg1[i,j,0]
        matchedimg[i,j,0] = new[m]
cv2.imshow("equalizes",matchedimg)


cv2.waitKey(0)
