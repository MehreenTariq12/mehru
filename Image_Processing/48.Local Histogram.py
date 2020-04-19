import cv2
import numpy as np
img1 = cv2.imread("images/local.png")
rows = img1.shape[0]
cols = img1.shape[1]
total = rows *cols
newimg1 = np.zeros((rows,cols,1),np.uint8)
newimg2 = np.zeros((rows,cols,1),np.uint8)
for i in range(rows):
    for j in range(cols):
        newimg1[i,j,0]=img1[i,j,0]
cv2.imshow("original",newimg1)

r = []
pr = []
x = []
for y in range(256):
    r.append(0)
    pr.append(0)
    x.append(0)
for i in range(1,rows-1):
    for j in range(1,cols-1):
        for y in range(256):
            r[y]=0
            pr[y]=0
            x[y]=y
        for m in range (-1,2,1):
            for n in range(-1,2,1):
                intensity = newimg1[i+m,j+n,0]
                r[intensity] += 1

        for f in range(256):
            pr[f]=r[f]/9
        #print(pr)
        center=img1[i,j,0]
        sum = 0
        for p in range(center+1):
            sum = sum + pr[p]
        sum = sum * 255
        sum = round(sum)
        if sum>255:
            sum=255
        elif sum<0:
            sum = 0
        newimg2[i,j,0] = sum

cv2.imshow("new",newimg2)

cv2.waitKey(0)

