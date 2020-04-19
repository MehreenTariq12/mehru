import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("images/bgr.png")
x=[0,0,0,0,0,0,0,0,0,0,0,0,0,0]

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        (b, g, r) = img[i, j]
        for k in range(20,280,20):
            if b<=k:
                a=(k/20)-1
                x[int(a)] += 1
                break
        for l in range(20,280,20):
            if g<=l:
                x[int((l/20)-1)] += 1
                break
        for m in range(20,280,20):
            if r<=m:
                x[int((m/20)-1)] += 1
                break
cv2.imshow("Image",img)
for i in range (len(x)):
    print("Pixels in range:{} to range: {}".format(i*20,(i+1)*20))
    print(x[i])
y=[20,40,60,80,100,120,140,160,180,200,210,220,240,260]
plt.bar(y,x,width=20)
plt.show()

cv2.waitKey(0)