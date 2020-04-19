import cv2
import numpy as np
image = cv2.imread("images/moon.jpg")
img=image
cv2.imshow("Original",img)
rows = img.shape[0]
cols = img.shape[1]
img1 = np.zeros((rows,cols,1),np.uint8)
new = np.zeros((rows,cols,1),np.uint8)
w = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
new_kernal=[]
#Inverting the kernal
for i in range(len(w)):
    new_kernal.insert(i,[])
    for j in range(len(list(zip(*w)))):
        new_kernal[i].insert(j,w[2-i][2-j])
print(new_kernal)
for i in range(1,rows-1):
    for j in range(1,cols-1):
        bsum = gsum = rsum = 0
        for x in range(-1,2,1):
            for y in range(-1,2,1):
                b = img[i+x, j+y,0]
                bsum = bsum + (b * new_kernal[1+x][1+y])
        if bsum > 255:
            bsum=255
        elif bsum < 0:
            bsum = 0
        img1[i, j] = bsum
cv2.imshow("Convoluted image",img1)
for i in range(rows):
    for j in range(cols):
        new[i,j,0] = cv2.add(np.uint8([img[i,j,0]]), np.uint8([img1[i,j,0]]))

cv2.imshow("result",new)
cv2.waitKey(0)
