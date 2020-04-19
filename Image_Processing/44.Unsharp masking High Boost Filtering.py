import cv2
import numpy as np
image = cv2.imread("images/moon.jpg")

rows = image.shape[0]
cols = image.shape[1]
img= np.zeros((rows,cols,1),np.uint8)
for i in range(rows):
    for j in range(cols):
        img[i,j,0]=image[i,j,0]
cv2.imshow("Original",img)

img1 = np.zeros((rows,cols,1),np.uint8)
mask = np.zeros((rows,cols,1),np.uint8)
w = [[1,1,1,1,1,1,1], [1,1,1,1,1,1,1], [1,1,1,1,1,1,1], [1,1,1,1,1,1,1], [1,1,1,1,1,1,1], [1,1,1,1,1,1,1], [1,1,1,1,1,1,1]]
new_kernal=[]
#Inverting the kernal
for i in range(len(w)):
    new_kernal.insert(i,[])
    for j in range(len(list(zip(*w)))):
        new_kernal[i].insert(j,w[4-i][4-j])

for i in range(3,rows-3):
    for j in range(3,cols-3):
        bsum = gsum = rsum = 0
        for x in range(-3,4,1):
            for y in range(-3,4,1):
                b = img[i+x, j+y,0]
                bsum = bsum + (b * new_kernal[1+x][1+y])
        bsum=bsum/49
        img1[i, j,0] = bsum
cv2.imshow("Convoluted image",img1)
for i in range(rows):
    for j in range(cols):
        mask[i,j,0] = cv2.subtract(np.uint8([img1[i,j,0]]), np.uint8([img[i,j,0]]))


#mask=img-img1
cv2.imshow("mask",mask)
new=img+mask
cv2.imshow("Unsharp masking",new)
new2=img+(mask*30)
cv2.imshow("Highboost filtering",new)
cv2.waitKey(0)
