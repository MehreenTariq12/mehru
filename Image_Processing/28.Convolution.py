import cv2
import numpy as np
img = cv2.imread("images/green.jpg")
cv2.imshow("Original",img)
rows = img.shape[0]
cols = img.shape[1]
img1 = np.zeros((rows,cols,3),np.uint8)
w = [[0.3679, 0.6065, 0.3679], [0.6065, 1.0000, 0.6065], [0.3679, 0.6065, 0.3679]]
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
                (b, g, r) = img[i+x, j+y]
                bsum = bsum + (b * new_kernal[1+x][1+y])
                gsum = gsum + (g * new_kernal[1 + x][1 + y])
                rsum = rsum + (r * new_kernal[1 + x][1 + y])
        bsum = bsum / 4.8976
        gsum = gsum / 4.8976
        rsum = rsum / 4.8976
        img1[i, j] = (bsum, gsum, rsum)
cv2.imshow("Convoluted image",img1)
cv2.imwrite("images/Convoluted guassian.jpg",img1)
cv2.waitKey(0)
