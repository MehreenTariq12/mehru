import cv2
import numpy as np
image = cv2.imread("images/gray2.png")
# a geometric mean filter achieves smoothing comparable to an arithmetic mean filter, but it tends to lose less image detail in the process.
gray = np.zeros((image.shape[0]+2,image.shape[1]+2,1),np.float)
filtered_image=np.zeros((image.shape[0]+2,image.shape[1]+2,1),np.float)
for x in range(0,image.shape[0]):
    for y in range(0,image.shape[1]):
        gray[x+1,y+1,0]=image[x,y,0]

for i in range(1,gray.shape[0]-1):
    for j in range(1,gray.shape[1]-1):
        mul=1
        for x in range(-1,2,1):
            for y in range(-1,2,1):
                mul= mul* gray[i+x,j+y,0]
        filtered_image[i,j,0]= (mul)**(1/9)
filtered_image = filtered_image.astype(np.uint8)
gray = gray.astype(np.uint8)

cv2.imshow("Original",gray)
cv2.imshow("geomatric mean filter", filtered_image)
cv2.waitKey(0)


