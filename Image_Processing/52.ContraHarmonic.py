import cv2
import numpy as np
image = cv2.imread("images/Sap.jpg")
#This filter is well suited for reducing or virtually eliminating the effects of salt-and-pepper noise. For positive values of Q, the filter eliminates pepper noise. For negative values of Q, it eliminates salt noise.
gray = np.zeros((image.shape[0]+2,image.shape[1]+2,1),np.float)
filtered_image=np.zeros((image.shape[0]+2,image.shape[1]+2,1),np.float)
for x in range(0,image.shape[0]):
    for y in range(0,image.shape[1]):
        gray[x+1,y+1,0]=image[x,y,0]

for i in range(1,gray.shape[0]-1):
    for j in range(1,gray.shape[1]-1):
        sum1=0
        sum2=0
        for x in range(-1,2,1):
            for y in range(-1,2,1):
                z = (gray[i+x,j+y,0])
                sum1= sum1 + (z**1.05)
                if z ==0:
                    sum2=sum2+0
                else:
                    sum2 = sum2 + (z ** 0.05)
        filtered_image[i,j,0]= sum1/sum2
filtered_image = filtered_image.astype(np.uint8)
gray = gray.astype(np.uint8)

cv2.imshow("Original",gray)
cv2.imshow("geomatric mean filter", filtered_image)
cv2.waitKey(0)


