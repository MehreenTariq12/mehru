import cv2
import numpy as np
print("Alpha Trimmed filter")
image = cv2.imread("images/gray2.png")
gray = np.zeros((image.shape[0]+2,image.shape[1]+2,1),np.uint8)
filtered_image=np.zeros((image.shape[0]+2,image.shape[1]+2,1),np.uint8)
for x in range(0,image.shape[0]):
    for y in range(0,image.shape[1]):
        gray[x+1,y+1,0]=image[x,y,0]

cv2.imshow("Original",gray)
for i in range(1,gray.shape[0]-1):
    for j in range(1,gray.shape[1]-1):
        m=[]
        for x in range(-1,2,1):
            for y in range(-1,2,1):
                m.append(gray[i+x,j+y,0])
        m.sort()
        sum = 0
        for z in range(2,7):
            sum=sum+m[z]
        sum = sum/5
        filtered_image[i,j,0] = sum
cv2.imshow("Order statistic filter", filtered_image)
cv2.waitKey(0)


