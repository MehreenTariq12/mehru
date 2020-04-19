import cv2
import numpy as np
inp_img1 = cv2.imread("images/point.png")
cv2.imshow("inp_img", inp_img1)
rows = inp_img1.shape[0]
cols = inp_img1.shape[1]
Laplace_image = np.zeros((rows,cols,1),np.float)
point = np.zeros((rows,cols,1),np.uint8)
#cv2.imshow("output", inp_img1)
kernal = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]

for i in range(1,rows-1,1):
    for j in range(1,cols-1,1):
        sum = 0
        for k in range(-1,2,1):
            for l in range(-1, 2, 1):
                sum = sum + (inp_img1[i+k,l+j,0] * kernal[k+1][l+1])
                #print(sum)
                #print(sum)
                #print(np.abs(sum))
        Laplace_image[i,j,0] = (sum)
        #print(Laplace_image[i,j,0])
cv2.imshow("output", Laplace_image)
min = np.min(Laplace_image)
max = np.max(Laplace_image)
print(min,max)
for i in range(0, rows , 1):
    for j in range(0, cols , 1):
        a = Laplace_image[i,j,0]
        b = (a-min)/(max-min)
        point[i,j,0] = b*255
        print(point[i,j,0])
for i in range(0, rows, 1):
    for j in range(0, cols , 1):
        if point[i,j,0]> 173:
            point[i,j,0] = 255
        else:
            point[i, j, 0] = 0
cv2.imshow("point",point)

cv2.waitKey(0)