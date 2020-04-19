import cv2
import numpy as np
image = cv2.imread("images/red.png")
cv2.imshow("original",image)
rows = image.shape[0]
cols = image.shape[1]
new_img = np.zeros((rows,cols,3),np.float)
center = (0.1922, 0.1608, 0.6863)
radius=0.2765

for i in range(rows):
    for j in range(cols):
        new_img[i,j] = image[i,j]/255

for i in range(rows):
    for j in range(cols):
        (b,g,r) = new_img[i,j]
        res = (b - center[0])**2 + (g - center[1])**2 + (r - center[2])**2
        if res> (radius)**2:
           new_img[i,j] = 0.5

cv2.imshow("1",new_img)
cv2.waitKey(0)

