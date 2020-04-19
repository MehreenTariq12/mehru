import cv2
import numpy as np
image = cv2.imread("images/brain.jpg")
cv2.imshow("org",image)
rows = image.shape[0]
cols = image.shape[1]
new_img = np.zeros((rows,cols,1),np.uint8)
new_img2 = np.zeros((rows,cols,3),np.uint8)
for i in range(rows):
    for j in range(cols):
        new_img[i,j] = image[i,j,0]

for i in range(rows):
    for j in range(cols):
        pix = new_img[i,j,0]
        if pix > 20 and pix <= 25:
            new_img2[i,j] = (pix,pix,100)
        elif pix > 25 and pix <= 50:
            new_img2[i,j] = (pix,100,pix)
        elif pix > 50 and pix <= 75:
            new_img2[i,j] = (100,pix,pix)
        elif pix > 75 and pix <= 100:
            new_img2[i,j] = (pix,pix,150)
        elif pix > 100 and pix <= 125:
            new_img2[i,j] = (pix,150,pix)
        elif pix > 125 and pix <= 150:
            new_img2[i,j] = (150,pix,pix)
        elif pix > 150 and pix <= 175:
            new_img2[i,j] = (150,pix,150)
        elif pix > 175 and pix <= 200:
            new_img2[i, j] = (pix, pix, 150)
        else:
            new_img2[i, j] = (pix, pix, pix)


cv2.imshow("1",new_img2)
cv2.waitKey(0)