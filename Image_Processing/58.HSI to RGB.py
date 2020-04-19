import cv2
import numpy as np
import math
image = cv2.imread("images/color.jpeg")
cv2.imshow("Original",image)
rows = image.shape[0]
cols = image.shape[1]
HSI = np.zeros((rows,cols,3),np.float)
RGB = np.zeros((rows,cols,3),np.float)
for i in range(rows):
    for j in range(cols):
        (b1,g1,r1) = image[i,j]
        b = b1/255
        g = g1/255
        r = r1/255

        I = (r+g+b)/3

        if b < g and b < r:
            min = b
        elif g < b and g < r:
            min = g
        else:
            min = r
        num5 = (min*3/(r+g+b+0.0000001))
        S = 1 - num5

        num = 0.5 *((r-g)+(r-b))/math.sqrt((r-g)**2+((r-b)*(g-b))+0.000001)
        theta = math.acos(num)

        if b <= g:
            H = theta
        elif b > g:
            H = 360 - theta
        HSI[i, j] = (I, S, H)

cv2.imshow("HSI",HSI)

for i in range(rows):
    for j in range(cols):
        (I,S,H) = HSI[i,j]
        if 0<=H and H<120:
            B= I *(1-S)
            R = I *(1 + ((S*math.cos(H))/math.cos(60 - H)))
            G = 3*I-(R+B)

        elif 120 <= H and H < 240:
            H = H-120
            R = I * (1 - S)
            G = I *(1 + ((S*math.cos(H))/math.cos(60 - H)))
            B = (3*I)-(R+G)

        elif 240 <= H and H < 360:
            H = H - 240
            G = I * (1 - S)
            B = I *(1 + ((S*math.cos(H))/math.cos(60 - H)))
            R = (3 * I) - (G + B)
        RGB[i,j] = (B,G,R)


cv2.imshow("BGR",RGB)

cv2.waitKey(0)

