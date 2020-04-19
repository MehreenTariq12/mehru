import cv2
import numpy as np


class Boundary_Processing:
    def Process_boundary(self, img,b,c):
        (b01, b02) = b
        (c01, c02) = c
        loc = 0
        chk = 0
        b1 = (0,0)
        c1 = (0, 0)
        neighbor = [(b01-1,b02-1),(b01-1,b02),(b01-1,b02+1),(b01,b02+1),(b01+1,b02+1),(b01+1,b02),(b01+1,b02-1),(b01,b02-1)]
        for i in range(8):
            if neighbor[i] == c:
                loc = i
                break
        for i in range(loc, 8):
            (x, y) = neighbor[i]
            if img[x, y, 0] > 0:
                b1 = neighbor[i]
                if i == 0:
                    c1 = neighbor[7]
                else:
                    c1 = neighbor[i - 1]
                chk = 0
                break
            else:
                chk = 1
        if chk == 1:
            for i in range(loc):
                (x, y) = neighbor[i]
                if img[x, y, 0] > 0:
                    b1 = neighbor[i]
                    if i == 0:
                        c1 = neighbor[7]
                    else:
                        c1 = neighbor[i - 1]
                    break
        return (b1,c1)



def main():
    object1 = Boundary_Processing()
    img = cv2.imread("images/Boundary.png")
    rows = img.shape[0]
    cols = img.shape[1]
    img2 = np.zeros((rows,cols,1),np.uint8)
    b0 = c0 = (0, 0)
    chk = 0
    for i in range(rows):
        for j in range(cols):
            if img[i, j, 0] > 0:
                c0 = (i, j - 1)
                b0 = (i, j)
                chk = 1
                break
        if chk ==1:
            break
    #print()
    b =b0
    c = c0
    (b1,c1) = object1.Process_boundary(img,b,c)
    img2[b1] = 255
    while b1 != b0:
        b = b1
        c = c1
        (b1, c1) = object1.Process_boundary(img, b, c)
        img2[b1] = 255
        img[b1] = (0,0,255)
    #cv2.imshow("d",img)
    cv2.imshow("1", img)
    cv2.imshow("2", img2)
    cv2.waitKey(0)
main()
