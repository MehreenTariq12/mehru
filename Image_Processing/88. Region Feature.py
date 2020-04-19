import cv2
import numpy as np
class RegionFeature:
    def Area(self,img):
        rows = img.shape[0]
        cols = img.shape[1]
        Acount = 0
        for i in range(rows):
            for j in range(cols):
                if img[i,j,0] > 0:
                    Acount +=1
        return Acount


    def Perimeter(self,img,b,c):
        (b01, b02) = b
        (c01, c02) = c
        loc = 0
        chk = 0
        b1 = (0, 0)
        c1 = (0, 0)
        neighbor = [(b01 - 1, b02 - 1), (b01 - 1, b02), (b01 - 1, b02 + 1), (b01, b02 + 1), (b01 + 1, b02 + 1),
                    (b01 + 1, b02), (b01 + 1, b02 - 1), (b01, b02 - 1)]
        for i in range(9):
            if neighbor[i] == c:
                loc = i
                break
        for i in range(loc, 8):
            (x, y) = neighbor[i]
            if img[x, y, 0] > 0:
                b1 = neighbor[i]
                if i == 0:
                    c1 = neighbor[8]
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
                        c1 = neighbor[8]
                    else:
                        c1 = neighbor[i - 1]
                    break
        return (b1, c1)


def main():
    object1 = RegionFeature()
    img = cv2.imread("images/RegionFeature.png")
    cv2.imshow("Oeiginal", img)
    p = Perimeter2(img)
    print("Perimeter = ",p)
    A = object1.Area(img)
    print("Area = ", A)
    Compactness = (p ** 2)/A
    print("Compactness = ", Compactness)
    Circularity = (4*3.14*A)/(p**2)
    print("Circularity = ",Circularity)
    Effective_Diameter = 2*((A/3.14)**0.5)
    print("Effective_Diameter = ", Effective_Diameter)
    cv2.waitKey(0)




def Perimeter2(img):
    object1 = RegionFeature()
    rows = img.shape[0]
    cols = img.shape[1]
    img2 = np.zeros((rows, cols, 1), np.uint8)
    b0 = c0 = (0, 0)
    chk = 0
    for i in range(rows):
        for j in range(cols):
            if img[i, j, 0] > 0:
                c0 = (i, j - 1)
                b0 = (i, j)
                chk = 1
                break
        if chk == 1:
            break
    b = b0
    c = c0
    (b1, c1) = object1.Perimeter(img, b, c)
    pcount = 1
    while b1 != b0:
        b = b1
        c = c1
        (b1, c1) = object1.Perimeter(img, b, c)
        pcount += 1
        img2[b1] = 255
    cv2.imshow("Perimeter", img2)
    return pcount


main()
