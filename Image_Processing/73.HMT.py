import cv2
import numpy as np
import math
class morphology:
    def Erosion(self,img,SE):
        rows = img.shape[0]
        cols = img.shape[1]
        img1 = np.zeros((rows,cols,1),np.uint8)
        #SE = [[1, 1, 1,1,1,1,1], [1, 1, 1,1,1,1,1],[1, 1, 1,1,1,1,1],[1, 1, 1,1,1,1,1],[1, 1, 1,1,1,1,1],[1, 1, 1,1,1,1,1],[1, 1, 1,1,1,1,1]]
        ratio = np.uint8(len(SE)/2)

        for i in range(ratio, rows - ratio):
            for j in range(ratio, cols - ratio):
                r = 0-ratio
                chk = 0
                for m in range(r, ratio+1, 1):
                    for n in range(r, ratio+1, 1):
                        if SE[ratio + m][ratio + n] == 255:
                            if SE[ratio + m][ratio + n] != img[i + m, j + n, 0]:
                                chk = 1
                                break
                if chk == 0:
                    img1[i, j, 0] = 255




        return img1
    def AND(self,img1,img2):
        rows = img1.shape[0]
        cols = img1.shape[1]
        img3 = np.zeros((rows,cols,1),np.uint8)
        for i in range(rows):
            for j in range(cols):
                if img1[i,j] != 0 and img2[i,j]!= 0:
                        img3[i,j]=255
                else:
                    img3[i, j] = 0
        return img3
def main():
    img1 = np.zeros((180, 210, 1), np.uint8)
    cv2.rectangle(img1, (30, 30), (90, 110), 255, -1)
    cv2.rectangle(img1, (100, 130), (124, 154), 255, -1)
    cv2.rectangle(img1, (160, 50), (181, 71), 255, -1)
    #cv2.imshow("Original", img1)
    B1 = []
    for i in range(25):
        B1.insert(i, [])
        for j in range(25):
            B1[i].insert(j, 255)
    B2 = []
    for i in range(27):
        B2.insert(i, [])
        for j in range(27):
            if i == 0 or j == 0 or i == 26 or j == 26:
                B2[i].insert(j, 255)
            else:
                B2[i].insert(j, 0)
    img2 = 255-img1
    cv2.imshow("Original", img1)
    cv2.imshow("Complement", img2)
    object1 = morphology()
    eroded1 = object1.Erosion(img1, B1)
    eroded2 = object1.Erosion(img2, B2)
    cv2.imshow("Erosion1", eroded1)
    cv2.imshow("Erosion2", eroded2)
    intersection = object1.AND(eroded1,eroded2)
    cv2.imshow("HMT", intersection)

    cv2.waitKey(0)
main()