import cv2
import numpy as np
class Hole:
    def find_hole_pix(self,img):
        SE = [['+',255,'+'],[255,0,'+'],['+','+','+']]
        rows = img.shape[0]
        cols = img.shape[1]
        chk = 0
        req=(0,0)
        for i in range(rows):
            for j in range(cols):
                chk = 0
                for x in range(-1,2,1):
                    for y in range(-1,2,1):
                        if SE[x + 1][y + 1] != '+':
                            if SE[x + 1][y + 1] != img[i + x, j + y, 0]:
                                chk = 1
                if(chk==0):
                    req = (i,j)
                    return req
    def Dilution(self,img):
        rows = img.shape[0]
        cols = img.shape[1]
        img1 = np.zeros((rows, cols, 1), np.uint8)
        SE = [[255,255,255],[255,255,255],[255,255,255]]
        ratio = np.uint8(len(SE) / 2)

        for i in range(ratio, rows - ratio):
            for j in range(ratio, cols - ratio):
                r = 0 - ratio
                chk = 0
                for m in range(r, ratio + 1, 1):
                    for n in range(r, ratio + 1, 1):
                            if SE[ratio + m][ratio + n] == img[i + m, j + n, 0]:
                                chk = 1
                                break
                if chk == 1:
                    img1[i, j, 0] = 255
        return img1

    def gray(self,img):
        rows = img.shape[0]
        cols = img.shape[1]
        bandw = np.zeros((rows,cols,1),np.uint8)
        for i in range(rows):
            for j in range(cols):
                bandw[i,j,0] = img[i,j,0]
        return bandw
    def Subtract(self,original,diluted):
        rows = original.shape[0]
        cols = original.shape[1]
        complement = np.zeros((rows,cols,1),np.uint8)
        x = np.zeros((rows, cols, 1), np.uint8)
        for i in range(rows):
            for j in range(cols):
                complement[i,j,0] = 255 - original[i,j,0]
                if complement[i,j,0] == diluted[i,j,0]:
                    x[i,j,0] = diluted[i,j,0]
        return x




def main():
    img = cv2.imread("images/hole2.png")
    img0 = np.zeros((img.shape[0],img.shape[1],1),np.uint8)
    final = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    rows = img.shape[0]
    cols = img.shape[1]
    object1 = Hole()
    grayimg = object1.gray(img)
    cv2.imshow("original",grayimg)
    (x,y) = object1.find_hole_pix(grayimg)
    img0[x,y,0] = 255
    while True:
        newimg2 = object1.Dilution(img0)
        rzlt = object1.Subtract(grayimg,newimg2)
        chk1 = 0
        for k in range(rows):
            for l in range(cols):
                if rzlt[k,l,0] != img0[k,l,0]:
                    chk1 = 1
                    break
        if chk1 ==1:
            img0 = rzlt
            continue
        else:
            break

    cv2.imshow("hole", rzlt)

    for i in range(rows):
        for j in range(cols):
            if grayimg[i,j,0]==255 or rzlt[i,j,0]==255:
                final[i,j,0] = 255
    cv2.imshow("filled", final)

    cv2.waitKey(0)
main()
