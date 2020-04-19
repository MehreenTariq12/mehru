import cv2
import numpy as np
import math
class MarrHildreth:
    def Gaussian(self,img,s):
        rows = img.shape[0]
        cols = img.shape[1]
        n =(6*s)+1
        const = int((6*s)/2)
        new_kernal = []
        k_sum = 0
        for i in range(int(n)):
            new_kernal.insert(i, [])
            for j in range(int(n)):
                power = (-(((i-const)**2) + ((j-const)**2))/ (2 * (s)**2))
                a = math.exp(power)
                k_sum = k_sum+a
                new_kernal[i].insert(j, a)
        new_img = np.zeros((rows+(const*2), cols+(const*2), 1), np.uint8)
        for i in range(rows):
            for j in range(cols):
                new_img[const+i,const+j,0] = img[i,j,0]

        im_gx = np.zeros((rows, cols, 1), np.uint8)
        for i in range(const, rows - const, 1):
            for j in range(const, cols - const, 1):
                sum = 0
                for k in range(-const, const+1, 1):
                    for l in range(-const, const+1, 1):
                        sum = sum + (new_img[i + k, l + j, 0] * new_kernal[k + const][l + const])
                sum = sum / k_sum
                im_gx[i, j, 0] = sum
        return (im_gx)

    def Laplacian(self, inp_img):
        rows = inp_img.shape[0]
        cols = inp_img.shape[1]
        Laplace_image = np.zeros((rows, cols, 1), np.float)
        kernal = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
        for i in range(1, rows - 1, 1):
            for j in range(1, cols - 1, 1):
                sum = 0
                for k in range(-1, 2, 1):
                    for l in range(-1, 2, 1):
                        sum = sum + ((inp_img[i + k, l + j, 0]) * (kernal[k + 1][l + 1]))
                Laplace_image[i, j, 0] = sum
        return Laplace_image
    def normalize(self, img):
        rows = img.shape[0]
        cols = img.shape[1]
        Normalize = np.zeros((rows, cols, 1), np.uint8)
        min = np.min(img)
        max = np.max(img)
        for i in range(0, rows, 1):
            for j in range(0, cols, 1):
                a = img[i, j, 0]
                b = (a - min) / (max - min)
                Normalize[i, j, 0] = b * 255
        return Normalize
    def zero_crossing(self,img,thresh):
        rows = img.shape[0]
        cols = img.shape[1]
        ZeroCross = np.zeros((rows, cols, 1), np.uint8)
        for i in range(1, rows-1, 1):
            for j in range(1, cols-1, 1):
                if ((img[i-1,j-1,0]<0) and (img[i+1,j+1,0]>0)) or ((img[i-1,j-1,0]>0) and (img[i+1,j+1,0]<0)):
                    if img[i,j,0]>thresh:
                        ZeroCross[i,j,0] = 255
                elif ((img[i-1,j+1,0]<0) and (img[i+1,j-1,0]>0)) or ((img[i-1,j+1,0]>0) and (img[i+1,j-1,0]<0)):
                    if img[i,j,0]>thresh:
                        ZeroCross[i,j,0] = 255
                elif ((img[i - 1, j , 0] < 0) and (img[i + 1, j, 0] > 0)) or ((img[i - 1, j, 0] > 0) and (img[i + 1, j, 0] < 0)):
                    if img[i, j, 0] > thresh:
                        ZeroCross[i, j, 0] = 255
                elif ((img[i , j + 1, 0] < 0) and (img[i , j - 1, 0] > 0)) or ((img[i, j + 1, 0] > 0) and (img[i, j - 1, 0] < 0)):
                    if img[i, j, 0] > thresh:
                        ZeroCross[i, j, 0] = 255
                else:
                    ZeroCross[i, j, 0] = 0
        return ZeroCross


def main():
    inp_img1 = cv2.imread("images/building.png")
    cv2.imshow("input image_1", inp_img1)
    object1 = MarrHildreth()
    gaussian = object1.Gaussian(inp_img1,1)
    cv2.imshow("smooth",gaussian)
    LOG = object1.Laplacian(gaussian)
    LOG_NORM = object1.normalize(LOG)
    cv2.imshow("LOG_NORM", LOG_NORM)
    ZerCross = object1.zero_crossing(LOG,0)
    cv2.imshow("Zero_Cross", ZerCross)
    max1 = np.max(LOG)
    percentage  = (4/100)*max1
    ZerCross = object1.zero_crossing(LOG, percentage)
    cv2.imshow("Threshold_Zero_Cross", ZerCross)
    cv2.waitKey(0)

main()