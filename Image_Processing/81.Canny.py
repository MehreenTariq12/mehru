import cv2
import numpy as np
import math
class Canny:
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
    def Sobel(self, inp_img):
        rows = inp_img.shape[0]
        cols = inp_img.shape[1]
        im_gx = np.zeros((rows, cols, 1), np.float32)
        im_gy = np.zeros((rows, cols, 1), np.float32)
        gx = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        gy = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        for i in range(1, rows - 1, 1):
            for j in range(1, cols - 1, 1):
                sum1 = sum2 = 0
                for k in range(-1, 2, 1):
                    for l in range(-1, 2, 1):
                        sum1 = sum1 + (inp_img[i + k, l + j, 0]/255 * gx[k + 1][l + 1])
                        sum2 = sum2 + (inp_img[i + k, l + j, 0]/255 * gy[k + 1][l + 1])
                    im_gx[i, j, 0] = sum1
                    im_gy[i, j, 0] = sum2
        return (im_gx, im_gy)

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

    def Magnitude(self, img1,img2):
        rows = img1.shape[0]
        cols = img1.shape[1]
        Magnitude = np.zeros((rows, cols, 1), np.uint8)
        for i in range(0, rows, 1):
            for j in range(0, cols, 1):
                Magnitude[i,j,0] = ((img1[i,j,0]**2)+(img2[i,j,0]**2))**0.5

        return Magnitude

    def Absolute(self, img):
        rows = img.shape[0]
        cols = img.shape[1]
        Abs = np.zeros((rows, cols, 1), np.uint8)
        for i in range(rows):
            for j in range(cols):
                a = abs(img[i, j, 0])*255
                if a > 255:
                    Abs[i, j, 0] = 255
                else:
                    Abs[i, j, 0] = a
        return Abs

    def angle(self,x,y):
        rows = x.shape[0]
        cols = x.shape[1]
        Angle = np.zeros((rows, cols, 1),np.float)
        for i in range(rows):
            for j in range(cols):
                a = y[i,j,0]
                b = x[i,j,0]
                Angle[i,j,0] = np.arctan2(a,b)
        return Angle

    def Supression(self,angle,magnitude):
        rows = angle.shape[0]
        cols = angle.shape[1]
        Supressed = np.zeros((rows, cols, 1),np.uint8)
        for i in range(1, rows-1,1):
            for j in range(1,cols-1,1):
                ang = (angle[i,j,0]*180)/np.pi
                a45 = abs(ang - 45)
                an45 = abs(ang + 45)
                ahor = abs(ang - 0)
                aver = abs(ang - 90)
                (n1x, n1y, n2x, n2y) = (0,0,0,0)
                if a45 < an45 and a45 < ahor and a45 < aver:
                    (n1x, n1y, n2x, n2y) = (i + 1, j - 1, i-1, j+1)
                elif an45 < a45 and an45 < ahor and an45 < aver:
                    (n1x, n1y, n2x, n2y) = (i - 1, j - 1, i + 1, j + 1)
                elif ahor < a45 and ahor < an45 and ahor < aver:
                    (n1x, n1y, n2x, n2y) = (i - 1, j, i + 1, j)
                elif aver < a45 and aver < an45 and aver < ahor:
                    (n1x, n1y, n2x, n2y) = (i, j - 1, i, j + 1)
                if (magnitude[n1x,n1y,0] > magnitude[i,j,0]) or (magnitude[n2x,n2y,0] > magnitude[i,j,0]):
                    Supressed[i,j,0] = 0
                else:
                    Supressed[i, j, 0] = magnitude[i,j,0]
        return Supressed


    def Threshold(self,img):
        rows = img.shape[0]
        cols = img.shape[1]
        gNL = np.zeros((rows, cols, 1), np.float)
        gNH = np.zeros((rows, cols, 1), np.float)
        for i in range(1, rows - 1, 1):
            for j in range(1, cols - 1, 1):
                Tl = 0.04
                Th = 0.1
                if img[i,j,0]/255 >= Tl:
                    gNL[i,j,0] = img[i,j,0]/255
                if img[i,j,0]/255 >= Th:
                    gNH[i,j,0] = img[i,j,0]/255
                gNL[i,j,0] = gNL[i,j,0] - gNH[i,j,0]
        return (gNL,gNH)
    def Last_Step(self,NH,NL):
        rows = NH.shape[0]
        cols = NL.shape[1]
        canny = np.zeros((rows, cols, 1), np.float)
        for i in range(1,rows-1):
            for j in range(1,cols-1):
                if NH[i,j,0] > 0:
                    if NL[i-1,j-1,0]>0:
                        canny[i-1,j-1,0] = NL[i-1,j-1,0]
                    if NL[i-1,j,0]>0:
                        canny[i-1, j, 0] = NL[i-1, j, 0]
                    if NL[i-1,j+1,0]>0:
                        canny[i-1,j+1,0] = NL[i-1,j+1,0]
                    if NL[i,j-1,0]>0:
                        canny[i,j-1,0] = NL[i,j-1,0]
                    if NL[i,j+1,0]>0:
                        canny[i,j+1,0] = NL[i,j+1,0]
                    if NL[i+1,j-1,0]>0:
                        canny[i+1,j-1,0] = NL[i+1,j-1,0]
                    if NL[i+1,j,0]>0:
                        canny[i+1,j,0] = NL[i+1,j,0]
                    if NL[i+1,j+1,0]>0:
                        canny[i+1,j+1,0] = NL[i+1,j+1,0]
                    canny[i,j,0] = NH[i, j, 0]
        return canny


def main():
    inp_img1 = cv2.imread("images/EdgeHouse.png")
    cv2.imshow("input image_1", inp_img1)
    object1 = Canny()
    Smooth = object1.Gaussian(inp_img1,2)
    cv2.imshow("Smooth", Smooth)
    (im_x,im_y) = object1.Sobel(Smooth)
    abs_x = object1.Absolute(im_x)
    abs_y = object1.Absolute(im_y)
    cv2.imshow("Abs_x", abs_x)
    cv2.imshow("Abs_y", abs_y)
    cv2.waitKey(0)
    Magnitude = object1.Magnitude(abs_x,abs_y)
    cv2.imshow("Magnitude", Magnitude)
    Angle = object1.angle(im_x,im_y)
    cv2.imshow("Angle", Angle)
    cv2.waitKey(0)
    #Suppressed = object1.non_max_suppression(Magnitude,Angle)
    Suppressed = object1.Supression(Angle,Magnitude)
    (gNL,gNH) = object1.Threshold(Suppressed)
    canny = object1.Last_Step(gNH,gNL)
    cv2.imshow("Suppressed", Suppressed)
    cv2.imshow("gNL", gNL)
    cv2.imshow("gNH", gNH)
    cv2.imshow("Canny", canny)
    cv2.waitKey(0)

main()