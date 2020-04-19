import cv2
import numpy as np
import math
class LocalProcessing:
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
                ang = np.arctan2(a,b)
                Angle[i, j, 0] = (ang * 180) / np.pi
        return Angle
    def Threshold(self,Magnitude,Angle):
        rows = Magnitude.shape[0]
        cols = Magnitude.shape[1]
        thresh = np.zeros((rows, cols, 1), np.float)
        thresh2 = np.zeros((rows, cols, 1), np.float)
        max1 = np.max(thresh)
        Thresh = (30/100)*max1
        for i in range(rows):
            for j in range(cols):
                if Magnitude[i,j,0]>Thresh:
                    if (Angle[i,j,0]>=-45 and Angle[i,j,0]<=45) or (Angle[i,j,0]>=-180 and Angle[i,j,0]<=-135) or (Angle[i,j,0]>=135 and Angle[i,j,0]<=180):
                        thresh[i,j,0] = 255
                    else:
                        thresh[i, j, 0] = 0
        for i in range(rows):
            for j in range(cols):
                if Magnitude[i,j,0]>Thresh:
                    if (Angle[i,j,0]>=-45 and Angle[i,j,0]<=45) or (Angle[i,j,0]>=-180 and Angle[i,j,0]<=-135) or (Angle[i,j,0]>=135 and Angle[i,j,0]<=180):
                        thresh[i,j,0] = 255
                    else:
                        thresh[i, j, 0] = 0
        #thresh2 = thresh
        #for i in range(rows):
         #   for j in range(cols):
          #      if thresh[i,j,0] > 0:
           #         p = j
            #        for k in range(1,25):
             #           if j+k < cols-1:
              #              if thresh[i,j+k,0] == 255:
               #                 for l in range(p+1,j+k):
                #                    thresh[i,l,0] = 255
                 #               p = j+k


        return thresh


def main():
    inp_img1 = cv2.imread("images/LocalCar.png")
    cv2.imshow("input image_1", inp_img1)
    object1 = LocalProcessing()
    (im_x, im_y) = object1.Sobel(inp_img1)
    abs_x = object1.Absolute(im_x)
    abs_y = object1.Absolute(im_y)
    cv2.imshow("Abs_x", abs_x)
    cv2.imshow("Abs_y", abs_y)
    cv2.waitKey(0)
    Magnitude = object1.Magnitude(abs_x, abs_y)
    cv2.imshow("Magnitude", Magnitude)
    Angle = object1.angle(im_x, im_y)
    cv2.imshow("Angle", Angle)
    cv2.waitKey(0)
    thresh = object1.Threshold(Magnitude,Angle)
    cv2.imshow("Threshold", thresh)
    #cv2.imshow("Threshold2", thresh2)
    cv2.waitKey(0)
main()

