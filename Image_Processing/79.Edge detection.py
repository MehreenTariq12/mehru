import cv2
import numpy as np
import math
class Edge_Detection:
    def Robert_Cross_Gradient(self, inp_img):
        rows = inp_img.shape[0]
        cols = inp_img.shape[1]
        im_gx = np.zeros((rows, cols, 1), np.float)
        im_gy = np.zeros((rows, cols, 1), np.float)
        gx = [[0, 0, 0], [0, 0, -1], [0, 1, 0]]
        gy = [[0, 0, 0], [0, -1, 0], [0, 0, 1]]
        for i in range(1, rows - 1, 1):
            for j in range(1, cols - 1, 1):
                sum1 = sum2 = 0
                for k in range(-1, 2, 1):
                    for l in range(-1, 2, 1):
                        sum1 = sum1 + (inp_img[i + k, l + j, 0]/255 * gx[k + 1][l + 1]/8)
                        sum2 = sum2 + (inp_img[i + k, l + j, 0]/255 * gy[k + 1][l + 1]/8)
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

    def Absolute(self, img):
        rows = img.shape[0]
        cols = img.shape[1]
        Abs = np.zeros((rows, cols, 1), np.uint8)
        for i in range(rows):
            for j in range(cols):
                a = abs(img[i, j, 0])*255*8
                if a > 255:
                    Abs[i, j, 0] = 255
                else:
                    Abs[i, j, 0] = a
        return Abs

    def angle(self,x,y):
        rows = x.shape[0]
        cols = y.shape[1]
        Angle = np.zeros((rows, cols, 1), float)
        for i in range(rows):
            for j in range(cols):
                a = y[i,j,0]*255*8
                b = x[i,j,0]*255*8
                if b!=0:
                    c= a/b
                else:
                    c = 0
                Angle[i,j,0] = math.atan(c)
        return Angle


    def Sobel(self, inp_img):
        rows = inp_img.shape[0]
        cols = inp_img.shape[1]
        im_gx = np.zeros((rows, cols, 1), np.uint8)
        im_gy = np.zeros((rows, cols, 1), np.uint8)
        gx = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        gy = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        for i in range(1, rows - 1, 1):
            for j in range(1, cols - 1, 1):
                sum1 = sum2 = 0
                for k in range(-1, 2, 1):
                    for l in range(-1, 2, 1):
                        sum1 = sum1 + (inp_img[i + k, l + j, 0] * gx[k + 1][l + 1])
                        sum2 = sum2 + (inp_img[i + k, l + j, 0] * gy[k + 1][l + 1])
                    im_gx[i, j, 0] = abs(sum1)
                    im_gy[i, j, 0] = abs(sum2)
        return (im_gx, im_gy)

    def Prewitt(self, inp_img):
        rows = inp_img.shape[0]
        cols = inp_img.shape[1]
        im_gx = np.zeros((rows, cols, 1), np.uint8)
        im_gy = np.zeros((rows, cols, 1), np.uint8)
        gx = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
        gy = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
        for i in range(1, rows - 1, 1):
            for j in range(1, cols - 1, 1):
                sum1 = sum2 = 0
                for k in range(-1, 2, 1):
                    for l in range(-1, 2, 1):
                        sum1 = sum1 + (inp_img[i + k, l + j, 0] * gx[k + 1][l + 1])
                        sum2 = sum2 + (inp_img[i + k, l + j, 0] * gy[k + 1][l + 1])
                    im_gx[i, j, 0] = abs(sum1)
                    im_gy[i, j, 0] = abs(sum2)
        return (im_gx, im_gy)

    def Smooth(self, inp_img):
        rows = inp_img.shape[0]
        cols = inp_img.shape[1]
        im_gx = np.zeros((rows, cols, 1), np.uint8)
        kernal = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
        for i in range(2, rows - 2, 1):
            for j in range(2, cols - 2, 1):
                sum = 0
                for k in range(-2, 3, 1):
                    for l in range(-2, 3, 1):
                        sum = sum + (inp_img[i + k, l + j, 0] * kernal[k + 2][l + 2])
                sum = sum / 15
                im_gx[i, j, 0] = sum
        return (im_gx)

    def Kirsch(self, inp_img):
        rows = inp_img.shape[0]
        cols = inp_img.shape[1]
        im_gN = np.zeros((rows, cols, 1), np.uint8)
        im_gNW = np.zeros((rows, cols, 1), np.uint8)
        im_gW = np.zeros((rows, cols, 1), np.uint8)
        im_gSW = np.zeros((rows, cols, 1), np.uint8)
        im_gS = np.zeros((rows, cols, 1), np.uint8)
        im_gSE = np.zeros((rows, cols, 1), np.uint8)
        im_gE = np.zeros((rows, cols, 1), np.uint8)
        im_gNE = np.zeros((rows, cols, 1), np.uint8)
        gN = [[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]
        gNW = [[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]
        gW = [[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]
        gSW = [[5, 5, -3], [5, 0, -3], [-3, -3, -3]]
        gS = [[5, -3, -3], [5, 0, -3], [5, -3, -3]]
        gSE = [[-3, -3, -3], [5, 0, -3], [5, 5, -3]]
        gE = [[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]
        gNE = [[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]
        for i in range(1, rows - 1, 1):
            for j in range(1, cols - 1, 1):
                sumN = sumNW = sumW= sumSW = sumS = sumSE = sumE = sumNE = 0
                for k in range(-1, 2, 1):
                    for l in range(-1, 2, 1):
                        sumN = sumN + (inp_img[i + k, l + j, 0] * gN[k + 1][l + 1])
                        sumNW = sumNW + (inp_img[i + k, l + j, 0] * gNW[k + 1][l + 1])
                        sumW = sumW + (inp_img[i + k, l + j, 0] * gW[k + 1][l + 1])
                        sumSW = sumSW + (inp_img[i + k, l + j, 0] * gSW[k + 1][l + 1])
                        sumS = sumS + (inp_img[i + k, l + j, 0] * gS[k + 1][l + 1])
                        sumSE = sumSE + (inp_img[i + k, l + j, 0] * gSE[k + 1][l + 1])
                        sumE = sumE + (inp_img[i + k, l + j, 0] * gE[k + 1][l + 1])
                        sumNE = sumNE + (inp_img[i + k, l + j, 0] * gNE[k + 1][l + 1])
                    if sumN>=0:
                        im_gN[i, j, 0] = abs(sumN)
                    if sumNW >= 0:
                        im_gNW[i, j, 0] = abs(sumNW)
                    if sumW >= 0:
                        im_gW[i, j, 0] = abs(sumW)
                    if sumSW >= 0:
                        im_gSW[i, j, 0] = abs(sumSW)
                    if sumS >= 0:
                        im_gS[i, j, 0] = abs(sumS)
                    if sumSE >= 0:
                        im_gSE[i, j, 0] = abs(sumSE)
                    if sumE >= 0:
                        im_gE[i, j, 0] = abs(sumE)
                    if sumNE >= 0:
                        im_gNE[i, j, 0] = abs(sumNE)
        return (im_gN, im_gNW, im_gW, im_gSW, im_gS, im_gSE, im_gE, im_gNE)

    def Magnitude(self, img1,img2):
        rows = img1.shape[0]
        cols = img1.shape[1]
        Magnitude = np.zeros((rows, cols, 1), np.uint8)
        for i in range(0, rows, 1):
            for j in range(0, cols, 1):
                if img1[i,j,0]>img2[i,j,0]:
                    Magnitude[i,j,0] = img1[i,j,0]
                else:
                    Magnitude[i, j, 0] = img2[i, j, 0]
        return Magnitude
    def Kirsch_Magnitude(self, im_gN, im_gNW, im_gW, im_gSW, im_gS, im_gSE, im_gE, im_gNE):
        rows = im_gN.shape[0]
        cols = im_gN.shape[1]
        Magnitude = np.zeros((rows, cols, 1), np.uint8)
        for i in range(0, rows, 1):
            for j in range(0, cols, 1):
                max1 = max(im_gN[i, j, 0], im_gNW[i, j, 0], im_gW[i, j, 0], im_gSW[i, j, 0], im_gS[i, j, 0], im_gSE[i, j, 0], im_gE[i, j, 0], im_gNE[i, j, 0])
                Magnitude[i, j, 0] = max1
        return Magnitude
    def Thereshold(self, img):
        rows = img.shape[0]
        cols = img.shape[1]
        Threshold = np.zeros((rows, cols, 1), np.uint8)
        for i in range(rows):
            for j in range(cols):
                if img[i,j,0] >120:
                    Threshold[i,j,0] = 255
        return Threshold



def main():
    inp_img1 = cv2.imread("images/building.png")
    cv2.imshow("input image_1", inp_img1)
    object1 = Edge_Detection()
    (x,y) = object1.Robert_Cross_Gradient(inp_img1)
    angle =  object1.angle(x,y)
    x = object1.Absolute(x)
    y = object1.Absolute(y)
    Robert_magnitude = object1.Magnitude(x,y)
    cv2.imshow("Robert_x",x)
    cv2.imshow("Robert_y", y)
    cv2.imshow("Robert_Magnitude", Robert_magnitude)
    cv2.imshow("Robert_angle", angle)
    cv2.waitKey(0)
    (x, y) = object1.Prewitt(inp_img1)
    Prewitt_magnitude = object1.Magnitude(x, y)
    cv2.imshow("Prewitt_x", x)
    cv2.imshow("Prewitt_y", y)
    cv2.imshow("Prewitt_Magnitude", Prewitt_magnitude)
    cv2.waitKey(0)
    (x,y) = object1.Sobel(inp_img1)
    Sobel_magnitude = object1.Magnitude(x, y)
    Threshold = object1.Thereshold(Sobel_magnitude)
    cv2.imshow("Sobel_x",x)
    cv2.imshow("Sobel_y", y)
    cv2.imshow("Sobel_Magnitude", Sobel_magnitude)
    cv2.imshow("Thresold", Threshold)
    cv2.waitKey(0)
    smooth = object1.Smooth(inp_img1)
    (sx,sy) = object1.Sobel(smooth)
    Magnitude = object1.Magnitude(sx,sy)
    Threshold = object1.Thereshold(Magnitude)
    cv2.imshow("SmoothSobel_x", sx)
    cv2.imshow("SmoothSobel_y", sy)
    cv2.imshow("SmoothSobel_Magnitude", Magnitude)
    cv2.imshow("SmoothSobel_Threshold", Threshold)
    cv2.waitKey(0)
    (im_gN, im_gNW, im_gW, im_gSW, im_gS, im_gSE, im_gE, im_gNE) = object1.Kirsch(inp_img1)
    Kirch_Magnitude = object1.Kirsch_Magnitude(im_gN, im_gNW, im_gW, im_gSW, im_gS, im_gSE, im_gE, im_gNE)
    cv2.imshow("Kirsch_North", im_gN)
    cv2.imshow("Kirsch_NorthWest", im_gNW)
    cv2.imshow("Kirsch_West", im_gW)
    cv2.imshow("Kirsch_SouthWest", im_gSW)
    cv2.imshow("Kirsch_South", im_gS)
    cv2.imshow("Kirsch_SouthEast", im_gSE)
    cv2.imshow("Kirsch_East", im_gE)
    cv2.imshow("Kirsch_NorthEast", im_gNE)
    cv2.imshow("Kirsch_Magnitude", Kirch_Magnitude)
    cv2.waitKey(0)

main()