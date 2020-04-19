import cv2
import numpy as np
import math

class Otsu_Threshold:
    def Histogram(self,img):
        rows = img.shape[0]
        cols = img.shape[1]
        total = rows*cols
        r = []
        pr = []
        for i in range(256):
            r.append(0)
            pr.append(0)
        for i in range(rows):
            for j in range(cols):
                intensity = img[i, j, 0]
                r[intensity] += 1
        for i in range(256):
            pr[i] = r[i] / total
        return pr
    def cumulative_sum(self, pr):
        cum_sum=[]
        for i in range(256):
            cum_sum.append(0)
        for i in range(256):
            sum = 0
            for j in range(i):
                sum = sum + pr[j]
            cum_sum[i] = sum
        return cum_sum
    def cumulative_mean(self, pr):
        cum_mean=[]
        for i in range(256):
            cum_mean.append(0)
        for i in range(256):
            sum = 0
            for j in range(i):
                sum = sum + (pr[j]*j)
            cum_mean[i] = sum
            #print(cum_mean)
        return cum_mean
    def global_mean(self,pr):
        global_mean = 0
        for i in range(256):
            global_mean = global_mean + (i * pr[i])
        return global_mean
    def between_class_variance(self,global_mean,cum_mean,cum_sum):
        sigma = []
        for i in range(256):
            sigma.append(0)
        for i in range(256):
            a = (global_mean*cum_sum[i])
            c = (a - (cum_mean[i]))**2
            b = (cum_sum[i]-(cum_sum[i])**2)
            if b == 0:
                sigma[i] = 0
            else:
                sigma[i] = (c/b)
        max = sigma[0]
        loc = [0]
        for i in range(1,256):
            if sigma[i] > max:
                max = sigma[i]
                loc = [i]
            elif sigma[i] == max:
                loc.append(i)
        length = len(loc)
        kstar = 0
        for i in range(length):
            kstar = kstar + loc[i]
        kstar = kstar/length
        return (kstar,sigma)
    def sefrability_measre(self,BCV,GM):
        n = BCV/(GM)**2
        return n
    def Threshold(self,img,T,rows,cols):
        new_img = np.zeros((rows,cols,1),np.uint8)
        for i in range(rows):
            for j in range(cols):
                if img[i,j,0] <= T:
                    new_img[i,j,0] = 0
                else:
                    new_img[i, j, 0] = 255
        return new_img

    def Smooth(self,img):
        rows = img.shape[0]
        cols = img.shape[1]
        newimg1 = np.zeros((rows, cols, 1), np.uint8)
        for i in range(rows):
            for j in range(cols):
                newimg1[i, j, 0] = img[i, j, 0]
        img1 = np.zeros((rows, cols, 1), np.uint8)
        w = [[1, 1, 1,1,1], [1, 1, 1,1,1], [1, 1, 1,1,1], [1, 1, 1,1,1], [1, 1, 1,1,1]]
        for i in range(2, rows - 2):
            for j in range(2, cols - 2):
                sum = 0
                for x in range(-2, 3, 1):
                    for y in range(-2, 3, 1):
                        b = newimg1[i + x, j + y, 0]
                        sum = sum + (b * w[1 + x][1 + y])
                sum = sum / 25
                img1[i, j, 0] = sum
        return img1
    def one_channel(self,img):
        rows = img.shape[0]
        cols = img.shape[1]
        newimg1 = np.zeros((rows, cols, 1), np.uint8)
        for i in range(rows):
            for j in range(cols):
                newimg1[i,j,0] = img[i,j,0]
        return newimg1

    def Laplacian1(self,img):
        color_image1 = img
        rows1 = color_image1.shape[0]
        cols1 = color_image1.shape[1]
        gray_image1 = np.zeros((rows1, cols1, 1), np.uint8)
        for i in range(rows1):
            for j in range(cols1):
                gray_image1[i, j] = color_image1[i, j, 0]

        img = gray_image1

        img1 = np.zeros((rows1, cols1, 1), np.uint8)
        w = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
        new_kernal = []
        # Inverting the kernal
        for i in range(len(w)):
            new_kernal.insert(i, [])
            for j in range(len(list(zip(*w)))):
                new_kernal[i].insert(j, w[2 - i][2 - j])
        for i in range(1, rows1 - 1):
            for j in range(1, cols1 - 1):
                bsum = 0
                for x in range(-1, 2, 1):
                    for y in range(-1, 2, 1):
                        b = img[i + x, j + y, 0]
                        bsum = bsum + (b * new_kernal[1 + x][1 + y])
                img1[i, j] = abs(bsum)
        new = np.zeros((rows1, cols1, 1), np.uint8)
        for i in range(rows1):
            for j in range(cols1):
                if img1[i,j,0] >22:
                    new[i,j,0] = 255

        return new
    def thresholded_histogram(self,thresh_img,original_img):
        rows = original_img.shape[0]
        cols = original_img.shape[1]
        total = rows * cols
        r = []
        pr = []
        for i in range(256):
            r.append(0)
            pr.append(0)
        for i in range(rows):
            for j in range(cols):
                if thresh_img[i,j,0] > 0:
                    intensity = original_img[i, j, 0]
                    r[intensity] += 1
        for i in range(256):
            pr[i] = r[i] / total
        return pr




def main():
    object1 = Otsu_Threshold()
    #Otsu on thumb
    inp_image = cv2.imread("images/thumb.png")
    inp_image = object1.one_channel(inp_image)
    thresh_img = main2(inp_image)
    numpy_horizontal = np.hstack((inp_image,thresh_img))
    cv2.imshow('Original OtsuThreshold', numpy_horizontal)

    #Otsu on noisy image
    inp_image = cv2.imread("images/noisythresh.png")
    inp_image = object1.one_channel(inp_image)
    thresh_img = main2(inp_image)

    # Otsu on noisy image with smoothing
    inp_image2 = cv2.imread("images/noisythresh.png")
    inp_image2 = object1.Smooth(inp_image2)
    thresh_img2 = main2(inp_image2)
    numpy_horizontal = np.hstack((inp_image, thresh_img, thresh_img2))
    cv2.imshow('Original ----- OtsuThresholdWithoutSmoothing ----- OtsuThresholdWithSmoothing', numpy_horizontal)

    inp_image3_1 = cv2.imread("images/bgr.png")
    inp_image3_1 = object1.one_channel(inp_image3_1)
    thresh_img3_1 = main2(inp_image3_1)

    inp_image3 = cv2.imread("images/bgr.png")
    inp_image3 = object1.one_channel(inp_image3)
    (Thresh,thresh_img3) = Edge_Improve_Global_Threshold(inp_image3)
    numpy_horizontal = np.hstack((inp_image3, Thresh, thresh_img3, thresh_img3_1))
    cv2.imshow('Original ----- Edge ----- OtsuafterEdge-----simpleOtsu', numpy_horizontal)


    cv2.waitKey(0)

def main2(inp_image):
    object1 = Otsu_Threshold()
    rows = inp_image.shape[0]
    cols = inp_image.shape[1]
    #inp_image = object1.Smooth(inp_image)
    pr = object1.Histogram(inp_image)
    cum_sum = object1.cumulative_sum(pr)
    cum_mean = object1.cumulative_mean(pr)
    global_mean = object1.global_mean(pr)
    (kstar, sigma) = object1.between_class_variance(global_mean, cum_mean, cum_sum)
    n = object1.sefrability_measre(sigma[int(kstar)], global_mean)
    thresholded_img = object1.Threshold(inp_image, kstar, rows, cols)
    return thresholded_img
def Edge_Improve_Global_Threshold(inp_image):
    object1 = Otsu_Threshold()
    rows = inp_image.shape[0]
    cols = inp_image.shape[1]
    threholded_img = object1.Laplacian1(inp_image)
    pr = object1.thresholded_histogram(threholded_img,inp_image)
    cum_sum = object1.cumulative_sum(pr)
    cum_mean = object1.cumulative_mean(pr)
    global_mean = object1.global_mean(pr)
    (kstar, sigma) = object1.between_class_variance(global_mean, cum_mean, cum_sum)
    n = object1.sefrability_measre(sigma[int(kstar)], global_mean)
    thresholded_img2 = object1.Threshold(inp_image, n, rows, cols)
    return (threholded_img,thresholded_img2)

main()