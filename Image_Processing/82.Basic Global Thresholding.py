import cv2
import numpy as np
import math

class Global_Threholding:
    def average(self,img):
        rows = img.shape[0]
        cols = img.shape[1]
        sum = 0
        n = 0
        for i in range(rows):
            for j in range(cols):
                n = n + 1
                sum = sum + img[i,j,0]
        avg = sum / n
        return avg
    def probaility(self, arr):
        rows = len(arr)
        pr = []
        r = []
        for i in range(256):
            pr.append(0)
            r.append(0)
        for i in range(rows):
            intensity = arr[i]
            r[intensity] += 1

        for i in range(256):
            pr[i] = r[i] / rows
        return pr
    def mean(self,pr):
        globalmean = 0
        for i in range(256):
            globalmean = globalmean + (i * pr[i])
        return globalmean
    def Threshold_arrays(self,img,val):
        rows = img.shape[0]
        cols = img.shape[1]
        arr1 = []
        arr2 = []
        for i in range(rows):
            for j in range(cols):
                if img[i,j,0]>val:
                    arr2.append(img[i,j,0])
                else:
                    arr1.append(img[i,j,0])
        return (arr1,arr2)
    def mid_mean(self,m1,m2):
        mid = (m1+m2)/2
        return mid


def main2(inp_image,avg_val):
    object1 = Global_Threholding()
    (arr1,arr2) = object1.Threshold_arrays(inp_image,avg_val)
    pr1 = object1.probaility(arr1)
    pr2 = object1.probaility(arr2)
    mean1 = object1.mean(pr1)
    mean2 = object1.mean(pr2)
    mid = object1.mid_mean(mean1,mean2)
    return mid
def main():
    inp_image = cv2.imread("images/thumb.png")
    cv2.imshow("input image", inp_image)
    rows = inp_image.shape[0]
    cols = inp_image.shape[1]
    Thresholded = np.zeros((rows, cols, 1), np.uint8)
    object1 = Global_Threholding()
    avg_val = object1.average(inp_image)
    mean1 = main2(inp_image,avg_val)
    mean2 = main2(inp_image, mean1)
    while mean1-mean2 != 0:
        mean1 = mean2
        mean2 = main2(inp_image, mean1)
    for i in range(rows):
        for j in range(cols):
            if inp_image[i,j,0] > mean2:
                Thresholded[i,j,0] = 255
            else:
                Thresholded[i, j, 0] = 0
    cv2.imshow("Thresholded",Thresholded)
    cv2.waitKey(0)



main()
