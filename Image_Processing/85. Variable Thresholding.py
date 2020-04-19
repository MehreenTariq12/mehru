import cv2
import numpy as np


class Variable_Threshold:
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
    def global_mean(self,pr):
        global_mean = 0
        for i in range(256):
            global_mean = global_mean + (i * pr[i])
        return global_mean
    def std(self,r):
        pr = []
        std = mean = 0
        for i in range(256):
            pr.append(0)
        for i in range(len(r)):
            pr[r[i]] += 1
        for i in range(256):
            pr[i] /= 9
            mean = mean + (pr[i]*i)
        for i in range(256):
            a = (i - mean) ** 2
            a = a * pr[i]
            std = std + a
        lstd = (std) ** 0.5
        return lstd
    def one_channel(self,img):
        rows = img.shape[0]
        cols = img.shape[1]
        newimg1 = np.zeros((rows, cols, 1), np.uint8)
        for i in range(rows):
            for j in range(cols):
                newimg1[i,j,0] = img[i,j,0]
        return newimg1

    def Variable_Threshold2(self,img):
        rows = img.shape[0]
        cols = img.shape[1]
        Thresholded_img = np.zeros((rows,cols,1),np.uint8)
        for i in range(rows):
            count = sum = 0
            for j in range(20):
                sum = sum + img[i,j,0]
                count += 1
                sum = sum/count
                sum *= 0.5
                if img[i,j,0] > sum:
                    Thresholded_img[i,j,0] = 255
                else:
                    Thresholded_img[i,j,0] = 0
        for i in range(rows):
            for j in range(20,cols):
                sum = 0
                for k in range(-19,1):
                    intensity = img[i,j+k,0]
                    sum = sum + intensity
                sum = sum/13
                sum = sum * 0.55
                if img[i,j,0] > sum:
                    Thresholded_img[i,j,0] = 255
                else:
                    Thresholded_img[i, j, 0] = 0
        return Thresholded_img




def Variable_Threshold1():
    object1 = Variable_Threshold()
    inp_image = cv2.imread("images/bgr.png")
    inp_image = object1.one_channel(inp_image)
    cv2.imshow("original", inp_image)
    rows = inp_image.shape[0]
    cols = inp_image.shape[1]
    std_img = np.zeros((rows,cols,1),np.float)
    Thresh = np.zeros((rows, cols, 1), np.uint8)
    r = []
    pr = object1.Histogram(inp_image)
    mean = object1.global_mean(pr)
    for i in range(9):
        r.append(0)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            for y in range(9):
                r[y] = 0
            count = 0
            for m in range(-1, 2, 1):
                for n in range(-1, 2, 1):
                    intensity = inp_image[i+m,j+n,0]
                    r[count] = intensity
                    count = count+1
            std = object1.std(r)
            std_img[i,j,0] = std
            if inp_image[i,j,0] > (30 * std):
                if inp_image[i,j,0] > (1.5 * mean):
                    Thresh[i,j,0] = 255
            else:
                Thresh[i,j,0] = 0
    cv2.imshow("Variable Threshold1",Thresh)
def Variable_Threshold2():
    object1 = Variable_Threshold()
    img = cv2.imread("images/text.png")
    cv2.imshow("Original",img)
    thresholded = object1.Variable_Threshold2(img)
    cv2.imshow("Variable_Threshold2",thresholded)

def main():
    Variable_Threshold1()
    Variable_Threshold2()
    cv2.waitKey(0)
main()


