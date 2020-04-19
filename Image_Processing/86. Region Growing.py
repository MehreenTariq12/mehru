import cv2
import numpy as np


class Region_Growing:
    def Threshold(self,img,T1):
        rows = img.shape[0]
        cols = img.shape[1]
        new_img = np.zeros((rows, cols, 1), np.uint8)
        for i in range(rows):
            for j in range(cols):
                if img[i, j, 0] <= T1:
                    new_img[i, j, 0] = 0
                else:
                    new_img[i, j, 0] = 255
        return new_img
    def Erosion(self,img):
        rows = img.shape[0]
        cols = img.shape[1]
        img1 = np.zeros((rows,cols,1),np.uint8)
        SE = [[1,1,1],[1,1,1],[1,1,1]]
        m1 = []
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                pix = img[i, j, 0]
                m1 = []
                for m in range(-1, 2, 1):
                    for n in range(-1, 2, 1):
                        m1.append(img[i + m, j + n, 0] * SE[1 + m][1 + n])
                min1 = min(m1)
                if min1 == 255:
                    for m in range(-1, 2, 1):
                        for n in range(-1, 2, 1):
                            img[i + m, j + n, 0] = 0
                    img[i,j,0] =255
        return img
    def Erosion2(self,img):
        rows = img.shape[0]
        cols = img.shape[1]
        img1 = np.zeros((rows,cols,1),np.uint8)
        SE = [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]
        m1 = []
        for i in range(2, rows - 2):
            for j in range(2, cols - 2):
                pix = img[i, j, 0]
                m1 = []
                for m in range(-2, 3, 1):
                    for n in range(-2, 3, 1):
                        m1.append(img[i + m, j + n, 0] * SE[1 + m][1 + n])
                min1 = min(m1)
                if min1 == 255:
                    for m in range(-2, 3, 1):
                        for n in range(-2, 3, 1):
                            img[i + m, j + n, 0] = 0
                    img[i,j,0] =255
        return img

    def Erosion3(self, img):
        rows = img.shape[0]
        cols = img.shape[1]
        img1 = np.zeros((rows, cols, 1), np.uint8)
        SE = [[1, 1, 1, 1, 1,1,1], [1, 1, 1, 1, 1,1,1],[1, 1, 1, 1, 1,1,1],[1, 1, 1, 1, 1,1,1],[1, 1, 1, 1, 1,1,1],[1, 1, 1, 1, 1,1,1],[1, 1, 1, 1, 1,1,1]]
        m1 = []
        for i in range(3, rows - 3):
            for j in range(3, cols - 3):
                pix = img[i, j, 0]
                m1 = []
                for m in range(-3, 4, 1):
                    for n in range(-3, 4, 1):
                        m1.append(img[i + m, j + n, 0] * SE[1 + m][1 + n])
                min1 = min(m1)
                if min1 == 255:
                    for m in range(-3, 4, 1):
                        for n in range(-3, 4, 1):
                            img[i + m, j + n, 0] = 0
                    img[i, j, 0] = 255
        return img

    def eight_neighb(self,rows,cols,sThreshold,final):
        for i in range(rows):
            for j in range(cols):
                if final[i, j, 0] > 0:
                    for x in range(-1, 2, 1):
                        for y in range(-1, 2, 1):
                            if sThreshold[i + x, j + y, 0] > 0:
                                final[i + x, j + y, 0] = 255
        return final


class Multiple_Otsu_Threshold:
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
        for i in range(254):
            sigma.insert(i, [])
            for j in range(255):
                sigma[i].insert(j, 0)
        for i in range(1,254):
            for j in range(i+1,255):
                p1 = cum_sum[i]
                p2 = (cum_sum[j]-cum_sum[i])
                p3 = (cum_sum[255]-cum_sum[j])
                if p1 != 0:
                    m1 = cum_mean[i]/p1
                else:
                    m1 = 0
                if p2 != 0:
                    m2 = (cum_mean[j]-cum_mean[i])/p2
                else:
                    m2 = 0
                if p3 != 0:
                    m3 = (cum_mean[255]-cum_mean[j])/p3
                else:
                    m3 = 0
                t1 = (p1*(m1-global_mean)**2)
                t2 = (p2 * (m2 - global_mean) ** 2)
                t3 = (p3 * (m3 - global_mean) ** 2)
                sigma[i][j] = t1 + t2 + t3
        max = sigma[0][0]
        k1 = k2 = [0]
        for i in range(254):
            for j in range(i + 1, 255):
                if sigma[i][j] > max:
                    max = sigma[i][j]
                    k1 = [i]
                    k2 = [j]
                elif sigma[i] == max:
                    k1.append(i)
                    k2.append(j)
        k1star = k2star = 0
        length = len(k1)
        for i in range(length):
            k1star = k1star + k1[i]
            k2star = k2star + k2[i]
        k1star = k1star/length
        k2star = k2star / length
        return (k1star,k2star,sigma)
    def sefrability_measre(self,BCV,GM):
        n = BCV/(GM)**2
        return n
    def Threshold(self,img,T1,T2,rows,cols):
        new_img = np.zeros((rows,cols,1),np.uint8)
        for i in range(rows):
            for j in range(cols):
                if img[i,j,0] <= T1:
                    new_img[i,j,0] = 0
                elif T1 < img[i,j,0] <= T2:
                    new_img[i, j, 0] = 120
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


def main():
    object1 = Region_Growing()
    image = cv2.imread("images/regionGrowing.png")
    rows = image.shape[0]
    cols = image.shape[1]
    subtract = np.zeros((rows,cols,1),np.uint8)
    cv2.imshow("Original",image)
    result = object1.Threshold(image,240)
    cv2.imshow("Thresholded", result)
    erosion1 = object1.Erosion3(result)
    #cv2.imshow("erosion1", erosion1)
    erosion2 = object1.Erosion3(erosion1)
    #cv2.imshow("erosion2", erosion2)
    erosion3 = object1.Erosion(erosion2)
    cv2.imshow("erosion3", erosion3)
    for i in range (rows):
        for j in range(cols):
            subtract[i,j,0] = abs(image[i,j,0] - result[i,j,0])
    cv2.imshow("Subtract", subtract)
    (img,k1) = main2(subtract)
    cv2.imshow("Multi Thresholded", img)
    sThreshold = object1.Threshold(image, k1)
    cv2.imshow("Single Thresholded", sThreshold)
    final = erosion3
    final2 = object1.eight_neighb(rows, cols, sThreshold, final)
    while final.any != final2.any:
        final = final2
        final2 = object1.eight_neighb(rows, cols, erosion3, sThreshold, final2)
    cv2.imshow("final",final2)

    cv2.waitKey(0)


def main2(img):
    object1 = Multiple_Otsu_Threshold()
    inp_image = img
    inp_image = object1.one_channel(inp_image)
    #cv2.imshow("original", inp_image)
    pr = object1.Histogram(inp_image)
    rows = inp_image.shape[0]
    cols = inp_image.shape[1]
    cum_sum = object1.cumulative_sum(pr)
    cum_mean = object1.cumulative_mean(pr)
    global_mean = object1.global_mean(pr)
    (k1star, k2star, sigma) = object1.between_class_variance(global_mean, cum_mean, cum_sum)
    n = object1.sefrability_measre(sigma[int(k1star)][int(k2star)], global_mean)
    thresholded_img = object1.Threshold(inp_image, k1star, k2star, rows, cols)
    return (thresholded_img,k2star)
main()