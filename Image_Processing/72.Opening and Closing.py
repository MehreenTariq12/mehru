import cv2
import numpy as np
class morphology:
    def Erosion(self,img):
        rows = img.shape[0]
        cols = img.shape[1]
        img1 = np.zeros((rows,cols,1),np.uint8)
        SE = [[1, 1, 1,1,1,1,1], [1, 1, 1,1,1,1,1],[1, 1, 1,1,1,1,1],[1, 1, 1,1,1,1,1],[1, 1, 1,1,1,1,1],[1, 1, 1,1,1,1,1],[1, 1, 1,1,1,1,1]]
        m1 = []
        for i in range(3, rows - 3):
            for j in range(3, cols - 3):
                pix = img[i, j, 0]
                m1 = []
                for m in range(-3, 4, 1):
                    for n in range(-3, 4, 1):
                        m1.append(img[i + m, j + n, 0] * SE[1 + m][1 + n])
                img1[i, j, 0] = min(m1)
        return img1

    def Dilution(self,img):
        rows = img.shape[0]
        cols = img.shape[1]
        img1 = np.zeros((rows, cols, 1), np.uint8)
        SE = [[1, 1, 1, 1, 1,1,1], [ 1, 1, 1,1,1,1,1],[1, 1, 1,1,1,1,1],[1, 1, 1,1,1,1,1],[1, 1, 1,1,1,1,1],[1, 1, 1,1,1,1,1],[1, 1, 1,1,1,1,1]]
        m1 = []
        for i in range(3, rows - 3):
            for j in range(4, cols - 4):
                pix = img[i, j, 0]
                m1 = []
                for m in range(-3, 4, 1):
                    for n in range(-3, 4, 1):
                        m1.append(img[i + m, j + n, 0] * SE[1 + m][1 + n])
                img1[i, j, 0] = max(m1)
        return img1
    def Complement(self,img1,img2):
        img3 = img1 - img2
        return img3
def main():
    image = cv2.imread("images/morphology.png")
    cv2.imshow("Original",image)
    object1 = morphology()
    eroded = object1.Erosion(image)
    open = object1.Dilution(eroded)
    cv2.imshow("Eosion",eroded)
    cv2.imshow("Opening",open)
    diluted = object1.Dilution(image)
    close = object1.Erosion(diluted)
    cv2.imshow("Dilution", diluted)
    cv2.imshow("Closing", close)
    cv2.waitKey(0)
main()