import cv2
import numpy as np
class Edge_Detection:
    def Laplacian(self,inp_img):
        rows = inp_img.shape[0]
        cols = inp_img.shape[1]
        Laplace_image = np.zeros((rows, cols, 1), np.float)
        kernal = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
        for i in range(1, rows - 1, 1):
            for j in range(1, cols - 1, 1):
                sum = 0
                for k in range(-1, 2, 1):
                    for l in range(-1, 2, 1):
                        sum = sum + (inp_img[i + k, l + j, 0] * kernal[k + 1][l + 1])
                Laplace_image[i, j, 0] = sum
        return Laplace_image
    def normalize_Laplacian(self,img):
        rows = img.shape[0]
        cols = img.shape[1]
        Laplace_image = img
        Laplace_image_int = np.zeros((rows, cols, 1), np.uint8)
        min = np.min(Laplace_image)
        max = np.max(Laplace_image)
        for i in range(0, rows, 1):
            for j in range(0, cols, 1):
                a = Laplace_image[i, j, 0]
                b = (a - min) / (max - min)
                Laplace_image_int[i, j, 0] = b * 255
        return Laplace_image_int
    def Find_Point(self,img,threshold):
        rows = img.shape[0]
        cols = img.shape[1]
        for i in range(0, rows, 1):
            for j in range(0, cols, 1):
                if img[i, j, 0] > threshold:
                    img[i, j, 0] = 255
                else:
                    img[i, j, 0] = 0
        return img
    def Absolute_Laplacian(self,img):
        rows = img.shape[0]
        cols = img.shape[1]
        Laplace_image_abs = np.zeros((rows, cols, 1), np.uint8)
        for i in range(rows):
            for j in range(cols):
                Laplace_image_abs[i,j,0] = np.abs(img[i,j,0])
        return Laplace_image_abs
    def Positive_Laplacian(self,img):
        rows = img.shape[0]
        cols = img.shape[1]
        Laplace_image_positive = np.zeros((rows, cols, 1), np.uint8)
        for i in range(rows):
            for j in range(cols):
                if img[i,j,0]>0:
                    Laplace_image_positive[i,j,0] = img[i,j,0]
        return Laplace_image_positive
    def Horizental_Lap_Kernal(self,inp_img):
        rows = inp_img.shape[0]
        cols = inp_img.shape[1]
        Laplace_image = np.zeros((rows, cols, 1), np.float)
        kernal = [[-1, -1, -1], [2, 2, 2], [-1, -1, -1]]
        for i in range(1, rows - 1, 1):
            for j in range(1, cols - 1, 1):
                sum = 0
                for k in range(-1, 2, 1):
                    for l in range(-1, 2, 1):
                        sum = sum + (inp_img[i + k, l + j, 0] * kernal[k + 1][l + 1])
                Laplace_image[i, j, 0] = sum
        return Laplace_image
    def Vertical_Lap_Kernal(self,inp_img):
        rows = inp_img.shape[0]
        cols = inp_img.shape[1]
        Laplace_image = np.zeros((rows, cols, 1), np.float)
        kernal = [[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]]
        for i in range(1, rows - 1, 1):
            for j in range(1, cols - 1, 1):
                sum = 0
                for k in range(-1, 2, 1):
                    for l in range(-1, 2, 1):
                        sum = sum + (inp_img[i + k, l + j, 0] * kernal[k + 1][l + 1])
                Laplace_image[i, j, 0] = sum
        return Laplace_image
    def Positive_diagonal_Lap_Kernal(self,inp_img):
        rows = inp_img.shape[0]
        cols = inp_img.shape[1]
        Laplace_image = np.zeros((rows, cols, 1), np.float)
        kernal = [[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]
        for i in range(1, rows - 1, 1):
            for j in range(1, cols - 1, 1):
                sum = 0
                for k in range(-1, 2, 1):
                    for l in range(-1, 2, 1):
                        sum = sum + (inp_img[i + k, l + j, 0] * kernal[k + 1][l + 1])
                Laplace_image[i, j, 0] = sum
        return Laplace_image
    def Negative_diagonal_Lap_Kernal(self,inp_img):
        rows = inp_img.shape[0]
        cols = inp_img.shape[1]
        Laplace_image = np.zeros((rows, cols, 1), np.float)
        kernal = [[-1, -1, 2], [-1, 2, -1], [2, -1, -1]]
        for i in range(1, rows - 1, 1):
            for j in range(1, cols - 1, 1):
                sum = 0
                for k in range(-1, 2, 1):
                    for l in range(-1, 2, 1):
                        sum = sum + (inp_img[i + k, l + j, 0] * kernal[k + 1][l + 1])
                Laplace_image[i, j, 0] = sum
        return Laplace_image


def main():
    inp_img1 = cv2.imread("images/point.png")
    cv2.imshow("input image_1", inp_img1)
    inp_img2 = cv2.imread("images/box.png")
    object1 = Edge_Detection()
    float_Laplace = object1.Laplacian(inp_img1)
    norm_Laplac = object1.normalize_Laplacian(float_Laplace)
    cv2.imshow("Laplacian",norm_Laplac)
    point_img = object1.Find_Point(norm_Laplac,173)
    cv2.imshow("Point_img", point_img)
    cv2.waitKey(0)
    cv2.imshow("input image_2", inp_img2)
    float_Laplace = object1.Laplacian(inp_img2)
    norm_Laplac = object1.normalize_Laplacian(float_Laplace)
    cv2.imshow("Laplacian2", norm_Laplac)
    cv2.waitKey(0)
    abs_img = object1.Absolute_Laplacian(float_Laplace)
    cv2.imshow("abs_img", abs_img)
    cv2.waitKey(0)
    positive_img = object1.Positive_Laplacian(float_Laplace)
    cv2.imshow("Positive_Laplacian", positive_img)
    cv2.waitKey(0)
    float_Laplace_Horizental = object1.Horizental_Lap_Kernal(inp_img2)
    norm_Laplac = object1.normalize_Laplacian(float_Laplace_Horizental)
    cv2.imshow("Horizental_Laplacian", norm_Laplac)
    cv2.waitKey(0)
    float_Laplace_Vertical = object1.Vertical_Lap_Kernal(inp_img2)
    norm_Laplac = object1.normalize_Laplacian(float_Laplace_Vertical)
    cv2.imshow("Vertical_Laplacian", norm_Laplac)
    cv2.waitKey(0)
    float_Laplace_positive_diagonal = object1.Positive_diagonal_Lap_Kernal(inp_img2)
    norm_Laplac = object1.normalize_Laplacian(float_Laplace_positive_diagonal)
    cv2.imshow("positive_diagonal_Laplacian", norm_Laplac)
    cv2.waitKey(0)
    float_Laplace_negative_diagonal = object1.Vertical_Lap_Kernal(inp_img2)
    norm_Laplac = object1.normalize_Laplacian(float_Laplace_negative_diagonal)
    cv2.imshow("negative_diagonal_Laplacian", norm_Laplac)
    cv2.waitKey(0)
    positive_img_hor = object1.Positive_Laplacian(float_Laplace_Horizental)
    cv2.imshow("Positive_Horizental_Laplacian", positive_img_hor)
    positive_img_ver = object1.Positive_Laplacian(float_Laplace_Vertical)
    cv2.imshow("Positive_Vertical_Laplacian", positive_img_ver)
    positive_img_pdiag = object1.Positive_Laplacian(float_Laplace_positive_diagonal)
    cv2.imshow("Positive_+ve diagonal_Laplacian", positive_img_pdiag)
    positive_img_ndiag = object1.Positive_Laplacian(float_Laplace_negative_diagonal)
    cv2.imshow("Positive_-ve diagonal_Laplacian", positive_img_ndiag)
    cv2.waitKey(0)
    hor_threshold = object1.Find_Point(positive_img_hor,170)
    cv2.imshow("Horizental", hor_threshold)
    ver_threshold = object1.Find_Point(positive_img_ver, 170)
    cv2.imshow("Verticall", ver_threshold)
    pdiag_threshold = object1.Find_Point(positive_img_pdiag, 170)
    cv2.imshow("Positive Diagonal", pdiag_threshold)
    ndiag_threshold = object1.Find_Point(positive_img_ndiag, 170)
    cv2.imshow("Negative Diagonal", ndiag_threshold)
    cv2.waitKey(0)
main()