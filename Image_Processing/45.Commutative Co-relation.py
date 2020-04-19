import cv2
import numpy as np
Image=[[0,0,0,0,0], [0,4,3,2,0], [0,2,3,4,0], [0,4,3,2,0], [0,0,0,0,0]]
Kernal = [[1,1,1], [1,1,1], [1,1,1]]
Result=[]
for i in range(1,len(Image)-1):
    Result.insert(i, [])
    for j in range(1,len(list(zip(*Image)))-1):
        sum=0
        for x in range(-1,2,1):
            for y in range(-1,2,1):
                sum = sum + (Image[i+x][j+y] * Kernal[1+x][1+y])
                print(sum)
                print(i,j)
                print(Result)
        sum = sum / 25
        Result[i].append(j, sum)
print(Result)
cv2.waitKey(0)
