import cv2
pic1 = cv2.imread("images/try.jpg")
pic2 = cv2.imread("images/cameraman.jpg")
corner = pic1[100:325,100:325]
cv2.imshow("Image1",corner)
cv2.imshow("Image2",pic2)
result = pic2
for x in range(0,225):
    for y in range(0, 225):
        (b1,g1,r1) = pic2[x,y]
        (b2, g2, r2) = corner[x, y]
        result[x,y] = (b1*b2,g1*g2,r1*r2)
#result=pic2*corner
cv2.imshow("Result",result)
cv2.imwrite("images/image multiplication.jpg",result)
cv2.waitKey(0)