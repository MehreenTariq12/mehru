import cv2
pic1 = cv2.imread("images/1.jpg")
pic2 = cv2.imread("images/cameraman.jpg")

cv2.imshow("Image1",pic1)
cv2.imshow("Image2",pic2)
result=pic1

for x in range(0,225):
    for y in range(0, 225):
        b1=pic1[x,y,0]
        av1=(b1*70)/100
        g1 = pic1[x, y, 1]
        av2 = (g1 * 70) / 100
        r1 = pic1[x, y, 2]
        av3 = (r1 * 70) / 100
        b2 = pic2[x, y, 0]
        av21 = (b2 * 30) / 100
        g2 = pic2[x, y, 1]
        av22 = (g2 * 30) / 100
        r2 = pic2[x, y, 2]
        av23 = (r2 * 30) / 100
        result[x,y]=(av1+av21,av2+av22,av3+av23)

#corner=pic2[0:pic1.shape[0],0:pic1.shape[1]]
#result=cv2.addWeighted(pic1,0.7,corner,0.3,0)
cv2.imshow("Result",result)
cv2.imwrite("images/image addition.jpg",result)
cv2.waitKey(0)