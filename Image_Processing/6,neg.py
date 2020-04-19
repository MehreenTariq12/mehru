import cv2
img=cv2.imread("images/try.jpg")
cv2.imshow("original image",img)
#for x in range(0,img.shape[0]):
 #   for y in range(0, img.shape[1]):
  #      result = img
   #     (b1,g1,r1) = img[x,y]
    #    result[x,y] = (255-b1,255-g1,255-r1)
result=255-img
cv2.imshow("Negative image",result)
cv2.imwrite("images/image negative.jpg",result)
cv2.waitKey(0)