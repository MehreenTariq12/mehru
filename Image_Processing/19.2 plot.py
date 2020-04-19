import cv2
import matplotlib.pyplot as plt
import numpy as np
img6 = cv2.imread("images/green.jpg")
in_array=[]
out_array=[]
out_array2=[]
out_array3=[]
img=img6[100:150,100:150]
x=np.arange(0,255)
y2=16*(x**(1/2))
y3=255*(x/255)**(2)
y1=(np.log(x+1))*46
y4=(np.exp(x/46))
y5=255-x
plt.plot(x,x, label="identity")
plt.plot(x,y1, label="log")
plt.plot(x,y2, label="nth root")
plt.plot(x,y3, label="nth power")
plt.plot(x,y4,label="anti-log")
plt.plot(x,y5,label="negative")
plt.title("Transformations")
plt.xlabel("input-intesity")
plt.ylabel("output-intensity")
plt.legend()
plt.show()
cv2.waitKey(0)