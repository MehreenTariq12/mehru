import cv2
import numpy as np
img1 = cv2.imread("images/local.png")
rows = img1.shape[0]
cols = img1.shape[1]
total = rows *cols
newimg1 = np.zeros((rows,cols,1),np.uint8)
newimg2 = np.zeros((rows,cols,1),np.uint8)
for i in range(rows):
    for j in range(cols):
        newimg1[i,j,0]=img1[i,j,0]
cv2.imshow("original",newimg1)

C = 4
k0=0.01
k1=0.9
k2=0.01
k3=0.9
r=[]
pr=[]
s=[]
ps=[]
for i in range(256):
    r.append(0)
    pr.append(0)
    s.append(0)
    ps.append(0)
for i in range(rows):
    for j in range(cols):
        intensity = newimg1[i,j,0]
        r[intensity] += 1
for i in range(256):
    pr[i] = r[i]/total
#Global Mean
globalmean = 0
for i in range(256):
    globalmean = globalmean + (i * pr[i])
#Global standard deviation
globalstd = 0
for i in range(256):
    a = (i - globalmean)**2
    a =a * pr[i]
    globalstd = globalstd + a
gstd = (globalstd) ** (0.5)
for i in range(1,rows-1):
    for j in range(1,cols - 1):
        mean = 0
        std = 0
        a=0
        for y in range(256):
            s[y] = 0
            ps[y] = 0
        for m in range(-1,2,1):
            for n in range(-1, 2, 1):
                intensity = newimg1[i+m,j+n,0]
                s[intensity] = s[intensity] + 1
        for o in range(256):
            ps[o] = s[o]/9
        for p in range(256):
            mean = mean + (p * ps[p])
        for q in range(256):
            a = (q - mean) ** 2
            a = a * ps[q]
            std = std + a
        lstd = (std) ** 0.5
        d = (k0*globalmean)
        e = (k1*globalmean)
        f = (k2*gstd)
        g = (k3*gstd)
        #print(d,e,f,g)
        if d <= mean and mean <= e and f <= lstd and lstd <= g:
            k = C * newimg1[i,j,0]
            if k>255:
                k=255
            newimg2[i, j, 0] = k
            print(i,j)
        else:
            newimg2[i, j, 0] = newimg1[i, j, 0]

cv2.imshow("local hist stat",newimg2)
cv2.waitKey(0)




