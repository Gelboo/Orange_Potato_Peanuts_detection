import  cv2
import numpy as np
import time
cap = cv2.VideoCapture(0)
ret = True
c = 0
inter = 0
while True:
    try:
        ret,frame = cap.read()
        # print(ret)
        if c%10 == 0:
            cv2.imshow('img',frame)
            inter += 1
        c+=1
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    except:
        print("finished || NotExisst")
        print(c)
        break
print(c)
print(inter)
cap.release()
cv2.destroyAllWindows()
