import cv2
import numpy as np

def getContours(img):
    kernel = np.ones((5,5),np.uint8)
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(imgray,-1,kernel)
    _,thresINv = cv2.threshold(dst,88,255,cv2.THRESH_BINARY_INV)
    closing = cv2.morphologyEx(thresINv, cv2.MORPH_CLOSE, kernel)
    im2, contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0,255,0), 3)
    return img,closing,thresINv


# img = cv2.imread('Training/potato/Potato_72.jpg')
# img = np.array([np.array(cv2.resize(cv2.imread('Training/potato/Potato_72.jpg'),(100,100)))])
# img,closing,threshold = getContours(img)
# cv2.imshow('img',img)
# cv2.imshow('threshold',threshold)
# cv2.imshow('closing',closing)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
