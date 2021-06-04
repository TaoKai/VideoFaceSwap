import cv2
import numpy as np
import os, sys

def show(img, name='', wait=0):
    cv2.imshow(name, img)
    cv2.waitKey(wait)

def sharp(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    img = cv2.filter2D(img, -1, kernel=kernel)
    return img

def gblur(img):
    img = cv2.GaussianBlur(img, (15,15), 0)
    return img

'''
美肤-磨皮算法
Dest =(Src * (100 - Opacity) + (Src + 2 * GuassBlur(EPFFilter(Src) - Src + 128) - 256) * Opacity) /100 ;
'''
def beauty_face(img):
    dst = np.zeros_like(img)
    #int value1 = 3, value2 = 1; 磨皮程度与细节程度的确定
    v1 = 5
    v2 = 1
    dx = v1 * 5 # 双边滤波参数之一
    fc = v1 * 12.5 # 双边滤波参数之一
    p = 0.1
    temp4 = np.zeros_like(img)
    temp1 = cv2.bilateralFilter(img,dx,fc,fc)
    temp2 = cv2.subtract(temp1,img)
    temp2 = cv2.add(temp2,(10,10,10,128))
    temp3 = cv2.GaussianBlur(temp2,(2*v2 - 1,2*v2-1),0)
    temp4 = cv2.add(img,temp3)
    dst = cv2.addWeighted(img,p,temp4,1-p,0.0)
    dst = cv2.add(dst,(10, 10, 10,255))
    return dst

def blurFilter(pic):
    pic = beauty_face(pic)
    pic = gblur(pic)
    pic = sharp(pic)
    return pic

if __name__=='__main__':
    path = 'zhao.jpg'
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    show(img)