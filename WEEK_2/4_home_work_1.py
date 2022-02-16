import cv2 as cv
import numpy as np
import sys,os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from clickx_y import MouseGesture


net = cv.dnn.readNetFromTorch('WEEK_2/models/instance_norm/mosaic.t7')

init_img = cv.imread('WEEK_2/img/hw.jpeg')


# 이미지 자르기
img = init_img[147:366, 483:812]

# cv.imshow('img', img)
# cv.setMouseCallback('img', MouseGesture().on_mouse, param=img)
# cv.waitKey(0)

# 전처리 
h, w, c = img.shape

img = cv.resize(img, dsize=(500, int(h/w * 500)))

# img = img[147:366, 483:812]

MEAN_VALUE = [103.933, 116.779, 123.680]
blob = cv.dnn.blobFromImage(img, mean=MEAN_VALUE)

# 후처리 
net.setInput(blob)
output = net.forward()

output = output.squeeze().transpose((1,2,0))
output += MEAN_VALUE

output = np.clip(output, 0, 255)
output = output.astype('uint8')


# 자른 이미지 붙이기
output = cv.resize(output, (w,h))
init_img[147:366, 483:812] = output

cv.imshow('output', output)
cv.imshow('img',init_img)
cv.waitKey(0)


