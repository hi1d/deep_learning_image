import cv2 as cv
import numpy as np

net = cv.dnn.readNetFromTorch('WEEK_2/models/instance_norm/mosaic.t7')

img = cv.imread('WEEK_2/img/02.jpeg')

# 전처리 
h, w, c = img.shape

img = cv.resize(img, dsize=(500, int(h/w * 500)))

img = img[162:513, 185:428]

MEAN_VALUE = [103.933, 116.779, 123.680]
blob = cv.dnn.blobFromImage(img, mean=MEAN_VALUE)

# 후처리 
net.setInput(blob)
output = net.forward()

output = output.squeeze().transpose((1,2,0))
output += MEAN_VALUE

output = np.clip(output, 0, 255)
output = output.astype('uint8')


cv.imshow('output', output)
cv.imshow('img', img)
cv.waitKey(0)
