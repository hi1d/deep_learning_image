import cv2 as cv
import numpy as np

net = cv.dnn.readNetFromTorch('WEEK_2/models/instance_norm/mosaic.t7')
net2 = cv.dnn.readNetFromTorch('WEEK_2/models/instance_norm/the_scream.t7')
net3 = cv.dnn.readNetFromTorch('WEEK_2/models/instance_norm/candy.t7')
net4 = cv.dnn.readNetFromTorch('WEEK_2/models/instance_norm/udnie.t7')

img = cv.imread('WEEK_2/img/03.jpeg')

# 전처리
h, w, c = img.shape

img = cv.resize(img, dsize=(500, int(h / w * 500)))
MEAN_VALUE = [103.939, 116.779, 123.680]
blob = cv.dnn.blobFromImage(img, mean=MEAN_VALUE)

# 후처리1
net.setInput(blob)
output = net.forward()

output = output.squeeze().transpose((1, 2, 0))

output += MEAN_VALUE
output = np.clip(output, 0, 255)
output = output.astype('uint8')

# 후처리2
net2.setInput(blob)
output2 = net2.forward()

output2 = output2.squeeze().transpose((1, 2, 0))

output2 += MEAN_VALUE
output2 = np.clip(output2, 0, 255)
output2 = output2.astype('uint8')

# 후처리3
net3.setInput(blob)
output3 = net3.forward()

output3 = output3.squeeze().transpose((1, 2, 0))

output3 += MEAN_VALUE
output3 = np.clip(output3, 0, 255)
output3 = output3.astype('uint8')

# 후처리4
net4.setInput(blob)
output4 = net4.forward()

output4 = output4.squeeze().transpose((1, 2, 0))

output4 += MEAN_VALUE
output4 = np.clip(output4, 0, 255)
output4 = output4.astype('uint8')

# 가로축 자르기
# output = output[0:157, :]
# output2 = output2[157:315, :]
# result = np.concatenate([output, output2], axis=0) # axis 1 = x 축

# cv.imshow('img', img)
# cv.imshow('result', result)
# cv.waitKey(0)

# 가로축 가운데 아닌 곳 자르기
# output = output[0:100, :]
# output_blank = img[100:200, :]
# output2 = output2[200:315, :]
# result = np.concatenate([output, output_blank, output2], axis=0) # axis 1 = x 축

# cv.imshow('result',result)
# cv.waitKey(0)

# 가로축 4개로 나누기 

output = output[0:80, :]
output2 = output2[80:160, :]
output3 = output3[160:240, :]
output4 = output4[240:315, :]
result = np.concatenate([output, output2, output3, output4], axis=0)

cv.imshow('result', result)
cv.waitKey(0)