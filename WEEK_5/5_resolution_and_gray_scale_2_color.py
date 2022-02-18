import cv2 as cv
import numpy as np

proto = "WEEK_5/models/colorization_deploy_v2.prototxt"
weights = "WEEK_5/models/colorization_release_v2.caffemodel"

net = cv.dnn.readNetFromCaffe(proto, weights)

pts_in_hull = np.load('WEEK_5/models/pts_in_hull.npy')
pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1).astype(np.float32)
net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull]

net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full((1, 313), 2.606, np.float32)]

origin_img = cv.imread('WEEK_5/media/3.jpg')
sr = cv.dnn_superres.DnnSuperResImpl_create()
sr.readModel('WEEK_5/models/EDSR_x4.pb')
sr.setModel('edsr', 4)

img = sr.upsample(origin_img)


h, w, c = img.shape

img_input = img.copy()

img_input = img_input.astype('float32') / 255.
img_lab = cv.cvtColor(img_input, cv.COLOR_BGR2LAB)
img_l = img_lab[:, :, 0:1]

blob = cv.dnn.blobFromImage(img_l, size=(224, 224), mean=[50, 50, 50])

net.setInput(blob)
output = net.forward()

output = output.squeeze().transpose((1, 2, 0))

output_resized = cv.resize(output, (w,h))

output_lab = np.concatenate([img_l, output_resized], axis=2)

output_bgr = cv.cvtColor(output_lab, cv.COLOR_LAB2BGR)
output_bgr = output_bgr * 255
output_bgr = np.clip(output_bgr, 0, 255)
output_bgr = output_bgr.astype('uint8')

cv.imshow('img',origin_img)
cv.imshow('result', output_bgr)
cv.waitKey(0)