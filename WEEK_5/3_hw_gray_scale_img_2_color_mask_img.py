import cv2 as cv
import numpy as np
import sys,os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from clickx_y import MouseGesture


proto = "WEEK_5/models/colorization_deploy_v2.prototxt"
weights = "WEEK_5/models/colorization_release_v2.caffemodel"

net = cv.dnn.readNetFromCaffe(proto, weights)

pts_in_hull = np.load('WEEK_5/models/pts_in_hull.npy')
pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1).astype(np.float32)
net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull]

net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full((1, 313), 2.606, np.float32)]

img = cv.imread('WEEK_5/media/1.jpeg')


# cv.imshow('img', img)
# cv.setMouseCallback('img', MouseGesture().on_mouse, param=img)
# cv.waitKey(0)


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




area = np.zeros_like(img, dtype='uint8')

# mask 작업 rectangle
area1 = cv.rectangle(area, pt1=(226, 101), pt2=(397, 357), color=(1, 1, 1), thickness=-1)

# mask 작업 circle
area2 = cv.circle(area, center=(313, 282), radius=30, color=(0,0,0), thickness=-1)
area3 = cv.circle(area, center=(304,168), radius=40, color=(0,0,0),thickness=-1)

color = output_bgr * area1
gray = img * (1-area1)

result = color + gray


cv.imshow('result', result)
cv.waitKey(0)