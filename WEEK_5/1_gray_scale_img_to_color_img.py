import cv2 as cv
import numpy as np

proto = "WEEK_5/models/colorization_deploy_v2.prototxt"
weights = "WEEK_5/models/colorization_release_v2.caffemodel"

net = cv.dnn.readNetFromCaffe(proto, weights)

pts_in_hull = np.load('WEEK_5/models/pts_in_hull.npy')
pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1).astype(np.float32)
net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull]

net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full((1, 313), 2.606, np.float32)]

img = cv.imread('WEEK_5/media/m.jpeg')


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

# img[0:793, 0:320] = output_bgr[0:793, 0:320]

# mask 작업
mask = np.zeros_like(img, dtype='uint8')
text = np.zeros_like(img, dtype='uint8')
mask = cv.circle(mask, center=(260, 260), radius=200, color=(1,1,1), thickness=-1)
text = cv.putText(text, text='Hello_gray_to_color_img', org=(0, 260), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(1,1,1), thickness=10)


color = output_bgr * mask
gray = img * (1-mask)

text_color = output_bgr * text
text_gray = img * (1-text)

output2 = color + gray
text_output = text_color + text_gray

cv.imshow('result2',output2)
cv.imshow('text', text_output)
cv.imshow('result', img)
cv.waitKey(0)