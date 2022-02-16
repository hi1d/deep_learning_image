import cv2 as cv
import numpy as np

# 동영상 불러오기 ( 동영상은 이미지의 연속 )
cap = cv.VideoCapture('WEEK_1/03.mp4')

net = cv.dnn.readNetFromTorch('WEEK_2/models/instance_norm/mosaic.t7')


# 웹캠 사용하기
# cap = cv.VideoCapture(0)

# 이미지 반복으로 동영상처럼 재생하기
while True:
    ret, init_img = cap.read()

    if ret == False:
        break
    img = init_img[0:720, 483:812]
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

    output = cv.resize(output, (w,h))
    init_img[0:720, 483:812] = output
    
    cv.imshow('result', init_img)

    
    if cv.waitKey(100) == ord('q'):
        break