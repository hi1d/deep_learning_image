from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2 as cv 
import re

facenet = cv.dnn.readNet('WEEK_3/models/deploy.prototxt', 'WEEK_3/models/res10_300x300_ssd_iter_140000.caffemodel')

gender_list = ['Male', 'Female']
age_list = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']

gender_net = cv.dnn.readNetFromCaffe('WEEK_3/models/deploy_gender.prototxt', 'WEEK_3/models/gender_net.caffemodel')
age_net = cv.dnn.readNetFromCaffe('WEEK_3/models/deploy_age.prototxt', 'WEEK_3/models/age_net.caffemodel')
cap = cv.VideoCapture(0)

while True:
    ret, img = cap.read()

    if ret == False:
        break

    h, w, c = img.shape
    # 이미지 전처리하기
    blob = cv.dnn.blobFromImage(img, size=(300, 300), mean=(104., 177., 123.))

    # 얼굴 영역 탐지 모델로 추론하기
    facenet.setInput(blob)
    dets = facenet.forward()
    # 각 얼굴에 대해서 반복문 돌기
    for i in range(dets.shape[2]):
        confidence = dets[0, 0, i, 2]

        if confidence < 0.5:
            continue

        # 사각형 꼭지점 찾기
        x1 = int(dets[0, 0, i, 3] * w)
        y1 = int(dets[0, 0, i, 4] * h)
        x2 = int(dets[0, 0, i, 5] * w)
        y2 = int(dets[0, 0, i, 6] * h)

        face = img[y1:y2, x1:x2]

        # 전처리
        blob = cv.dnn.blobFromImage(face, size=(227,227), mean=(78.4263377603, 87.7689143744, 114.895847746))

        gender_net.setInput(blob)
        gender_index = gender_net.forward().squeeze().argmax()
        gender = gender_list[gender_index]

        age_net.setInput(blob)
        age_index = age_net.forward().squeeze().argmax()
        age = age_list[age_index]
        age = re.sub('[^0-9,]','',age).split(',')
        


        cv.putText(img, text=f'{gender}/ {age[0]}~{age[1]}', org=(x1, y1), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0), thickness=2)
        cv.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0), thickness=2)
        

    cv.imshow('result', img)

    if cv.waitKey(1) == ord('q'):
        break