from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2 as cv 

facenet = cv.dnn.readNet('WEEK_3/models/deploy.prototxt', 'WEEK_3/models/res10_300x300_ssd_iter_140000.caffemodel')
model = load_model('WEEK_3/models/mask_detector.model')

cap = cv.VideoCapture('WEEK_3/media/03.mp4')
# cap = cv.VideoCapture(0)

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

        #  전처리
        face = img[y1:y2, x1:x2]

        face_input = cv.resize(face, dsize=(224, 224))
        face_input = cv.cvtColor(face_input, cv.COLOR_BGR2RGB)
        face_input = preprocess_input(face_input)
        face_input = np.expand_dims(face_input, axis=0)

        # 추론
        mask, nomask = model.predict(face_input).squeeze()

        if mask > nomask:
            color = (0, 255, 0)

        else:
            color = (0, 0, 255)
        # 사각형 그리기
        cv.putText(img, text=f'{round(mask*100,2)}%', org=(x1, y1), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0,255,0), thickness=2)
        cv.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color)
	
    cv.imshow('result', img)

    if cv.waitKey(1) == ord('q'):
        break