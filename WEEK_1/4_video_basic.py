import cv2 as cv

# 동영상 불러오기 ( 동영상은 이미지의 연속 )
cap = cv.VideoCapture('WEEK_1/04.mp4')

# 이미지 반복으로 동영상처럼 재생하기
while True:
    ret, img = cap.read()

    if ret == False:
        break

    cv.imshow('result', img)

    if cv.waitKey(1) == ord('q'):
        break