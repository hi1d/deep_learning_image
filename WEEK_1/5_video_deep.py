from turtle import color
import cv2 as cv

# 동영상 불러오기 ( 동영상은 이미지의 연속 )
# cap = cv.VideoCapture('WEEK_1/04.mp4')

# 웹캠 사용하기
cap = cv.VideoCapture(0)

# 이미지 반복으로 동영상처럼 재생하기
while True:
    ret, img = cap.read()

    if ret == False:
        break
    
    # 사각형 표시
    cv.rectangle(img, pt1=(721, 183), pt2=(878, 465), color=(255, 0, 0), thickness=2)
    
    # 컬러 변경
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 사이즈 변경
    img = cv.resize(img, dsize=(640, 360))

    # 이미지 자르기
    # img = img[100:200, 150:250]
    
    cv.imshow('result', img)

    

    if cv.waitKey(50) == ord('q'):
        break