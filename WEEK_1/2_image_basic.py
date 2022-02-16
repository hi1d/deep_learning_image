from turtle import color
import cv2 as cv

img = cv.imread('WEEK_1/01.jpg')

# 이미지는 숫자의 형태 
print(img)

# (세로 pix, 가로 pix , 채널 (BGR))
print(img.shape)

# 사각형 도형 표시하기 
cv.rectangle(img, pt1=(259, 89), pt2 = (380, 348), color=(255,0,0), thickness=2) # x,y순서

# 원 도형 표시하기
cv.circle(img, center=(320, 220), radius=100, color=(0, 0, 255), thickness=3)

# 이미지 자르기
cropped_img = img[89:348, 259:380] # y, x 순서

# 이미지 크기 변경하기
img_resized = cv.resize(img, (512, 256))

# 이미지 컬러 변경하기
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# 이미지 미리보기
def img_basic():
    cv.imshow('img', img)
    cv.imshow('crop', cropped_img)
    cv.imshow('resized', img_resized)
    cv.imshow('rgb', img_rgb)
    # 키가 눌리면 종료
    cv.waitKey(0)

img_basic()