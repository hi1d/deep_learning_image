import cv2 as cv

img = cv.imread('WEEK_1/01.jpg')
# png파일을 오버레이를 입힐때는 IMREAD_UNCHANGED 사용
overlay_img = cv.imread('WEEK_1/dices.png' , cv.IMREAD_UNCHANGED)
# 이미지 크기 변환
overlay_img = cv.resize(overlay_img, dsize=(150, 150))


# 투명도 Aplha 구하기 
overlay_alpha = overlay_img[:, :, 3:] / 255.0
background_alpha = 1.0 - overlay_alpha

# 이미지 합성하기
x1 = 100
y1 = 100
x2 = x1 + 150
y2 = y1 + 150

print(img.shape)
print(overlay_img.shape)

img[y1:y2, x1:x2] = overlay_alpha * overlay_img[:, :, :3] + background_alpha * img[y1:y2, x1:x2]

cv.imshow('img',img)
cv.waitKey(0)