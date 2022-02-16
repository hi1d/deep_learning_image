import cv2 as cv

cap = cv.VideoCapture('WEEK_1/03.mp4')

while True:

    ret, img = cap.read()

    if ret == False:
        break

    croped_img = img[183:465, 721:878]
    croped_img = cv.cvtColor(croped_img, cv.COLOR_BGR2GRAY)

    cv.imshow('crop', croped_img)
    cv.imshow('result', img)

    if cv.waitKey(50) == ord('q'):
        break