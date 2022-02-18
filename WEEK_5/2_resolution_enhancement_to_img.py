import cv2 as cv



result[:, 0:100] = resized_img[:, 0:100] 


cv.imshow('img', result)
cv.waitKey()