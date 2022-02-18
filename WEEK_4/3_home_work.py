import cv2 as cv
import dlib


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('WEEK_4/models/shape_predictor_5_face_landmarks.dat')

# cap = cv.VideoCapture('WEEK_4/media/02.mp4')
cap = cv.VideoCapture(0)
sticker_img = cv.imread('WEEK_4/media/stickers/pig.png', cv.IMREAD_UNCHANGED)


while True:
    ret, img = cap.read()

    if ret == False:
        break

    dets = detector(img)


    for det in dets:
        shape = predictor(img, det)
        x1 = det.left()
        y1 = det.top()
        x2 = det.right() 
        y2 = det.bottom()

        # for i, point in enumerate(shape.parts()):
        #     cv.circle(img, center=(point.x, point.y), radius=2, color=(0,0,255),thickness=-1)
        #     cv.putText(img, text=str(i), org=(point.x,point.y), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255,255,255), thickness=2)
        try: 
            pig_x = shape.parts()[4].x
            pig_y = shape.parts()[4].y - 20

            h,w,c = sticker_img.shape

            nose_w = int((x2-x1) / 2.8)
            nose_h = int(h/w * nose_w)

            nose_x1 = int(pig_x - nose_w / 2)
            nose_x2 = nose_x1 + nose_w

            nose_y1 = int(pig_y - nose_h / 2) 
            nose_y2 = nose_y1 + nose_h

            overlay_img = sticker_img.copy()
            overlay_img = cv.resize(overlay_img, dsize=(nose_w, nose_h))
            overlay_alpha = overlay_img[:, :, 3:4] / 255.0
            background_alpha = 1.0 - overlay_alpha

            img[nose_y1:nose_y2, nose_x1:nose_x2] = overlay_alpha * overlay_img[:,:,:3] + background_alpha * img[nose_y1: nose_y2, nose_x1:nose_x2]
        except:
            pass
    cv.imshow('result', img)
    if cv.waitKey(1) == ord('q'):
        break

