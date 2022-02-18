from ctypes import pointer
import cv2 as cv
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('WEEK_4/models/shape_predictor_5_face_landmarks.dat')

# cap = cv.VideoCapture('WEEK_4/media/01.mp4')
cap = cv.VideoCapture(0)
sticker_img = cv.imread('WEEK_4/media/stickers/glasses.png', cv.IMREAD_UNCHANGED)


while True:
    ret, img = cap.read()

    if ret == False:
        break

    dets = detector(img)

    for det in dets:
        shape = predictor(img, det)

        # for i, point in enumerate(shape.parts()):
        #     cv.circle(img, center=(point.x, point.y), radius=2, color=(0,0,255),thickness=-1)
        #     cv.putText(img, text=str(i), org=(point.x,point.y), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255,255,255), thickness=2)
        try:
            glasses_x1 = shape.parts()[2].x - 20
            glasses_x2 = shape.parts()[0].x + 20

            h, w, c = sticker_img.shape

            glasses_w = glasses_x2-glasses_x1
            glasses_h = int(h/w * glasses_w)

            center_y = (shape.parts()[0].y + shape.parts()[2].y) / 2

            glasses_y1 = int(center_y - glasses_h / 2)
            glasses_y2 = glasses_y1 + glasses_h

            overlay_img = sticker_img.copy()
            overlay_img = cv.resize(overlay_img, dsize=(glasses_w, glasses_h))

            overlay_alpha = overlay_img[:, :, 3:4] / 255.0
            background_alpha = 1.0 - overlay_alpha
            print()
            img[glasses_y1:glasses_y2, glasses_x1: glasses_x2] = overlay_alpha * overlay_img[:,:,:3] + background_alpha * img[glasses_y1: glasses_y2, glasses_x1:glasses_x2]
        except:
            pass

    cv.imshow('result', img)
    if cv.waitKey(1) == ord('q'):
        break