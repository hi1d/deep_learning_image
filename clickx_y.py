import cv2 as cv 

class MouseGesture():
    def __init__(self) -> None:
        self.is_dragging = False 
        self.x0, self.y0, self.w0, self.h0 = -1,-1,-1,-1

    def on_mouse(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            print("x : {} y : {}".format(x,y) )
        return 