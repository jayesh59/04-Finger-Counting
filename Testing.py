import cv2
import numpy as np
import matplotlib.pyplot as plt

cam = cv2.VideoCapture(0)
w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    
    ret,frame = cam.read()
    
    gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
    roi = gray[h-300:h,w-300:w]
    _ , thresholded = cv2.threshold(roi, 240, 255, cv2.THRESH_BINARY )

    cv2.rectangle(frame, (w-300,h-300) , (w,h), 255, 5)
    thresholded = np.expand_dims(thresholded, axis = 2)

    frame[h-300:h,w-300:w] = thresholded
    
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(10) & 0xFF == 27:
        print(frame.shape)
        type(w)
        break

cam.release()
cv2.destroyAllWindows()
