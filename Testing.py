import cv2
import numpy as np
import matplotlib.pyplot as plt

cam = cv2.VideoCapture(0)
w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    
    ret,frame = cam.read()
    blured = cv2.blur(frame, (3,3))
    gray = cv2.cvtColor(blured, cv2.COLOR_BGR2GRAY)
    roi = gray[h-250:h,w-250:w]
    
    _ , thresholded = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    transformed = cv2.distanceTransform(thresholded, cv2.DIST_L2, 3)
    trans_thresholded = cv2.thresholded(transformed, 0.1*transformed.max(),255,0)

    cv2.rectangle(frame, (w-250,h-250) , (w,h), 255, 5)
    thresholded = np.expand_dims(thresholded, axis = 2)
    transformed = np.expand_dims(transformed, axis = 2)
    trans_thresholded = np.expand_dims(trans_thresholded, axis = 2)


    frame[h-250:h,w-250:w] = trans_thresholded
    
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(10) & 0xFF == 27:
        print(frame.shape)
        type(w)
        break

cam.release()
cv2.destroyAllWindows()
