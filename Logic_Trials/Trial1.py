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
    cv2.rectangle(frame, (w-250,h-250) , (w,h), 255, 5)
    
    _ , thresholded = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    transformed = cv2.distanceTransform(thresholded, cv2.DIST_L2, 3)
    _ , trans_thresholded40 = cv2.threshold(transformed, 0.40*transformed.max(),255,0)
    _ , trans_thresholded10 = cv2.threshold(transformed, 0.10*transformed.max(),255,0)
    diff = cv2.absdiff(trans_thresholded10,trans_thresholded40)

    
    thresholded = np.expand_dims(thresholded, axis = 2)
    transformed = np.expand_dims(transformed, axis = 2)
    trans_thresholded10 = np.expand_dims(trans_thresholded10, axis = 2)
    trans_thresholded40 = np.expand_dims(trans_thresholded40, axis = 2)
    diff = np.expand_dims(diff, axis = 2)


    frame[h-250:h,w-250:w] = thresholded
    
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(10) & 0xFF == 27:
        print(frame.shape)
        type(w)
        break

cam.release()
cv2.destroyAllWindows()
