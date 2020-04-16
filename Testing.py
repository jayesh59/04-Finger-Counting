import cv2
import numpy as np
import matplotlib.pyplot as plt

cam = cv2.VideoCapture(0)
w = cam.get(cv2.CAP_PROP_FRAME_WIDTH)   
h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

while True:
    
    ret,frame = cam.read()
    #background = frame.copy().astype('float')
    
    diff = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _ , thresholded = cv2.threshold(diff, 127, 255, cv2.THRESH_BINARY)




    
    '''
    for (x,y,w,h) in hand_rect:
        cv2.rectangle(frame, (x,y) , (x+w,y+h), (255,255,255), 5)
    '''
    
    cv2.imshow('frame',thresholded)
    
    if cv2.waitKey(10) & 0xFF == 27:
        print(frame.shape)
        break

cam.release()
cv2.destroyAllWindows()
