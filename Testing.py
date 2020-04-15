import cv2
import numpy as np
import matplotlib.pyplot as plt

cam = cv2.VideoCapture(0)
w = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
obj = cv2.CascadeClassifier('data/haarcascade/Hand.Cascade.1.xml')


while True:
    ret,frame = cam.read()
    
    cv2.imshow('frame', frame)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hand_rect = obj.detectMultiScale(frame_gray, 1.2,5)

    for (x,y,w,h) in hand_rect:
        cv2.rectangle(frame, (x,y) , (x+w,y+h), 255, 5)

    if cv2.waitKey(10) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
