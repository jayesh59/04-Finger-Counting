import cv2 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise

background = None
accum_weight = 0.5

def calc_avg(frame, accum_weight):
    global background

    if background is None:
        background = frame.copy().astype('float')
        return None

    cv2.accumulateWeighted(frame, background, accum_weight)

def segment(frame, threshold_min = 25):
    global background

    diff = cv2.absdiff(frame, background.astype('uint8'))
    ret, thresholded = cv2.threshold(diff, threshold_min, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    else:
        hand_segment = max(contours, key = cv2.contourArea)

    return (thresholded, hand_segment)

def count_fingers(thresholded, hand_segment):

    conv_hull = cv2.convexHull(hand_segment)

    top = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
    top = tuple(conv_hull[conv_hull[:,:,1].argmin()][0])
    bottom = tuple(conv_hull[conv_hull[:,:,1].argmax()][0])
    left = tuple(conv_hull[conv_hull[:,:,0].argmin()][0])
    right = tuple(conv_hull[conv_hull[:,:,0].argmax()][0])

    cX = (right[0] + left[0]) // 2
    cY = (top[1] + bottom[1]) // 2

    distance = pairwise.euclidean_distances([(cX, cY)], Y=[left, right, top, bottom])[0]

    radius = int(0.8* distance.max())
    perimeter = 2* np.pi * radius

    circular_roi = np.zeros(thresholded.shape[:2], dtype = 'uint8')
    cv2.circle(circular_roi, (cX,cY), radius, 255, 10)

    circular_roi = cv2.bitwise_and(thresholded,thresholded, mask = circular_roi)

    contours, hierarchy = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0

    for cont in contours:
        
        (x,y,w,h) = cv2.boundingRect(cont)
        out_wrist = (cY + cY*(0.25)) > (y+h)
        limit =  ((perimeter*0.25) > cont.shape[0])

        if limit and out_wrist:
            count += 1

    return count



cam = cv2.VideoCapture(0)
num_frame = 0

while True:
    
    ret, frame = cam.read()
    frame = cv2.flip(frame,1)
    frame_copy = frame.copy()
    gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7,7), 0)

    h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))

    roi = frame[h-250:h, w-250:w]

    if num_frame < 60:
        calc_avg(gray, accum_weight)

    else:

        hand = segment(gray)    

        if hand is not None:

            th, hand_segment = hand
            cv2.drawContours(frame_copy, hand_segment, -1, 255, 5)
            finger = count_fingers(th, hand_segment)

            print(finger)
            cv2.imshow('Threshold', th)

    cv2.rectangle(frame, (250-h, 250-w), (h,w), 255, 5)
    num_frame += 1 

    cv2.imshow('Fingers',frame_copy)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()


            
            

    
