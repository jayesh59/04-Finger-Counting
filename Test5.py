import cv2
import numpy as np
import matplotlib.pyplot as plt

background = None
accum_weight = 0.5


def calc_avg(frame, accum_weight):
    global background

    if background is None:
        background = frame.copy().astype('float')
    

        return None
    else:
        pass

    cv2.accumulateWeighted(frame, background, accum_weight)

def segment(frame, thresh_min = 20):
    global background

    diff = cv2.absdiff(frame,background.astype('uint8'))

    ret, thresholded = cv2.threshold(diff, thresh_min, 255 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contour, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contour) == 0:
        return None

    else:
        hand = max(contour, key = cv2.contourArea)
        #hand = contour

    return (thresholded, hand)

def display(frame,frame_dict):
    
    cv2.imshow('Original', frame[h-250:h,w-250:w])
    l1 = list(frame_dict.keys())
    l2 = list(frame_dict.values())
    for i in range(len(frame_dict)):
        cv2.imshow(l1[i],l2[i])

    return None    

def diff_contours(dist_trans):
    #dist_trans = cv2.cvtColor(dist_trans, cv2.COLOR_BGR2GRAY)
    _ , Dist_Thresh10 = cv2.threshold(dist_trans, 0.1*dist_trans.max(), 255, cv2.THRESH_BINARY)
    _ , Dist_Thresh40 = cv2.threshold(dist_trans, 0.4*dist_trans.max(), 255, cv2.THRESH_BINARY)
    fingers = cv2.absdiff(Dist_Thresh10, Dist_Thresh40)

    return (Dist_Thresh10, Dist_Thresh40, fingers)

cap = cv2.VideoCapture(0)

_, frame = cap.read()
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
num_frames = 0


#black = frame.copy()[h-250:h,w-250:w]
#black = black[:,:,:
while True:

    _, frame = cap.read()
    frame = cv2.GaussianBlur(frame, (7,7), 0)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    black = np.zeros((250,250))
    cont_img = black
    
    if num_frames < 60:
        calc_avg(frame[h-250:h,w-250:w],accum_weight)

    else:

        a = segment(frame[h-250:h,w-250:w])
        
        if a is not None:
            th, cont = a
            
            for i in range(len(cont)):
                cont_img = cv2.drawContours(cont_img, cont, i, 255, -1)
                #cont_img = cv2.drawContours(black, cont, 0, [255,255,255], -1)
                #cont_img = np.expand_dims(cont_img, axis = 2)
                #_,cont_img = cv2.threshold(cont_img,127, 255, cv2.THRESH_BINARY)
                th2 = cv2.threshold(cont_img, 127, 255, cv2.THRESH_BINARY)
                dist_trans = cv2.distanceTransform(th2, cv2.DIST_L2, 3)
                b = diff_contours(dist_trans)
                Dist_Thresh10, Dist_Thresh40, fingers = b


                frame_dict = {'Thresholded':th, 'Contour':cont_img, 'Distance Transformation':dist_trans, '10% Distance Threshold':Dist_Thresh10, '40% Distance Threshold':Dist_Thresh40, 'Fingers':fingers}
                
                
            #c = np.array(cont)
            #print(len(cont))
            #print(cont_img.shape)
            #cv2.imshow('cont' , frame)
            display(frame, frame_dict)
            #cv2.imshow('dist atransform',th2)
    num_frames += 1

    if cv2.waitKey(10) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
