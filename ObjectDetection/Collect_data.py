import cv2 as cv
import numpy as np
import os
import time

def dir_size(dir):
    return len(os.listdir(dir))

def get_frame(cap):
    ret, frame = cap.read()
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame = frame[0:height,int((0.5*width)-0.5*height):int((0.5*width)+0.5*height)]
    frame = cv.flip(frame, 1)
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        cap.release()
        cv.destroyAllWindows()
        exit()
    return frame

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

now = 0
taken = 0

while True:
    frame = get_frame(cap)
    cv.imshow('frame',frame)

    if (time.time() > 2+now+0.5*taken) and (time.time() < now+4):
        taken +=1
        filename = os.path.join(var, 'new_{}{}.jpg'.format(var,dir_size(var)))
        print("saving frame to {}".format(filename))
        cv.imwrite(filename,frame)
    elif time.time() > now+4:
        taken = 0

    if cv.waitKey(1) == ord('r'):
        print("taking photos for \"rock\" in 2 seconds")
        now = time.time()
        var = 'rock'

    if cv.waitKey(1) == ord('p'):
        print("taking photos for \"paper\" in 2 seconds")
        now = time.time()
        var = 'paper'

    if cv.waitKey(1) == ord('s'):
        print("taking photos for \"scissors\" in 2 seconds")
        now = time.time()
        var = 'scissors'

    if cv.waitKey(1) == ord('q'):
        break


cap.release()
cv.destroyAllWindows()