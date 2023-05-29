import numpy as np
import cv2 as cv
import sys, os

extension = ".png"

if (len(sys.argv) < 1):
    print("Usage: python videoCapture.py <deviceID>")
    exit()

deviceID = int(sys.argv[1])

cap = cv.VideoCapture(deviceID)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('s'):
        files = os.listdir(os.getcwd())
        idx = len([f for f in files if extension in f])
        filename = 'image'+str(idx)+extension
        print("current filename = ", filename)
        cv.imwrite(filename,frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
