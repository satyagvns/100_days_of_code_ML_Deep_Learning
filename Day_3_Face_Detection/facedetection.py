import cv2
import numpy as np
import sys

facePath = "haarcascade_frontalface_default.xml"
smilePath = "haarcascade_smile.xml"
faceCascade = cv2.CascadeClassifier(facePath)
smileCascade = cv2.CascadeClassifier(smilePath)

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

sF = 1.05

while True:

    ret, frame = cap.read() # Capture frame-by-frame
    img = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor= sF,
        minNeighbors=8,
        minSize=(55, 55)
    )
    # ---- Draw a rectangle around the faces

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        smile = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor= 1.7,
            minNeighbors=22,
            minSize=(25, 25)
            )

        # Set region of interest for smiles
        for (x, y, w, h) in smile:
            #print "Found", len(smile), "smiles!"
            cv2.rectangle(smile, (x, y), (x+w, y+h), (0, 255, 0), 2)
            #print "!!!!!!!!!!!!!!!!!"

    #cv2.cv.Flip(frame, None, 1)
    cv2.imshow('Smile Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()