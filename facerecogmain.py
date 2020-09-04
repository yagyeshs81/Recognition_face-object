import cv2
import numpy as np
import face_recognition
import os
# from datetime import datetime
# import pickle
# import _string as sr
# import pyaudio
# import random
import definefunctions as defu
#import speech_recognition as s
# from _datetime import datetime
# import time
names = 'kavya'
whT = 320
confThreshold = 0.5
nmsThreshold = 0.2


# from PIL import ImageGrab
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
path = 'faces-only'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)



#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr

encodeListKnown = defu.findEncodings(images)
print('Encoding Complete')




## Model Files
modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


sr = s.Recognizer()


cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    # img = captureScreen()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        # print(matches)


        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
             # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            defu.markAttendance(name)
        else:
            
            for (x, y, w, h) in faces:
                # print(x,y,w,h)
                roi_gray = gray[y:y + h, x:x + w]  # (ycord_start, ycord_end)
                roi_color = img[y:y + h, x:x + w]
                img_item = "build_f/" + names + ".png"
                cv2.imwrite(img_item, roi_color)

                color = (255, 0, 0)  # BGR 0-255
                stroke = 2
                end_cord_x = x + w
                end_cord_y = y + h
                cv2.rectangle(img, (x, y), (end_cord_x, end_cord_y), color, stroke)

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    defu.findObjects(outputs, img)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cv2.waitKey(0)
cv2.destroyAllWindows()
