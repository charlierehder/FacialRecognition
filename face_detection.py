import cv2
import os

path = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_default.xml"
casc_classifier = cv2.CascadeClassifier(path)

video_capture = cv2.VideoCapture(0)

while True:

    # capture and analyze frames
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = casc_classifier.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30,30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    
    # draw rectangle around face
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow('Video', frame)

    # check for quit command
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
