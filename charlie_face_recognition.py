import face_recognition
import imutils
import pickle
import time
import cv2
import os

# load opencv face detection, instantiate our classifier and load pickle
# files of target faces
path = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
casc_classifier = cv2.CascadeClassifier(path)
data = pickle.loads(open('enc', "rb").read())
 
video_capture = cv2.VideoCapture(0)

while True:
    
    # capture and analyze frame
    ret, frame = video_capture.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = casc_classifier.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    encodings = face_recognition.face_encodings(rgb)

    names = []
    for encoding in encodings:

        # compare encodings from webcam to stored encodings. represent as list of 
        # booleans
        matches = face_recognition.compare_faces(data["encodings"], encoding)

        # placeholder name
        name = ""

        # dictionary that contains names and the number of matches in the encoding
        name_count = {}

        if True in matches:
    
            for i,v in enumerate(matches):
                if v:
                    name_at_index = data["names"][i]
                    name_count[name_at_index] = name_count.get(name_at_index, 0) + 1
            
            name = max(name_count, key=name_count.get) 

        names.append(name)

        # draw rectangle around face        
        for ((x, y, w, h), name) in zip(faces, names):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    # check for quit command
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# close window and quit
video_capture.release()
cv2.destroyAllWindows()
