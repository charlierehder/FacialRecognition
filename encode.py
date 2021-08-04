from imutils import paths
import face_recognition
import pickle
import cv2
import os

# get all paths to images
image_paths = list(paths.list_images('train'))

# collect all encodings in each image and the corresponding name
encodings = []
names = []
for i,v in enumerate(image_paths):

    # capture name
    name = v.split(os.path.sep)[-2]

    # caputure image
    image = cv2.imread(v)
    rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
   
    # capture faces from image
    faces = face_recognition.face_locations(rgb, model='hog')

    # encode faces
    f_encodings = face_recognition.face_encodings(rgb, faces)

    # store names and encodings 
    for f_encoding in f_encodings:
        encodings.append(f_encoding)
        names.append(name)

# merge name and encoding lists into a dictionary
data = {"encodings": encodings, "names": names}

# store dictionary in pickle file
f = open("enc", "wb")
f.write(pickle.dumps(data))
f.close()
