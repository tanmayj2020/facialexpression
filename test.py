
import numpy as np
import cv2
from keras.models import model_from_json

from keras.preprocessing import image

face_classifier = cv2.CascadeClassifier("./face.xml")
model = model_from_json(open("model.json" , "r").read())
model.load_weights("my_model.h5")

class_labels = ['angry' , 'disgust' , 'fear','happy' , 'neutral' , 'sad' , 'suprise']


cap = cv2.VideoCapture(0)

while True:
    ret , frame = cap.read()
    if not ret:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray , 1.3 , 5)

    for (x , y , w,h) in faces:
        cv2.rectangle(frame , (x,y) ,(x+w , y+h) , (255 , 0 , 0) ,2)
        roi_gray = gray[y:y+h , x : x+w]
        roi_gray = cv2.resize(roi_gray , (48,48) , interpolation = cv2.INTER_AREA)
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        max_value = round(np.max(predictions[0]) * 100 , 2)
        predicted_emotion = class_labels[max_index]

    cv2.putText(frame , predicted_emotion + " {}%".format(max_value) , (int(x-5) , int(y-5)) , cv2.FONT_HERSHEY_SIMPLEX  , 1 , (0,0,255),2,cv2.LINE_AA)
    resized_img = cv2.resize(frame , (1000,700))
    cv2.imshow("Facial Emotion Detection" , resized_img)

    if cv2.waitKey(1) == 'q':
        break

cap.release()
cv2.destroyAllWindows()
        


