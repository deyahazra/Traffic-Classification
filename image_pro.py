import numpy as np
import cv2
from keras.models import load_model
import labels


model = load_model("my_model.h5")
print(model.summary())

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error opening webcam")
    exit()
else:
    print("Webcam is opened")

f = 0
while True:
    f += 1
    ret, frame = cap.read()
    frame = cv2.resize(frame, (720, 480))
    display = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (90,90))
    frame = np.array(frame)
    frame = np.expand_dims(frame, axis=-1)
    frame = np.expand_dims(frame, axis=0)

    # classify the image
    if f % 60 == 0:
        predictions = model.predict(frame)
        predicted_labels = np.argmax(predictions, axis=1)
        print(labels.labels[predicted_labels[0]])

    cv2.imshow('frame', display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()






