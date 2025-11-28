# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model

# # Load your trained model (change filename if needed)
# model = load_model(r"C:\Users\Harshit Sharma\OneDrive\Desktop\ML\Project 1\emotion_detection_model.keras")
 

# # Define emotion labels (in the same order as your training)
# emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# # Load face detector
# face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Start video capture
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame")
#         break

#     # Convert frame to grayscale for face detection
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_classifier.detectMultiScale(gray, 1.3, 5)

#     for (x, y, w, h) in faces:
#         # Draw rectangle around face
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

#         # Crop and preprocess the face
#         roi_gray = gray[y:y + h, x:x + w]
#         roi_gray = cv2.resize(roi_gray, (48, 48))
#         roi_gray = roi_gray.astype('float') / 255.0
#         roi_gray = np.expand_dims(roi_gray, axis=0)
#         roi_gray = np.expand_dims(roi_gray, axis=-1)

#         # Predict emotion
#         prediction = model.predict(roi_gray)
#         label = emotion_labels[np.argmax(prediction)]

#         # Display emotion label
#         cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

#     # Show video window
#     cv2.imshow('Real-time Emotion Detection', frame)

#     # Press 'q' to quit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model (update path if needed)
model = load_model(r"C:\Users\Harshit Sharma\OneDrive\Desktop\ML\Project 1\emotion_detection_model.keras")

# Emotion class labels — adjust if your dataset used different labels
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

# Initialize camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    frame = cv2.flip(frame, 1)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype("float") / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        # Prediction
        preds = model.predict(roi, verbose=0)[0]
        emotion = emotion_labels[np.argmax(preds)]

        # Draw
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imshow("Real-Time Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





