import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# -----------------------------
# Model Path
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "emotion_model.keras")

print("Loading model from:")
print(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    print("ERROR: Model file not found!")
    exit()

# Load Model
model = load_model(MODEL_PATH)

# Emotion Labels
emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# Face Detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open Webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:

    ret, frame = camera.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float32") / 255.0

        roi = np.expand_dims(roi, axis=-1)
        roi = np.expand_dims(roi, axis=0)

        prediction = model.predict(roi, verbose=0)

        label = emotion_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        text = f"{label} ({confidence:.1f}%)"

        cv2.putText(
            frame,
            text,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Real-Time Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()