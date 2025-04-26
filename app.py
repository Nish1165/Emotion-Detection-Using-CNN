from flask import Flask, Response, render_template
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load trained model
MODEL_PATH = "model/emotion_model.h5"
model = load_model(MODEL_PATH)

# Emotion labels based on training order
CLASS_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to process frames from the webcam
def detect_emotion():
    video = cv2.VideoCapture(0)  # Open webcam

    while True:
        ret, frame = video.read()
        if not ret:
            break  # Stop if there's an issue with the webcam
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))  # Resize to model input size
            face = face.astype("float32") / 255.0  # Normalize
            face = np.expand_dims(face, axis=0)
            face = np.expand_dims(face, axis=-1)

            # Predict emotion
            prediction = model.predict(face)
            predicted_class = np.argmax(prediction, axis=1)[0]
            emotion_label = CLASS_LABELS[predicted_class]

            # Draw rectangle around face & label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
            cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Encode frame to send it via Flask
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    video.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template("index.html")  # HTML page to display the video feed

@app.route('/video_feed')
def video_feed():
    return Response(detect_emotion(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)


