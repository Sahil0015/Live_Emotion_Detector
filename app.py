import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from keras.models import model_from_json, Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.saving import register_keras_serializable
from PIL import Image

# Register Sequential and other layers as serializable to prevent errors
register_keras_serializable()(Sequential)
register_keras_serializable()(Conv2D)
register_keras_serializable()(MaxPooling2D)
register_keras_serializable()(Dropout)
register_keras_serializable()(Flatten)
register_keras_serializable()(Dense)

@st.cache_resource
def load_model():
    try:
        with open("emotiondetector.json", "r") as json_file:
            model_json = json_file.read()
        model = tf.keras.models.model_from_json(model_json, custom_objects={"Sequential": Sequential})
        model.load_weights("emotiondetector.h5")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Load model safely
model = load_model()
if model is None:
    st.stop()

# Load Haarcascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Preprocessing function
def extract_features(image):
    feature = np.array(image).reshape(1, 48, 48, 1)
    return feature / 255.0  # Normalize

# Streamlit UI
st.title("😃 Facial Emotion Recognition")
st.write("Choose whether to upload an image or use live webcam for emotion detection.")

option = st.radio("Select an option:", ("Upload an Image", "Live Emotion Detection"))

if option == "Upload an Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
        image = np.array(image)
        
        # Improve face detection by adjusting contrast
        image = cv2.equalizeHist(image)

        # Resize for better detection (Optional)
        image = cv2.resize(image, (500, 500), interpolation=cv2.INTER_CUBIC)

        # Try multiple face detectors
        face_cascades = [
            cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'),
            cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        ]
        
        faces = None
        for cascade in face_cascades:
            faces = cascade.detectMultiScale(image, scaleFactor=1.05, minNeighbors=3)
            if len(faces) > 0:
                break  # Stop when faces are detected
        
        if len(faces) == 0:
            st.warning("⚠ No face detected! Try a clearer image with better lighting or a closer face.")
        else:
            for (x, y, w, h) in faces:
                face = image[y:y+h, x:x+w]
                face = cv2.resize(face, (48, 48))
                img = extract_features(face)
                pred = model.predict(img)
                prediction_label = labels[pred.argmax()]
                st.success(f"🎭 Predicted Emotion: {prediction_label}")
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

elif option == "Live Emotion Detection":
    use_webcam = st.checkbox("Start Webcam")
    if use_webcam:
        webcam = cv2.VideoCapture(0)
        frame_placeholder = st.empty()

        while True:
            ret, frame = webcam.read()
            if not ret:
                st.error("Failed to access webcam")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (48, 48))
                img = extract_features(face)
                pred = model.predict(img)
                prediction_label = labels[pred.argmax()]
                
                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, prediction_label, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            frame_placeholder.image(frame, channels="BGR")

            if not use_webcam:
                webcam.release()
                break
