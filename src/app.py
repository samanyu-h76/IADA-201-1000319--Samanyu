# src/app.py
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2
import os

# --------------------------- #
# Streamlit Config
# --------------------------- #
st.set_page_config(page_title="Driver Drowsiness Detector", layout="centered")
st.title("🚗 Driver Drowsiness & Yawn Detection")

# --------------------------- #
# Load Model
# --------------------------- #
@st.cache_resource
def load_drowsiness_model():
    model_path = "models/drowsiness_model.h5"
    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Please ensure it's in the 'models/' folder.")
        st.stop()
    return load_model(model_path)

model = load_drowsiness_model()
classes = ['Closed', 'Open', 'no_yawn', 'yawn']  # adjust if needed

# --------------------------- #
# Face Detection Helper
# --------------------------- #
def detect_and_crop_face(img):
    """Detects and crops the largest face region from an image."""
    img_cv = np.array(img)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        # No face found — return original resized image
        return img.resize((224, 224))
    # Choose the largest detected face
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    face = img.crop((x, y, x + w, y + h))
    return face.resize((224, 224))

# --------------------------- #
# Prediction Helper
# --------------------------- #
def predict_image(img_array):
    x = np.expand_dims(img_array / 255.0, axis=0)
    preds = model.predict(x)[0]
    label = classes[np.argmax(preds)]
    conf = float(np.max(preds))
    return label, conf

# --------------------------- #
# Streamlit Interface
# --------------------------- #
option = st.radio("Choose Input Type:", ["Upload Image", "Use Webcam"])

# --- Upload Image Option ---
if option == "Upload Image":
    file = st.file_uploader("Upload a driver's face image:", type=["jpg", "png", "jpeg"])
    if file:
        # Load and preprocess
        img = Image.open(file).convert("RGB")
        cropped = detect_and_crop_face(img)
        st.image(cropped, caption="Detected Face (used for prediction)", use_container_width=True)

        label, conf = predict_image(np.array(cropped))
        st.markdown(f"### Prediction: **{label}** — Confidence: {conf:.2f}")

        # Drowsiness logic
        if label in ["Closed", "yawn"]:
            st.error("⚠️ Driver seems **DROWSY** 😴 — Please take a break.")
        else:
            st.success("✅ Driver seems **ALERT** 😎 — Keep driving safely!")

# --- Webcam Option ---
elif option == "Use Webcam":
    st.info("Press 'Start' to open webcam (works locally; limited on Streamlit Cloud).")
    run = st.checkbox("Start")
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.warning("Unable to access webcam.")
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
            face = cv2.resize(img_rgb[y:y + h, x:x + w], (224, 224))
            label, conf = predict_image(face)
            color = (0, 255, 0) if label in ["Open", "no_yawn"] else (255, 0, 0)
            cv2.rectangle(img_rgb, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img_rgb, f"{label} ({conf:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        else:
            label, conf = "No face", 0.0

        FRAME_WINDOW.image(img_rgb)

    camera.release()
