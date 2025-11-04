# src/app.py
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

st.set_page_config(page_title="Driver Drowsiness Detector", layout="centered")
st.title("🚗 Driver Drowsiness & Yawn Detection")

@st.cache_resource
def load_drowsiness_model():
    model = load_model("models/drowsiness_model.h5")
    return model

model = load_drowsiness_model()
classes = ['Closed', 'Open', 'no_yawn', 'yawn']  # adjust order if needed

option = st.radio("Choose Input Type:", ["Upload Image", "Use Webcam"])

if option == "Upload Image":
    file = st.file_uploader("Upload a driver's face image:", type=["jpg","png","jpeg"])
    if file:
        img = Image.open(file).convert("RGB").resize((224,224))
        st.image(img, caption="Uploaded Image", use_container_width=True)
        x = np.expand_dims(np.array(img)/255.0, axis=0)
        preds = model.predict(x)[0]
        label = classes[np.argmax(preds)]
        conf = np.max(preds)
        st.markdown(f"**Prediction:** {label} — Confidence: {conf:.2f}")
        if label in ["Closed", "yawn"]:
            st.error("⚠️ Driver seems DROWSY! Please take a break.")
        else:
            st.success("✅ Driver seems ALERT.")

elif option == "Use Webcam":
    st.info("Press 'Start' to open webcam (works on Streamlit Cloud with webcam permission).")
    run = st.checkbox("Start")
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    while run:
        ret, frame = camera.read()
        if not ret:
            st.write("Failed to grab frame")
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img, (224,224))
        x = np.expand_dims(resized/255.0, axis=0)
        preds = model.predict(x)[0]
        label = classes[np.argmax(preds)]
        color = (0,255,0) if label in ["Open","no_yawn"] else (255,0,0)
        cv2.putText(img, f"{label}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        FRAME_WINDOW.image(img)
    camera.release()
