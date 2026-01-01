import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

st.set_page_config(
    page_title="Brain Tumor Detection System",
    layout="centered"
)

st.title("🧠 Brain Tumor Detection System (AI Powered)")

# ---------- MODEL LOADING (SAFE WAY) ----------
@st.cache_resource
def load_model():
    model_path = os.path.join("model", "brain_tumor_model.keras")

    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Please check the 'model' folder.")
        st.stop()

    return tf.keras.models.load_model(model_path)

model = load_model()

# ---------- IMAGE UPLOAD ----------
uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "png", "jpeg"]
)

# ---------- PREPROCESS ----------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ---------- PREDICTION ----------
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    if st.button("Detect Tumor"):
        processed_image = preprocess_image(image)

        prediction = model.predict(processed_image)[0][0]

        confidence = (
            prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
        )

        if prediction > 0.5:
            st.error("🛑 Prediction: Tumor Detected")
        else:
            st.success("✅ Prediction: No Tumor Detected")

        st.info(f"Confidence Score: {confidence:.2f}%")
