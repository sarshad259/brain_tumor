import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(page_title="Brain Tumor Detection System")

st.title("🧠 Brain Tumor Detection System (AI Powered)")

model = tf.keras.models.load_model("brain_tumor_model.keras")

uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "png", "jpeg"]
)

def preprocess_image(image):
    image = image.convert("RGB")   # 🔴 IMPORTANT FIX
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", width=200)

    if st.button("Detect Tumor"):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)[0][0]

        confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100

        if prediction > 0.5:
            st.error(f"🛑 Prediction: **Tumor Detected**")
        else:
            st.success(f"✅ Prediction: **No Tumor Detected**")

        st.info(f"Confidence Score: **{confidence:.2f}%**")
