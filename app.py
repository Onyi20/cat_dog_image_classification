import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import gdown
import os

st.title("🐶🐱 Cat vs Dog Classifier")

drive_url = "https://drive.google.com/uc?id=1PpUGfLU6_DUEXA_dKp-U6tZopHQT_OP-"
model_path = "binary_image_classifier_model.keras"

def clear_streamlit_cache():
    st.cache_data.clear()
    st.cache_resource.clear()

if not os.path.exists(model_path):
    with st.spinner("📥 Downloading model..."):
        gdown.download(drive_url, model_path, quiet=False)
    st.success("✅ Model downloaded")
    clear_streamlit_cache()

@st.cache_resource
def load_model():
    # Reset TF session to clean stale state
    tf.keras.backend.clear_session()
    loaded_model = tf.keras.models.load_model(model_path)
    return loaded_model

model = load_model()
st.info("Model loaded.")

class_names = ["Cat", "Dog"]

uploaded_file = st.file_uploader("Upload image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width='stretch')

    # Preprocess
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Debug input info
    st.write("Input image shape:", img_array.shape)
    st.write("Input image mean pixel value:", np.mean(img_array))
    st.write("Input image std pixel value:", np.std(img_array))

    preds = model.predict(img_array)

    st.write("Raw model output:", preds)
    predicted_class_index = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))

    st.write(f"Predicted Class: {class_names[predicted_class_index]}")
    st.write(f"Confidence: {confidence:.4f}")

    if st.button("Clear Cache"):
        clear_streamlit_cache()
        # No need for st.experimental_rerun() here - widget actions refresh app automatically
