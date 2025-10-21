import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import requests
import os

st.title("🐶🐱 Cat vs Dog Classifier")

# URL to your GitHub LFS model (raw link)
# Replace <your-username> and <repo-name> with your actual repo info
MODEL_URL = "https://raw.githubusercontent.com/Onyi20/cat_dog_image_classification/main/catdog_densenet_model.h5"
MODEL_PATH = "catdog_densenet_model.h5"

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Downloading model from GitHub..."):
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            st.success("✅ Model downloaded successfully.")
        else:
            st.error(f"❌ Failed to download model. Status: {response.status_code}")
            st.stop()

@st.cache_resource
def load_model():
    """Load the trained Keras model (cached)."""
    tf.keras.backend.clear_session()
    return tf.keras.models.load_model(MODEL_PATH)

# Load model
model = load_model()
st.info("✅ Model loaded and ready!")

# Define class labels
class_names = ["Cat", "Dog"]

# Upload image
uploaded_file = st.file_uploader("📤 Upload an image of a cat or dog", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    preds = model.predict(img_array)
    predicted_class_index = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))

    st.markdown("---")
    st.subheader("🔍 Prediction Result")
    st.write(f"**Predicted Class:** {class_names[predicted_class_index]}")
    st.write(f"**Confidence:** {confidence:.2%}")
