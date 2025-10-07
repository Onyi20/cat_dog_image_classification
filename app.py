import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import gdown
import os

# --- App title ---
st.title("🐶🐱 Cat vs Dog Classifier")

# --- Define the Google Drive link ---
drive_url = "https://drive.google.com/uc?id=1PpUGfLU6_DUEXA_dKp-U6tZopHQT_OP-"
model_path = "binary_image_classifier_model.keras"

# --- Clear cache if model or new file uploaded ---
def clear_streamlit_cache():
    """Force clear Streamlit’s cache."""
    st.cache_data.clear()
    st.cache_resource.clear()

# --- Download the model if it doesn't exist locally ---
if not os.path.exists(model_path):
    with st.spinner("📥 Downloading model from Google Drive..."):
        gdown.download(drive_url, model_path, quiet=False)
    st.success("✅ Model downloaded successfully!")
    clear_streamlit_cache()  # clear cache after downloading new model


# --- Load the trained model with caching ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path)

model = load_model()
st.info("Model loaded and ready for predictions!")

# --- Define class names ---
class_names = ["Cat", "Dog"]

# --- Upload image ---
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Clear cache when a new image is uploaded
    clear_streamlit_cache()

    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # --- Preprocess image ---
    img = img.resize((224, 224))  # Match model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # --- Predict ---
    preds = model.predict(img_array)
    predicted_class_index = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))

    # --- Show result ---
    st.write(f"✅ **Predicted Class:** {class_names[predicted_class_index]}")
    st.write(f"📊 **Confidence:** {confidence:.4f}")
    
     # --- Optional button to clear cache manually ---
    if st.button("♻️ Clear Cache"):
        clear_streamlit_cache()
        st.experimental_rerun()
