import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model("image_classifier_model.h5")

# App title
st.title("🧠 Image Classification App")

# Upload image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((224, 224))  # adjust to your model’s input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalization

    # Predict
    preds = model.predict(img_array)
    predicted_class = np.argmax(preds, axis=1)
    st.write(f"✅ Predicted class: {predicted_class}")
