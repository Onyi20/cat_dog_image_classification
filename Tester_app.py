import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱🐶",
    layout="centered"
)

# Title and description
st.title("🐱🐶 Cat vs Dog Image Classifier")
st.markdown("""
Upload an image of a cat or dog, and the AI will predict which one it is!
This model was trained on 15,160 images and achieves **86% accuracy**.
""")

# Load model (cached to avoid reloading on every interaction)
@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model("binary_image_classifier_Multi_layer_model.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Make sure 'binary_image_classifier_Multi_layer_model.keras' is in the same directory as this script.")
        return None

# Preprocess image for model
def preprocess_image(image):
    """
    Preprocess uploaded image to match model input requirements:
    - Resize to 224x224
    - Convert to RGB
    - Normalize pixel values
    - Add batch dimension
    """
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to model input size
    image = image.resize((224, 224))
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize pixel values to [0, 1] range
    img_array = img_array.astype('float32') / 255.0
    
    return img_array

# Make prediction
def predict(model, image):
    """
    Make prediction on preprocessed image
    Returns: class_name, confidence
    """
    # Preprocess
    processed_image = preprocess_image(image)
    
    # Predict
    predictions = model.predict(processed_image, verbose=0)
    
    # Get class (0 = Cat, 1 = Dog based on alphabetical order)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx] * 100
    
    class_names = ['Cat', 'Dog']
    predicted_class = class_names[class_idx]
    
    return predicted_class, confidence, predictions[0]

# Main app
def main():
    # Load model
    model = load_model()
    
    if model is None:
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload a clear image of a cat or dog for best results"
    )
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    if uploaded_file is not None:
        # Read and display image
        image = Image.open(uploaded_file)
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("Prediction")
            
            # Add a predict button
            if st.button("🔍 Predict", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    # Make prediction
                    predicted_class, confidence, raw_predictions = predict(model, image)
                    
                    # Display result with styling
                    if predicted_class == "Cat":
                        st.markdown(f"### 🐱 It's a **Cat**!")
                        emoji = "🐱"
                    else:
                        st.markdown(f"### 🐶 It's a **Dog**!")
                        emoji = "🐶"
                    
                    # Confidence meter
                    st.metric("Confidence", f"{confidence:.2f}%")
                    
                    # Progress bar for confidence
                    st.progress(confidence / 100)
                    
                    # Show detailed probabilities
                    with st.expander("📊 Detailed Probabilities"):
                        cat_prob = raw_predictions[0] * 100
                        dog_prob = raw_predictions[1] * 100
                        
                        st.write(f"**Cat**: {cat_prob:.2f}%")
                        st.write(f"**Dog**: {dog_prob:.2f}%")
                    
                    # Confidence interpretation
                    if confidence > 90:
                        st.success("Very confident prediction! ✨")
                    elif confidence > 70:
                        st.info("Moderately confident prediction.")
                    else:
                        st.warning("Low confidence - the image might be unclear or ambiguous.")
    
    # Sidebar with info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        ### Model Architecture
        - **Type**: Convolutional Neural Network (CNN)
        - **Input Size**: 224×224×3 (RGB)
        - **Layers**: 4 Conv blocks + Dense layers
        - **Parameters**: ~1.5M trainable
        
        ### Training Dataset
        - **Cat Images**: 2,554
        - **Dog Images**: 12,606
        - **Total**: 15,160 images
        - **Accuracy**: 86%
        
        ### Tips for Best Results
        - Use clear, well-lit images
        - Ensure the animal is the main subject
        - Avoid heavily filtered or edited images
        - JPG, PNG formats work best
        """)
        
        st.markdown("---")
        st.markdown("Made with ❤️ using TensorFlow & Streamlit")

if __name__ == "__main__":
    main()