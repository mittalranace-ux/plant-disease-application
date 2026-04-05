import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Page config
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Disease Detection App")
st.write("Upload a leaf image and detect plant disease using Deep Learning")

# Load model (ensure file is in same folder OR use Drive download)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("plant_disease_resnet50.h5")
    return model

model = load_model()

# Class labels (👉 replace with your 38 classes)
class_names = [
    "Apple Scab",
    "Black Rot",
    "Cedar Apple Rust",
    "Healthy"
]

uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocessing
    img = image.resize((224, 224))
    img_array = np.array(img)

    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"🌱 Predicted Disease: {predicted_class}")
    st.info(f"🔬 Confidence: {confidence:.2f}%")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit + TensorFlow")
