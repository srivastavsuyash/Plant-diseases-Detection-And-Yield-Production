import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("🌿 Crop Disease Detection using Deep Learning")

# Load trained model
model = tf.keras.models.load_model("crop_disease_model.h5")

# Define class names (adjust to your dataset)
class_names = ['Healthy', 'Tomato Leaf Blight', 'Potato Late Blight', 'Leaf Spot']

uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = round(100 * np.max(prediction), 2)

    st.success(f"Predicted Disease: **{class_names[class_index]}** ({confidence}%)")
