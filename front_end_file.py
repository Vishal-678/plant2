import streamlit as st
import numpy as np
import os
import cv2
import tensorflow as tf
from PIL import Image

# Load the model once
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

model = load_model()

# Prediction function
def model_predict(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = img.reshape(1, 224, 224, 3)
    prediction = np.argmax(model.predict(img), axis=-1)[0]
    return prediction

# Class labels
class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
              'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
              'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
              'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
              'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
              'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
              'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
              'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
              'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
              'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
              'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
              'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
              'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
              'Tomato___healthy']

# Sidebar
st.sidebar.title("🌿 Plant Disease Detection")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Home Page
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture 🌾</h1>", unsafe_allow_html=True)

# Disease Recognition Page
elif app_mode == "DISEASE RECOGNITION":
    st.header("Upload a Plant Leaf Image 🌱")
    test_image = st.file_uploader("Choose an Image", type=["jpg", "png", "jpeg"])

    if test_image is not None:
        image_path = os.path.join("temp", test_image.name)
        os.makedirs("temp", exist_ok=True)
        with open(image_path, "wb") as f:
            f.write(test_image.getbuffer())
        st.image(test_image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            st.snow()
            st.write("Model Prediction in progress...")
            try:
                result_index = model_predict(image_path)
                prediction = class_name[result_index]
                st.success(f"✅ The leaf is classified as: **{prediction}**")
            except Exception as e:
                st.error(f"⚠️ Prediction failed: {e}")
