import streamlit as st
import numpy as np
import os
from PIL import Image
import datetime
import requests
import tensorflow as tf

# ---------------- MODEL PREDICTION FUNCTION ----------------
def model_predict(image):
    try:
        import tensorflow as tf
        import cv2
        
        # Load model from cloud directory (change this to where you have your model)
        model_path = './models/plant_disease_model.h5'
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error("Error loading model: " + str(e))
        return None, None

    img = cv2.imread(image)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = img.reshape(1, 224, 224, 3)

    prediction_probs = model.predict(img)[0]
    predicted_class_index = np.argmax(prediction_probs)
    confidence = prediction_probs[predicted_class_index] * 100

    return predicted_class_index, confidence

# ---------------- WEATHER FUNCTION ----------------
def show_weather():
    st.subheader("🌤️ Real-Time Weather")
    city = st.text_input("Enter your city:", "Delhi")

    if city:
        # Detailed format: Location, weather, temp, humidity, wind
        url = f"https://wttr.in/{city}?format=%l:+%c+%t\nHumidity:+%h\nWind:+%w\nCondition:+%C"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                weather_lines = response.text.strip().split("\n")

                # Display the main info
                st.success(f"📍 {weather_lines[0]}")  # City + Emoji + Temp
                for line in weather_lines[1:]:

                    if "Humidity" in line:
                        st.info(f"💧 {line}")
                    elif "Wind" in line:
                        st.info(f"🌬️ {line}")
                    elif "Condition" in line:
                        condition = line.split(":")[-1].strip().lower()

                        st.info(f"📋 Weather Condition: {condition.capitalize()}")

                        # Smart tips
                        if "rain" in condition:
                            st.warning("🌧️ It's rainy. Protect your plants from excess water.")
                        elif "sun" in condition or "clear" in condition:
                            st.success("☀️ Sunny day! Great for photosynthesis.")
                        elif "cloud" in condition:
                            st.info("☁️ Cloudy. Monitor humidity for potential fungal growth.")
                        elif "storm" in condition:
                            st.error("🌩️ Storm alert! Ensure crops are secured.")
                        elif "fog" in condition:
                            st.info("🌫️ Foggy. Reduce watering to avoid fungal infections.")
                        else:
                            st.info("🌤️ Mild weather. A good time to inspect plant health.")
            else:
                st.error("⚠️ Weather service unavailable. Please try again later.")
        except Exception as e:
            st.error(f"❌ Error fetching weather data: {e}")

# ---------------- SIDEBAR NAVIGATION ----------------
st.sidebar.title("🌿 Plant Disease Detection")
app_mode = st.sidebar.selectbox("Go to", [
    "🏠 Home", "🔍 Disease Recognition", "🖼️ Gallery", 
    "ℹ️ About", "📞 Contact", "❓ Help", "📊 Model Performance"
])

# ---------------- PAGE FUNCTIONS ----------------
def show_home():
    st.markdown("<h1 style='text-align: center; color: green;'>🌿 Plant Disease Detection System</h1>", unsafe_allow_html=True)

    st.markdown("""
    Welcome to the **Plant Disease Detection System** — an intelligent tool powered by deep learning that helps you detect diseases in plants just by uploading a photo of a leaf.
    ---  
    """)
    st.subheader("🛠️ How It Works")
    st.markdown("""
    1. 📷 Upload a clear photo of the leaf.
    2. 🤖 Our AI model analyzes the image.
    3. 🌾 You get the predicted disease and confidence score.
    4. 🩺 Use the results to take timely action for your crops.
    """)
    st.subheader("✨ Features")
    st.markdown("""
    - 🌱 Detects 38+ common plant diseases  
    - 📊 Provides accurate predictions with confidence levels  
    - 🖼️ Simple interface with drag-and-drop image upload  
    - ⚡ Fast and responsive – results in seconds!  
    - 🌐 Accessible from desktop and mobile  
    """)
    show_testimonials()
    show_weather()

def show_disease_recognition():
    st.title("🔍 Disease Recognition")
    app_mode = st.selectbox("Choose a feature", ["Upload Image"])  # Live capture removed for simplicity

    if app_mode == "Upload Image":
        uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            save_path = os.path.join(os.getcwd(), uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

            if st.button("🔍 Predict"):
                with st.spinner("Analyzing Image..."):
                    result_index, confidence = model_predict(save_path)

                if result_index is not None:
                    class_names = [ 
                        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
                        'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
                        'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                        'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                        'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                        'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                        'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                        'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
                        'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                        'Tomato___healthy'
                    ]
                    predicted_class = class_names[result_index]
                    st.success(f"🌾 Prediction: **{predicted_class}**")
                    st.info(f"📊 Confidence: {confidence:.2f}%")
                    show_disease_info(predicted_class)

def show_disease_info(disease_name):
    info = {
        'Apple___Apple_scab': 'Apple Scab is a fungal disease...',
        'Tomato___Early_blight': 'Early Blight on tomatoes is caused by a fungal pathogen...',
        # Add more disease info here
    }
    st.subheader(f"🦠 Disease Information for {disease_name}")
    st.write(info.get(disease_name, "No information available for this disease."))

def show_treatment_tips(disease_name):
    tips = {
        'Apple___Apple_scab': "🧪 Apply fungicides like Captan or Mancozeb. 🍃 Prune infected leaves.",
        'Tomato___Early_blight': "🧴 Use copper-based sprays. 🌾 Practice crop rotation.",
        # Add more...
    }
    st.subheader("💊 Suggested Treatment")
    st.info(tips.get(disease_name, "Treatment info coming soon!")) 
    
def show_gallery():
    st.title("🖼️ Leaf Gallery")
    st.write("Here are some sample leaves and diseases for reference.")
    gallery_folder = "sample_gallery"
    image_captions = {
        "apple_healthy.jpg": "Apple Leaf – Healthy",
        "corn_blight.jpg": "Corn – Northern Leaf Blight",
        "tomato_leaf_mold.jpg": "Tomato – Leaf Mold",
