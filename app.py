import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries

from src.fetch_data import fetch_data
from src.preprocess_data import preprocess_data
from src.build_model import build_model
from src.predict import predict_next_20_days
from src.news import fetch_stock_news_alpha_vantage
from src.recommendation import get_recommendation
st.markdown(
    """
    <style>
    /* Sidebar and styling */
    .css-1d391kg { background-color: #2C3E50; color: white; font-family: 'Helvetica Neue', sans-serif; border-radius: 10px; padding: 10px; }
    .css-1d391kg .stSidebar { background-color: #34495E; }
    .css-1d391kg .stButton { background-color: #E74C3C; color: black; border-radius: 5px; padding: 10px; width: 100%; }
    .css-1d391kg .stSidebar h1 { color: #E74C3C; font-size: 24px; font-weight: bold; }
    .css-1d391kg .stSidebar p { font-size: 16px; line-height: 1.5; }
    .css-1d391kg .stRadio > div { padding: 15px; font-size: 18px; font-weight: bold; color: white; }
    .css-1d391kg .stRadio > div:hover { background-color: #16A085; }
    .css-1d391kg .stSidebar i { font-size: 40px; color: #E74C3C; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True
)

# Sidebar Navigation with Custom Icon
st.sidebar.markdown(
    """
    <div style="display: flex; flex-direction: column; align-items: center; padding: 10px;">
        <i class="fas fa-chart-line"></i>
        <h1 style="color: white; font-size: 22px; font-weight: bold;">AI Suite</h1>
        <p style="color: white; font-size: 14px;">Stock Market Predictions & Plant Disease Detection</p>
    </div>
    """, unsafe_allow_html=True
)

# Load environment variables
load_dotenv()

# TensorFlow Model Prediction for Plant Disease
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Add batch dimension
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)

    # List of disease class names corresponding to the index
    class_names = [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 
        'Apple___healthy', 'Blueberry___healthy', 
        'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
        'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 
        'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
        'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 
        'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
        'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
        'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
    ]
    
    
    # Return the corresponding class name
    predicted_class_name = class_names[result_index]
    return predicted_class_name

# Sidebar Navigation
st.sidebar.title("📊 AI-Powered Dashboard")
app_mode = st.sidebar.radio(
    "Select a Feature",
    ["🚀Home", "📈Stock Market Prediction", "🌱Plant Disease Detection", "📚About"]
)

def home_page():
    st.header("🌱 AI-Powered Productivity Suite")
    image_path = "img.webp"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    Welcome to the **AI-Powered Productivity Suite** – your ultimate tool to leverage advanced **AI** and **machine learning** technologies for smarter decision-making.

    🚀 **Explore Cutting-Edge Features**:
    - **Stock Market Prediction**: Get insights on whether stock prices will rise or fall, based on historical trends and market data.
    - **Plant Disease Detection**: Quickly identify plant diseases using AI-powered image recognition.

    📊 **Stock Market Prediction**:
    In our stock market prediction feature, we predict the general trend of stock prices—whether they will go up or down—by analyzing historical data patterns.
    
    ⚠️ **Important Disclaimer**:
    - These predictions are based on past market behavior and trends, but they do not guarantee specific future values.
    - While our model can suggest potential price directions, stock prices can be volatile and unpredictable. External factors and market changes can cause fluctuations.
    - Always use predictions in conjunction with other research and analysis for informed decision-making.

    🌿 **Plant Disease Detection**:
    The plant disease recognition system helps farmers and plant enthusiasts to diagnose diseases in their plants. Upload a clear image of a plant leaf, and our system will use AI to detect potential diseases and suggest remedies.
    
    **Model Training Details**:
    - The plant disease detection model has been trained on a dataset of **70,000 plant images**, covering various common plant species and diseases. 
    - Due to **GPU limitations**, the model can only recognize the plants that are within the trained dataset. It may not accurately recognize plants that are not part of this dataset, especially those found on external websites or Google.
    - However, for plants included in the training set, the model offers high accuracy and reliable predictions.

    🔍 **Why Use This Suite?**
    - **Real-Time Data**: Stay updated with market trends and plant health.
    - **Smart Predictions**: Make better decisions by understanding future trends.
    - **User-Friendly Interface**: Access all features with just a few clicks.
    
    Embrace the power of AI to make more informed, data-driven decisions that lead to better outcomes in both business and agriculture.
    """)




# Stock Market Prediction Page
def stock_market_prediction():
    st.header("📈 Stock Market Prediction")
    stock_symbol = st.text_input("Enter Stock Symbol", "AAPL")
    
    if st.button("Fetch and Predict"):
        with st.spinner("Fetching data..."):
            data = fetch_data(stock_symbol)
            if data is None:
                return
        
        st.success("Data fetched successfully.")
        st.line_chart(data["Close"])
        
        with st.spinner("Processing data..."):
            scaled_data, scaler = preprocess_data(data)
            training_size = int(len(scaled_data) * 0.8)
            train_data, test_data = scaled_data[:training_size], scaled_data[training_size-60:]
        
        with st.spinner("Building model..."):
            model = build_model()
            model.fit(train_data.reshape(train_data.shape[0], train_data.shape[1], 1), train_data, epochs=50, batch_size=32, verbose=0)
        st.success("Model trained successfully.")
        
        with st.spinner("Predicting..."):
            predictions = model.predict(test_data.reshape(test_data.shape[0], test_data.shape[1], 1))
            predictions = scaler.inverse_transform(predictions)
            next_20_days = predict_next_20_days(scaled_data, model, scaler)
        
        last_date = data.index[-1]
        dates = pd.date_range(last_date, periods=21, freq='D')[1:]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(dates, next_20_days, label="Predicted Price", linestyle='--', color='red')
        plt.xticks(rotation=45)
        ax.set_title("Next 20 Days Predicted Stock Prices")
        ax.set_xlabel("Date")
        ax.set_ylabel("Stock Price")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        
        st.subheader("Present Day Value")
        st.write(f"{data['Close'].iloc[-1]:.2f}")
        
        st.subheader("Future Price Range")
        st.write(f"{next_20_days[-1][0]:.2f}")
        
        recommendation = get_recommendation(data['Close'].iloc[-1], next_20_days[-1][0])
        st.subheader(f"Recommendation: {recommendation}")
        
        news = fetch_stock_news_alpha_vantage(stock_symbol)
        st.subheader("Latest News")
        for article in news:
            st.markdown(f"### [{article['title']}]({article['url']})")
            st.write(f"{article['description']}")



import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries

def plant_disease_model():
    st.header("🌿 Plant Disease Detection")

    st.markdown(
        """
        **Model Information:**
        - 📸 The model is trained on 70,000 images from this [dataset](https://github.com/vam-luffy/dataSet/tree/main/train). 
        - ⚠️ Due to GPU limitations, further training was not performed.
        - ✅ The model provides good accuracy and delivers reliable results for the images within this dataset.
        """
    )

    test_image = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        st.image(test_image, use_container_width=True, caption="Uploaded Image")

        if st.button("Predict"):
            with st.spinner("Processing image..."):
                try:
                    result_class_name = model_prediction(test_image)  # Assume this function exists
                    st.success(f"✅ Prediction: {result_class_name}")
                    st.session_state.prediction_result = result_class_name
                except Exception as e:
                    st.error(f"Error: {e}")

        if "prediction_result" in st.session_state:
            st.write(f"✅ Prediction: {st.session_state.prediction_result}")

        if st.checkbox("Show Model Explanation (LIME)"):
            with st.spinner("Generating explanation..."):
                try:
                    model = tf.keras.models.load_model("trained_model.h5")
                    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
                    input_arr = tf.keras.preprocessing.image.img_to_array(image)
                    input_arr = np.array([input_arr])
                    prediction = model.predict(input_arr)
                    explainer = lime_image.LimeImageExplainer()

                    explanation = explainer.explain_instance(
                        input_arr[0].astype('double'),
                        model.predict,
                        top_labels=1,
                        hide_color=0,
                        num_samples=1000
                    )

                    temp, mask = explanation.get_image_and_mask(
                        explanation.top_labels[0],
                        positive_only=True,
                        num_features=5,
                        hide_rest=False
                    )

                    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                    ax[0].imshow(mark_boundaries(temp / 255.0, mask))
                    ax[0].set_title("LIME Explanation")
                    ax[1].imshow(image)
                    ax[1].set_title("Original Image")
                    for a in ax:
                        a.axis("off")
                    st.pyplot(fig)

                    # Display disease info after LIME explanation
                    if "prediction_result" in st.session_state:
                        display_disease_info(st.session_state.prediction_result)


                except Exception as e:
                    st.error(f"An error occurred while generating explanation: {e}")


disease_info = {
    'Apple___Apple_scab': {
        'description': 'A fungal disease caused by Venturia inaequalis. It affects leaves and fruits, causing dark, scabby lesions.',
        'causes': 'Cool, wet conditions favor the growth of fungal spores.',
        'symptoms': 'Velvety, olive-green to black spots on leaves and fruit.',
        'treatment': 'Apply fungicides, ensure proper pruning for airflow, and remove infected debris.'
    },
    'Apple___Black_rot': {
        'description': 'A fungal disease caused by Botryosphaeria obtusa leading to fruit rot and leaf spotting.',
        'causes': 'Fungal infection thriving in warm, humid conditions.',
        'symptoms': 'Concentric rings of decay on fruit, yellowing, and curling leaves.',
        'treatment': 'Remove infected fruit, prune dead branches, and apply fungicides.'
    },
    'Apple___Cedar_apple_rust': {
        'description': 'A fungal disease caused by Gymnosporangium juniperi-virginianae requiring both apple and juniper trees to complete its lifecycle.',
        'causes': 'Fungal spores from nearby juniper trees.',
        'symptoms': 'Orange, gelatinous galls on juniper and yellow spots on apple leaves.',
        'treatment': 'Remove juniper galls and apply fungicides to apple trees.'
    },
    'Apple___healthy': {
        'description': 'No disease detected. The plant appears healthy.',
        'causes': 'Proper care and optimal growth conditions.',
        'symptoms': 'Healthy, green leaves and well-developed fruit.',
        'treatment': 'Continue good agricultural practices.'
    },
    'Blueberry___healthy': {
        'description': 'No disease detected. The plant appears healthy.',
        'causes': 'Optimal growing conditions and proper care.',
        'symptoms': 'Healthy, vibrant foliage and fruit.',
        'treatment': 'Maintain proper watering and fertilization.'
    },
    'Cherry_(including_sour)_Powdery_mildew': {
        'description': 'A fungal disease caused by Podosphaera clandestina, forming a white, powdery coating.',
        'causes': 'High humidity and poor air circulation.',
        'symptoms': 'White, powdery fungal growth on leaves and fruit.',
        'treatment': 'Apply fungicides and improve air circulation through pruning.'
    },
    'Cherry_(including_sour)_healthy': {
        'description': 'No disease detected. The plant appears healthy.',
        'causes': 'Proper care and optimal growing conditions.',
        'symptoms': 'Healthy leaves and well-developed fruit.',
        'treatment': 'Maintain good agricultural practices.'
    },
    'Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot': {
        'description': 'A fungal disease caused by Cercospora zeae-maydis, forming grayish lesions on leaves.',
        'causes': 'Warm, humid conditions favor fungal spore development.',
        'symptoms': 'Narrow, elongated gray lesions on leaves.',
        'treatment': 'Rotate crops, remove debris, and use fungicides if necessary.'
    },
    'Corn_(maize)Common_rust': {
        'description': 'A fungal disease caused by Puccinia sorghi, forming rust-colored pustules.',
        'causes': 'Cool, moist conditions favor fungal infection.',
        'symptoms': 'Rusty, orange pustules on leaves.',
        'treatment': 'Plant resistant varieties and apply fungicides.'
    },
    'Corn_(maize)_Northern_Leaf_Blight': {
        'description': 'A fungal disease caused by Exserohilum turcicum, forming long, elliptical lesions on leaves.',
        'causes': 'Prolonged leaf wetness in warm conditions.',
        'symptoms': 'Large, tan, spindle-shaped lesions.',
        'treatment': 'Use resistant hybrids, rotate crops, and apply fungicides.'
    },
    'Corn_(maize)_healthy': {
        'description': 'No disease detected. The plant appears healthy.',
        'causes': 'Proper care and management.',
        'symptoms': 'Vibrant, green leaves and robust growth.',
        'treatment': 'Continue optimal crop care practices.'
    },
    'Grape___Black_rot': {
        'description': 'A fungal disease caused by Guignardia bidwellii, affecting leaves, stems, and fruit.',
        'causes': 'Wet, warm conditions encourage fungal growth.',
        'symptoms': 'Black, circular lesions with concentric rings.',
        'treatment': 'Remove infected parts and apply fungicides.'
    },
    'Grape__Esca(Black_Measles)': {
        'description': 'A fungal disease complex caused by various pathogens, leading to black streaks on fruit and leaves.',
        'causes': 'Infection through wounds or pruning cuts.',
        'symptoms': 'Brownish spots on leaves and dark streaks on fruit.',
        'treatment': 'Use clean tools, apply fungicides, and avoid overwatering.'
    },
    'Grape__Leaf_blight(Isariopsis_Leaf_Spot)': {
        'description': 'A fungal disease caused by Isariopsis clavispora, forming dark spots on leaves.',
        'causes': 'High humidity and poor air circulation.',
        'symptoms': 'Small, dark, angular lesions on leaves.',
        'treatment': 'Prune to improve airflow and apply fungicides.'
    },
    'Grape___healthy': {
        'description': 'No disease detected. The plant appears healthy.',
        'causes': 'Proper care and optimal growing conditions.',
        'symptoms': 'Healthy, green leaves and well-formed fruit.',
        'treatment': 'Maintain good vineyard practices.'
    },
    'Orange__Haunglongbing(Citrus_greening)': {
        'description': 'A bacterial disease caused by Candidatus Liberibacter species, transmitted by psyllid insects.',
        'causes': 'Infection through psyllid insects.',
        'symptoms': 'Yellowing of veins, misshapen fruit, and tree decline.',
        'treatment': 'Control psyllids and remove infected trees.'
    },
    'Peach___Bacterial_spot': {
        'description': 'A bacterial disease caused by Xanthomonas campestris pv. pruni.',
        'causes': 'Warm, wet conditions encourage bacterial spread.',
        'symptoms': 'Dark, water-soaked spots on leaves and fruit.',
        'treatment': 'Use copper sprays and plant resistant varieties.'
    },
    'Peach___healthy': {
        'description': 'No disease detected. The plant appears healthy.',
        'causes': 'Proper care and management.',
        'symptoms': 'Healthy foliage and fruit.',
        'treatment': 'Continue optimal care practices.'
    },
    'Pepper,bell__Bacterial_spot': {
        'description': 'A bacterial disease caused by Xanthomonas campestris pv. vesicatoria.',
        'causes': 'Warm, humid conditions.',
        'symptoms': 'Water-soaked spots on leaves and fruit.',
        'treatment': 'Apply copper sprays and ensure proper spacing.'
    },
    'Pepper,bell__healthy': {
        'description': 'No disease detected. The plant appears healthy.',
        'causes': 'Proper care and optimal growing conditions.',
        'symptoms': 'Healthy, green leaves and vibrant fruit.',
        'treatment': 'Maintain good agricultural practices.'
    },
    'Potato___Early_blight': {
        'description': 'A fungal disease caused by Alternaria solani that affects the leaves, stems, and tubers of potatoes.',
        'causes': 'Warm, wet conditions and poor crop rotation.',
        'symptoms': 'Concentric, dark lesions on leaves with yellow halos.',
        'treatment': 'Apply fungicides and rotate crops to prevent reinfection.'
    },
    'Potato___Late_blight': {
        'description': 'A devastating fungal disease caused by Phytophthora infestans, known for causing rapid plant collapse.',
        'causes': 'Cool, wet conditions, especially during rainy seasons.',
        'symptoms': 'Dark, water-soaked lesions on leaves and stems.',
        'treatment': 'Apply fungicides, ensure good drainage, and remove infected plants.'
    },
    'Potato___healthy': {
        'description': 'No disease detected. The plant appears healthy.',
        'causes': 'Proper care, good soil conditions, and optimal growing temperatures.',
        'symptoms': 'Healthy, vigorous growth with no visible disease symptoms.',
        'treatment': 'Maintain optimal agricultural practices and crop rotation.'
    },
    'Raspberry___healthy': {
        'description': 'No disease detected. The plant appears healthy.',
        'causes': 'Proper care, optimal watering, and good soil conditions.',
        'symptoms': 'Vibrant green leaves and healthy fruit development.',
        'treatment': 'Continue with proper care and regular monitoring.'
    },
    'Soybean___healthy': {
        'description': 'No disease detected. The plant appears healthy.',
        'causes': 'Proper care, ideal growing conditions, and pest management.',
        'symptoms': 'Healthy, green leaves with no signs of disease.',
        'treatment': 'Maintain optimal conditions for growth and pest control.'
    },
    'Squash___Powdery_mildew': {
        'description': 'A fungal disease caused by Erysiphe cichoracearum that creates a white powdery coating on leaves.',
        'causes': 'Dry conditions with high humidity, often exacerbated by poor air circulation.',
        'symptoms': 'White, powdery fungal growth on the upper side of leaves.',
        'treatment': 'Apply fungicides and improve airflow through the plant canopy.'
    },
    'Strawberry___Leaf_scorch': {
        'description': 'A bacterial disease caused by Xanthomonas fragariae, leading to leaf damage.',
        'causes': 'Bacterial infection during wet conditions.',
        'symptoms': 'Brown, necrotic spots on leaves, often with a yellow halo.',
        'treatment': 'Use copper-based bactericides and practice good sanitation.'
    },
    'Strawberry___healthy': {
        'description': 'No disease detected. The plant appears healthy.',
        'causes': 'Proper care and cultivation with optimal watering and soil conditions.',
        'symptoms': 'Healthy, green leaves and good fruit development.',
        'treatment': 'Continue good agricultural practices and monitor for pests.'
    },
    'Tomato___Bacterial_spot': {
        'description': 'A bacterial disease caused by Xanthomonas campestris pv. vesicatoria that leads to leaf spotting and fruit lesions.',
        'causes': 'Wet, humid conditions and poor plant spacing.',
        'symptoms': 'Small, water-soaked spots that enlarge and turn brown on leaves and fruit.',
        'treatment': 'Apply copper-based bactericides and practice proper spacing and watering.'
    },
    'Tomato___Early_blight': {
        'description': 'A fungal disease caused by Alternaria solani, affecting leaves, stems, and fruits.',
        'causes': 'Wet, humid conditions and poor crop rotation.',
        'symptoms': 'Dark, concentric lesions on leaves with a yellow halo.',
        'treatment': 'Apply fungicides and rotate crops to reduce the risk of recurrence.'
    },
    'Tomato___Late_blight': {
        'description': 'A serious fungal disease caused by Phytophthora infestans, affecting both leaves and fruit.',
        'causes': 'Cool, wet weather that favors fungal growth.',
        'symptoms': 'Water-soaked lesions on leaves, followed by rapid plant collapse.',
        'treatment': 'Apply fungicides, remove infected plants, and improve air circulation.'
    },
    'Tomato___Leaf_Mold': {
        'description': 'A fungal disease caused by Passalora fulva, primarily affecting the leaves of tomatoes.',
        'causes': 'High humidity and poor airflow around plants.',
        'symptoms': 'Yellowing and moldy growth on the underside of leaves.',
        'treatment': 'Improve ventilation and apply fungicides.'
    },
    'Tomato___Septoria_leaf_spot': {
        'description': 'A fungal disease caused by Septoria lycopersici, forming round lesions with dark borders on tomato leaves.',
        'causes': 'Wet, humid weather and infected plant debris.',
        'symptoms': 'Small, round lesions with dark borders on leaves.',
        'treatment': 'Remove infected leaves, apply fungicides, and rotate crops.'
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'description': 'A pest-related issue caused by Tetranychus urticae, leading to yellowing and speckling on leaves.',
        'causes': 'Dry, dusty conditions with poor plant health.',
        'symptoms': 'Speckled yellow patches on leaves, often with a web-like appearance.',
        'treatment': 'Use miticides or insecticidal soap, and ensure proper watering.'
    },
    'Tomato___Target_Spot': {
        'description': 'A fungal disease caused by Corynespora cassiicola that forms circular, dark lesions on tomato leaves.',
        'causes': 'High humidity and warm temperatures favor the disease.',
        'symptoms': 'Round, dark lesions with a lighter center on leaves.',
        'treatment': 'Use fungicides and avoid overcrowding of plants.'
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'description': 'A viral disease caused by the Tomato yellow leaf curl virus (TYLCV), transmitted by whiteflies.',
        'causes': 'Whitefly transmission of the virus.',
        'symptoms': 'Curling, yellowing of leaves, stunted growth, and poor fruit set.',
        'treatment': 'Control whiteflies, remove infected plants, and use resistant varieties.'
    },
    'Tomato___Tomato_mosaic_virus': {
        'description': 'A viral disease caused by Tomato mosaic virus (ToMV), leading to mottling and distortion of leaves and fruit.',
        'causes': 'Contact with infected plant debris or mechanical injury.',
        'symptoms': 'Mosaic pattern on leaves, leaf curling, and fruit distortion.',
        'treatment': 'Remove infected plants, practice good sanitation, and use resistant varieties.'
    },
    'Tomato___healthy': {
        'description': 'No disease detected. The plant appears healthy.',
        'causes': 'Proper care, optimal watering, and pest control.',
        'symptoms': 'Healthy, green foliage and strong fruit development.',
        'treatment': 'Maintain good agricultural practices and monitor for pests.'
    }
}
    

def display_disease_info(class_name):
    info = disease_info.get(class_name)
    if info:
        st.subheader("Disease Information:")
        for key, value in info.items():
            st.write(f"**{key.capitalize()}:** {value}")


def about_page():
    st.header("📚 About This Application")
    st.write("""
    Welcome to the **AI-Powered Productivity Suite**! This application is a powerful tool that combines advanced machine learning and **artificial intelligence** to help you in two key areas: **Stock Market Prediction** and **Plant Disease Detection**.

    🌍 **Our Vision**:
    Our goal is to empower users with the ability to make smarter decisions through AI-driven insights. Whether you are managing investments or taking care of your plants, we provide tools to help you optimize your choices, backed by data and machine learning models.

    🧑‍💻 **Key Features**:
    - **Stock Market Prediction**: Utilize AI to predict market trends and help you understand whether stock prices are likely to go up or down. By analyzing historical market data, we help forecast future stock movements, though these are based on patterns, and market fluctuations can vary.
    - **Plant Disease Detection**: Upload images of plant leaves to diagnose potential diseases. With the power of deep learning, our system identifies various plant diseases, enabling better agricultural practices and helping you take timely actions to protect your plants.

    🌱 **Why Choose This Suite?**
    - **AI-Driven Insights**: Harness the power of artificial intelligence to make informed decisions in both financial markets and agriculture.
    - **User-Friendly Interface**: Simple, easy-to-navigate interface designed for both beginners and experts.
    - **Real-Time Predictions**: Receive up-to-date predictions and analyses based on the latest available data and trends.

    🔧 **How It Works**:
    - **Stock Market Prediction**: Our system uses historical stock data, applies machine learning models, and generates predictions based on identified trends. While predictions provide a likely direction (up or down), they should be interpreted with caution, as markets are inherently volatile.
    - **Plant Disease Detection**: By using computer vision techniques and pre-trained deep learning models, the system analyzes images of plant leaves to identify symptoms of diseases, suggesting potential remedies.

    🚀 **Future Developments**:
    - We're constantly working to enhance the capabilities of both stock market prediction and plant disease detection. Future updates may include improved models, additional plant species, more in-depth stock analysis tools, and even personalized recommendations based on your usage.

    🏆 **Empowering Users**: Whether you’re an investor looking to make informed stock market decisions, a farmer trying to safeguard crops, or just someone curious about the power of AI, this suite is here to assist you every step of the way.

    Thank you for choosing the **AI-Powered Productivity Suite** – **AI** for smarter living!
    """)


# Main Function
if app_mode == "🚀Home":
    home_page()
elif app_mode == "📈Stock Market Prediction":
    stock_market_prediction()
elif app_mode == "🌱Plant Disease Detection":
    plant_disease_model()
elif app_mode == "📚About":
    about_page()
    
hide_footer_style = """
    <style>
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_footer_style, unsafe_allow_html=True)
