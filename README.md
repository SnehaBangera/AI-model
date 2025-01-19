# AI Models Web Application

An interactive platform to leverage machine learning models for **Stock Price Prediction** and **Plant Disease Detection**, integrated with the latest news and sentiment analysis.

## Features

### Stock Price Prediction
- **LSTM-based Model**: Predicts the next 20 days of stock prices.
- **News Sentiment Analysis**: Integrates stock-related news and sentiment analysis via the Alpha Vantage API.
- **Buy/Sell/Hold Recommendations**: Actionable recommendations based on predicted stock trends.
- **Data Visualization**: Interactive charts to visualize historical stock prices and predictions.

### Plant Disease Detection
- **Leaf Image Upload**: Upload a plant leaf image to detect diseases.
- **Treatment Suggestions**: Get immediate feedback on diseases and recommended treatments.

### Stock Shortcuts for Indian and American Markets
- Easily select popular stocks using predefined ticker symbols.

#### Indian Stock Market (NSE & BSE)
- **Reliance Industries**: RELIANCE.NS
- **Tata Consultancy Services (TCS)**: TCS.NS
- **HDFC Bank**: HDFCBANK.NS
- **Infosys**: INFY.NS
- **ICICI Bank**: ICICIBANK.NS
- **Bajaj Finance**: BAJFINANCE.NS
- **Hindustan Unilever**: HINDUNILVR.NS
- **Larsen & Toubro**: LT.NS

#### American Stock Market (NASDAQ & NYSE)
- **Apple Inc.**: AAPL
- **Microsoft Corporation**: MSFT
- **Tesla Inc.**: TSLA
- **Amazon.com Inc.**: AMZN
- **Meta Platforms (Facebook)**: META
- **Alphabet Inc. (Google)**: GOOGL
- **NVIDIA Corporation**: NVDA
- **Netflix Inc.**: NFLX

## Technologies Used

- **Streamlit**: For building the interactive web application.
- **TensorFlow (Keras)**: For the LSTM-based model in stock price prediction.
- **PyTorch**: For training the plant disease detection model.
- **Yahoo Finance API**: For fetching historical stock data.
- **Alpha Vantage API**: For stock-related news and sentiment analysis.
- **OpenCV & TensorFlow**: For image processing and plant disease detection.
- **Python**: For data processing, machine learning, and web application logic.

## Setup Instructions

To run the application locally, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ai-models.git
```
2. Navigate to the Project Folder
```bash
cd ai-models
```
4. Create a Virtual Environment (optional but recommended)
```bash
python3 -m venv venv
```
5. Activate the Virtual Environment
macOS/Linux:
```bash
source venv/bin/activate
```
Windows:
```bash
venv\Scripts\activate
```
7. Install Required Dependencies
```bash
pip install -r requirements.txt
```
8. Add API Keys
Create a .env file in the root directory and add your API keys:
```bash
echo "ALPHA_VANTAGE_API_KEY=your_api_key" > .env
```
7. Run the Streamlit App
```bash
streamlit run app.py
```
8. Open Your Browser and Navigate to:
```bash
http://localhost:8501
```
Deployment
The application is deployed on Streamlit and can be accessed online at:
AI Models Web Application

How It Works
Fetch Stock Data: The app fetches historical stock data using the Yahoo Finance API.
Preprocess Data: Stock data is normalized and preprocessed for the LSTM model.
Train LSTM Model: The LSTM model is trained on historical data to predict the next 20 days of stock prices.
Stock Prediction: The model predicts the next 20 days of stock prices.
News Sentiment: The app fetches the latest news related to the stock via the Alpha Vantage API and performs sentiment analysis.
Plant Disease Detection: Upload a plant leaf image, and the model detects potential diseases and offers treatment suggestions.
Recommendations: For stocks, the app provides a recommendation to buy, sell, or hold based on predicted price changes.
Contributing
If you would like to contribute to this project, please follow these steps:

Fork the repository.
Create a new branch.
Submit a pull request with your proposed changes.
License
This project is licensed under the MIT License - see the LICENSE file for details.
