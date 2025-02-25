import streamlit as st
import joblib
import time
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import tensorflow as tf

# Load models and vectorizers
nb_model = joblib.load('models/naive_bayes_model.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')

# Initialize VADER
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Load LSTM model and tokenizer
lstm_model = tf.keras.models.load_model('models/lstm_model-Resampling.h5')
lstm_tokenizer = joblib.load('models/tokenizer-Resampling.pkl')

# Load XGBoost Model
xgb_model = joblib.load('models/xgboost_sentiment_model_tune.pkl')
tfidf_vectorizer_xgb = joblib.load("models/xgboost_tfidf_vectorizer_tune.pkl")
# Sentiment Mapping
SENTIMENT_LABELS = {0: "Negative 😞", 1: "Neutral 😐", 2: "Positive 😊"}

# Streamlit UI

st.markdown("<h2 style='text-align: center;'>Real-Time Sentiment Analysis</h2>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Enter text to analyze sentiment using different models</h4>", unsafe_allow_html=True)

# User input
user_input = st.text_area("", "", placeholder="Type your sentence here...")

if st.button("Analyze Sentiment"):
    if user_input:
        st.write("### Results from Different Models:")

        # VADER Model
        vader_scores = sia.polarity_scores(user_input)
        if vader_scores['compound'] >= 0.05:
            vader_sentiment = "Positive 😊"
        elif vader_scores['compound'] <= -0.05:
            vader_sentiment = "Negative 😞"
        else:
            vader_sentiment = "Neutral 😐"
        st.write(f"**VADER:** {vader_sentiment}")

        # Naïve Bayes Model
        text_vectorized = tfidf.transform([user_input])
        nb_prediction = nb_model.predict(text_vectorized)[0]  # Ensure output is an integer label
        nb_sentiment = SENTIMENT_LABELS.get(nb_prediction, "Unknown")
        st.write(f"**Naïve Bayes:** {nb_sentiment}")

        ### **XGBoost Model**
        text_vectorized_xgb = tfidf_vectorizer_xgb.transform([user_input])
        if text_vectorized_xgb.shape[1] != xgb_model.n_features_in_:
            raise ValueError(
                f"Feature shape mismatch: expected {xgb_model.n_features_in_}, got {text_vectorized_xgb.shape[1]}"
            )
        xgb_prediction = xgb_model.predict(text_vectorized_xgb)[0]
        xgb_sentiment = SENTIMENT_LABELS.get(xgb_prediction, "Unknown")
        st.write(f"**XGBoost:** {xgb_sentiment}")

        # LSTM Model
        lstm_sequence = lstm_tokenizer.texts_to_sequences([user_input])
        lstm_padded = tf.keras.preprocessing.sequence.pad_sequences(lstm_sequence, maxlen=100)
        lstm_prediction = lstm_model.predict(lstm_padded)[0]  # Assuming the output is a probability distribution

        if lstm_prediction[0] > 0.6:
            lstm_sentiment = "Positive 😊"
        elif lstm_prediction[0] < 0.4:
            lstm_sentiment = "Negative 😞"
        else:
            lstm_sentiment = "Neutral 😐"

        st.write(f"**LSTM:** {lstm_sentiment}")




    else:
        st.warning("Please enter text to analyze.")

        # streamlit run main.py