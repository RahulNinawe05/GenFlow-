# streamlit_sentiment_app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index
st.set_page_config(page_title="IMDB Sentiment Analyzer", layout="centered")
st.title("IMDB Movie Review Sentiment Analysis")

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load pre-trained model
@st.cache_resource
def load_sentiment_model():
    return load_model("simple_rnn_imdb.h5")

model = load_sentiment_model()

# Preprocessing functions
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    return sentiment, prediction[0][0]

# Streamlit UI
user_input = st.text_area("Enter your movie review here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        sentiment, score = predict_sentiment(user_input)
        st.subheader("Model Prediction")
        st.write(f"**Review Sentiment:** {sentiment}")
        st.write(f"**Confidence Score:** {score:.2f}")
