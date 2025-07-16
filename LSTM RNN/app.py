import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the trained model
model = tf.keras.models.load_model("next_word_lstm.h5")

# Load the tokenizer
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# Get max sequence length from model input shape
max_sequence_len = model.input_shape[1] + 1

# Prediction function
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return "Unknown"

# Streamlit UI
st.set_page_config(page_title="Next Word Predictor", layout="centered")
st.title("🔮 Next Word Prediction using LSTM")
st.markdown("Enter a sentence below and get the predicted next word.")

# User Input
user_input = st.text_input("Enter your sentence:", "")

# Predict button
if st.button("Predict Next Word"):
    if user_input.strip() == "":
        st.warning("Please enter a sentence to predict the next word.")
    else:
        predicted_word = predict_next_word(model, tokenizer, user_input, max_sequence_len)
        st.success(f"**Predicted Next Word:** `{predicted_word}`")
