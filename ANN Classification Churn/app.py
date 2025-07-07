import streamlit as st

st.set_page_config(page_title="Customer Churn ANN", layout="centered")
st.title("Customer Churn Prediction with ANN")


import os
import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

base_path = os.path.dirname(__file__)

model = tf.keras.models.load_model(os.path.join(base_path, 'model.h5'))

with open(os.path.join(base_path, 'label_encoder_gender.pkl'), 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open(os.path.join(base_path, 'onehot_encoder_geo.pkl'), 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open(os.path.join(base_path, 'scaler.pkl'), 'rb') as file:
    scaler = pickle.load(file)

st.title('Customer Churn Prediction')

geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credits_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

gender_encoded = label_encoder_gender.transform([gender])[0]

input_data = pd.DataFrame({
    'CreditScore': [credits_score],
    'Gender': [gender_encoded],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out())

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df.reset_index(drop=True)], axis=1)

scaled_input_data = scaler.transform(input_data)

prediction = model.predict(scaled_input_data)
prediction_prob = prediction[0][0]

st.write(f'Churn Probablity : {prediction_prob: .2f}')

if prediction_prob > 0.5:
    st.error('The Customer is likely to Churn.')
else:
    st.success('The Customer is not likely to Churn.')
