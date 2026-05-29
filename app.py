import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

from markdown_it.rules_inline import balance_pairs
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

## loading the trained model

model = tf.keras.models.load_model('model.h5')

## load the encoder and scaler

with open("lable_encoder_gender.pkl", "rb") as f:
    label_encoder_gender= pickle.load(f)
with open("oneHot_encoder_geo.pkl", "rb") as f:
    oneHot_encoder_geo = pickle.load(f)

with open("scaler.pickle", "rb") as f:
    scaler = pickle.load(f)

## streamlit app

st.title("Customer Churn Prediction")

#user Input

geography= st.selectbox("Geography", oneHot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 20, 100 )
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.number_input("Tenure",0,10)
num_of_products = st.slider("Number of Products",1,4)
has_cr_card = st.selectbox("Has Credit Card",[0,1])
is_active_member = st.selectbox("Is Active",[0,1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

## encoding the one_hot geo

geo_encoded = oneHot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=oneHot_encoder_geo.get_feature_names_out(['Geography']))

## combine
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# scale the data
input_data_scaled = scaler.transform(input_data)

# prediction
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]
st.write("Churn Probability",prediction_prob)
if prediction_prob > 0.5:
    st.write("Customer is likely to Churn")
else:
    st.write("Customer is not likely to Churn")