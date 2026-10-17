import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

st.title("Heart Disease Prediction System")
st.write("Enter patient details below to predict the risk of heart disease.")

# Fit LabelEncoders on the exact same categories used in the notebook
sex_encoder = LabelEncoder().fit(["Male", "Female"])
cp_encoder = LabelEncoder().fit(["Typical", "Atypical", "NonAnginal", "Asymptomatic"])
fbs_encoder = LabelEncoder().fit(["High", "Normal"])
exang_encoder = LabelEncoder().fit(["Yes", "No"])

# Load the trained Heart Disease SVC model
model = pickle.load(open("heart_svc_trained_model.pkl", "rb"))

# User input dropdowns for the 4 attributes
sex_input = st.selectbox("Sex", ["Male", "Female"])
cp_input = st.selectbox("Chest Pain Type", ["Typical", "Atypical", "NonAnginal", "Asymptomatic"])
fbs_input = st.selectbox("Fasting Blood Sugar (> 120 mg/dl)", ["Normal", "High"])
exang_input = st.selectbox("Exercise Induced Angina", ["No", "Yes"])

if st.button("Predict"):
    encoded_input = pd.DataFrame({
        "Sex": sex_encoder.transform([sex_input]),
        "ChestPain": cp_encoder.transform([cp_input]),
        "FastingBS": fbs_encoder.transform([fbs_input]),
        "ExerciseAngina": exang_encoder.transform([exang_input])
    })
    
    result = model.predict(encoded_input)[0]
    
    if result == 1:
        st.error("Prediction: HEART DISEASE DETECTED")
    else:
        st.success("Prediction: NO HEART DISEASE")