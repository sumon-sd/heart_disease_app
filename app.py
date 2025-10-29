import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="❤️ Heart Disease Predictor", layout="wide")
st.title("❤️ Heart Disease Predictor")
st.markdown("Enter patient details below to predict the likelihood of heart disease.")

# --------------------------
# Helper function to load files safely
# --------------------------
def safe_load(filename):
    if os.path.exists(filename):
        return joblib.load(filename)
    else:
        st.error(f"❌ File not found: {filename}")
        return None

# Load model, scaler, and columns
model = safe_load("logistic_model.pkl")
scaler = safe_load("scaler.pkl")
model_columns = safe_load("model_columns.pkl")

if model is None or scaler is None or model_columns is None:
    st.stop()

# --------------------------
# User Inputs
# --------------------------
col1, col2 = st.columns(2)

with col1:
    thalach = st.number_input("Maximum Heart Rate Achieved (thalch)", 50, 250, 150)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type (cp)", ["atypical angina", "non-anginal", "typical angina"])
    restecg = st.selectbox("Resting ECG results", ["normal", "st-t abnormality"])
    slope = st.selectbox("Slope of ST segment", ["flat", "upsloping"])
    thal = st.selectbox("Thalassemia", ["normal", "reversable defect"])
    
with col2:
    dataset = st.selectbox("Dataset origin", ["Hungary", "Switzerland", "VA Long Beach"])

# --------------------------
# Prepare input as one-hot
# --------------------------
input_dict = {
    "thalch": thalach,
    "sex_Male": 1 if sex == "Male" else 0,
    "cp_atypical angina": 1 if cp=="atypical angina" else 0,
    "cp_non-anginal": 1 if cp=="non-anginal" else 0,
    "cp_typical angina": 1 if cp=="typical angina" else 0,
    "restecg_normal": 1 if restecg=="normal" else 0,
    "restecg_st-t abnormality": 1 if restecg=="st-t abnormality" else 0,
    "slope_flat": 1 if slope=="flat" else 0,
    "slope_upsloping": 1 if slope=="upsloping" else 0,
    "thal_normal": 1 if thal=="normal" else 0,
    "thal_reversable defect": 1 if thal=="reversable defect" else 0,
    "dataset_Hungary": 1 if dataset=="Hungary" else 0,
    "dataset_Switzerland": 1 if dataset=="Switzerland" else 0,
    "dataset_VA Long Beach": 1 if dataset=="VA Long Beach" else 0,
}

# Create DataFrame and align with model columns
input_df = pd.DataFrame([input_dict])
input_df = input_df.reindex(columns=model_columns, fill_value=0)

# Scale input
input_scaled = scaler.transform(input_df)

# --------------------------
# Prediction
# --------------------------
if st.button("Predict"):
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("### Prediction Result")
    if prediction[0] == 1:
        st.error(f"⚠️ The model predicts heart disease.")
    else:
        st.success(f"✅ The model predicts no heart disease.")

    st.markdown("### Heart Disease Probability")
    st.progress(float(probability))
    st.write(f"Probability: {probability*100:.2f}%")
