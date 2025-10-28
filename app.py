import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

# -----------------------------
# Load model, scaler, columns
# -----------------------------
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")
with open("model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# Page config
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")
st.markdown("<h1 style='text-align: center; color: red;'>❤️ Heart Disease Predictor</h1>", unsafe_allow_html=True)
st.markdown("Predict your heart disease risk with probability and visualize your risk score.")

# -----------------------------
# Input fields
# -----------------------------
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 1, 120, 30)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["typical angina","atypical angina","non-anginal","asymptomatic"])
    trestbps = st.number_input("Resting BP", 50, 250, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes","No"])
    restecg = st.selectbox("Resting ECG", ["normal","ST-T abnormality","left ventricular hypertrophy"])
with col2:
    thalach = st.number_input("Max Heart Rate", 60, 250, 150)
    exang = st.selectbox("Exercise Angina", ["Yes","No"])
    oldpeak = st.number_input("ST depression", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope of ST segment", ["upsloping","flat","downsloping"])
    ca = st.number_input("Number of major vessels (0-3)", 0, 3, 0)
    thal = st.selectbox("Thalassemia", ["normal","fixed defect","reversible defect"])
    dataset = st.selectbox("Dataset", ["Hungary","Switzerland","Cleveland","Long Beach VA"])

# -----------------------------
# Convert categorical
# -----------------------------
sex = 1 if sex=="Male" else 0
fbs = 1 if fbs=="Yes" else 0
exang = 1 if exang=="Yes" else 0

# -----------------------------
# Prepare input DataFrame
# -----------------------------
input_dict = {
    "age": age, "sex": sex, "trestbps": trestbps, "chol": chol,
    "thalach": thalach, "oldpeak": oldpeak, "ca": ca,
    "cp": cp, "fbs": fbs, "restecg": restecg, "exang": exang,
    "slope": slope, "thal": thal, "dataset": dataset
}

input_df = pd.DataFrame([input_dict])
input_df = pd.get_dummies(input_df)

# Add missing columns & match order
for col in model_columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model_columns]

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict"):
    input_scaled = scaler.transform(input_df)
    probability = model.predict_proba(input_scaled)[0][1]  # Probability of disease
    prediction = model.predict(input_scaled)[0]

    st.markdown(f"<h3 style='text-align:center;'>Predicted Risk Probability: {probability*100:.2f}%</h3>", unsafe_allow_html=True)
    
    if prediction == 1:
        st.markdown(
            "<div style='padding:20px;background-color:#ffcccc;border-radius:10px;'>"
            "<h3 style='color:red;text-align:center;'>⚠️ High risk of Heart Disease!</h3></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='padding:20px;background-color:#ccffcc;border-radius:10px;'>"
            "<h3 style='color:green;text-align:center;'>✅ Low risk of Heart Disease</h3></div>",
            unsafe_allow_html=True
        )

    # Visualize probability as progress bar
    st.progress(int(probability*100))