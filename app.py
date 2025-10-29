import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --------------------------
# Page config
# --------------------------
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
# Layout: two columns for inputs
# --------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", 80, 200, 120)
    chol = st.number_input("Serum Cholesterol (chol)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
    restecg = st.selectbox("Resting ECG results (restecg)", [0, 1, 2])

with col2:
    thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", 50, 250, 150)
    exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
    oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0, 0.1)
    slope = st.selectbox("Slope of ST segment (slope)", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

# --------------------------
# Prepare input
# --------------------------
input_dict = {
    "age": age,
    "sex": 1 if sex == "Male" else 0,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal
}

input_df = pd.DataFrame([input_dict])

# Ensure input columns match model
try:
    input_df = input_df[model_columns]
except KeyError as e:
    st.error(f"Column mismatch: {e}")
    st.stop()

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

    # Probability progress bar
    st.markdown("### Heart Disease Probability")
    st.progress(float(probability))
    st.write(f"Probability: {probability*100:.2f}%")
