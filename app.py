import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time

# --------------------------
# Page config
# --------------------------
st.set_page_config(page_title="❤️ Heart Disease Predictor", layout="wide")
st.title("❤️ Heart Disease Predictor")
st.markdown("Enter patient details in the sidebar to predict the likelihood of heart disease.")

# --------------------------
# Helper to safely load files
# --------------------------
def safe_load(filename):
    if os.path.exists(filename):
        return joblib.load(filename)
    else:
        st.error(f"❌ File not found: {filename}")
        return None

model = safe_load("logistic_model.pkl")
scaler = safe_load("scaler.pkl")
model_columns = safe_load("model_columns.pkl")

if model is None or scaler is None or model_columns is None:
    st.stop()

# --------------------------
# Sidebar Inputs
# --------------------------
st.sidebar.header("Patient Details")
thalach = st.sidebar.number_input("Maximum Heart Rate Achieved", 50, 250, 150)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
cp = st.sidebar.selectbox("Chest Pain Type", ["atypical angina", "non-anginal", "typical angina"])
restecg = st.sidebar.selectbox("Resting ECG Result", ["normal", "st-t abnormality"])
slope = st.sidebar.selectbox("Slope of ST Segment", ["flat", "upsloping"])
thal = st.sidebar.selectbox("Thalassemia", ["normal", "reversable defect"])
dataset = st.sidebar.selectbox("Dataset Source", ["Hungary", "Switzerland", "VA Long Beach"])

# --------------------------
# Prepare input (one-hot encoding)
# --------------------------
input_dict = {
    "thalch": thalach,
    "sex_Male": 1 if sex=="Male" else 0,
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

input_df = pd.DataFrame([input_dict])
input_df = input_df.reindex(columns=model_columns, fill_value=0)
input_scaled = scaler.transform(input_df)

# --------------------------
# Prediction & Interactive Gauge
# --------------------------
if st.button("Predict"):
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]
    prob_percent = probability*100

    # Prediction card
    if prediction[0] == 1:
        st.markdown(
            f'<div style="background-color:#ffcccc;padding:15px;border-radius:10px;font-size:18px;">⚠️ Heart Disease Predicted</div>', 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div style="background-color:#ccffcc;padding:15px;border-radius:10px;font-size:18px;">✅ No Heart Disease</div>', 
            unsafe_allow_html=True
        )

    st.markdown("### Heart Disease Probability (Animated & Interactive)")

    # Animated gauge placeholder
    gauge_placeholder = st.empty()

    # Gradient colors for smooth transition
    def get_color(val):
        if val < 30:
            return "#2ecc71"
        elif val < 70:
            return "#f1c40f"
        else:
            return "#e74c3c"

    # Animate gauge fill with floating heart icon
    for i in range(0, int(prob_percent)+1):
        color = get_color(i)
        gauge_html = f'''
        <div style="display:flex; justify-content:center; align-items:center; height:220px;">
            <svg width="180" height="180">
                <circle cx="90" cy="90" r="80" stroke="#e0e0e0" stroke-width="20" fill="none"/>
                <circle cx="90" cy="90" r="80" stroke="{color}" stroke-width="20" fill="none"
                    stroke-dasharray="{440 * i / 100} 440"
                    stroke-dashoffset="0" transform="rotate(-90 90 90)"/>
                <text x="90" y="100" font-size="24" text-anchor="middle" fill="#000">{i:.0f}%</text>
                <text x="90" y="135" font-size="28" text-anchor="middle" fill="{color}" style="animation: pulse 1s infinite;">❤️</text>
                <style>
                    @keyframes pulse {{
                        0% {{ transform: scale(1); }}
                        50% {{ transform: scale(1.3); }}
                        100% {{ transform: scale(1); }}
                    }}
                </style>
            </svg>
        </div>
        '''
        gauge_placeholder.markdown(gauge_html, unsafe_allow_html=True)
        time.sleep(0.01)
