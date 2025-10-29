import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import plotly.graph_objects as go

# --------------------------
# Page config
# --------------------------
st.set_page_config(page_title="❤️ Heart Disease Predictor", layout="wide")
st.title("❤️ Heart Disease Predictor")
st.markdown("Enter all patient details below to predict the likelihood of heart disease and see risk factors.")

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
# Sidebar Inputs with collapsible sections
# --------------------------
st.sidebar.header("Patient Details")

with st.sidebar.expander("Basic Info"):
    age = st.number_input("Age", 1, 120, 50, help="Patient age in years")
    sex = st.selectbox("Sex", ["Male", "Female"], help="Patient sex")

with st.sidebar.expander("Chest & Heart Info"):
    cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
    trestbps = st.number_input("Resting BP (trestbps)", 80, 200, 120)
    chol = st.number_input("Serum Cholesterol (chol)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
    restecg = st.selectbox("Resting ECG", ["normal", "st-t abnormality", "left ventricular hypertrophy"])
    thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", 50, 250, 150)
    exang = st.selectbox("Exercise Induced Angina (exang)", [0,1])
    oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
    slope = st.selectbox("Slope of ST Segment", ["upsloping", "flat", "downsloping"])
    ca = st.selectbox("Number of Major Vessels (ca)", [0,1,2,3,4])
    thal = st.selectbox("Thalassemia (thal)", ["normal", "fixed defect", "reversable defect"])
    dataset = st.selectbox("Dataset Source", ["Hungary", "Switzerland", "VA Long Beach"])

# --------------------------
# Prepare input (one-hot encoding)
# --------------------------
input_dict = {
    "age": age,
    "trestbps": trestbps,
    "chol": chol,
    "thalach": thalach,
    "oldpeak": oldpeak,
    "ca": ca,
    "sex_Male": 1 if sex=="Male" else 0,
    "cp_typical angina": 1 if cp=="typical angina" else 0,
    "cp_atypical angina": 1 if cp=="atypical angina" else 0,
    "cp_non-anginal": 1 if cp=="non-anginal" else 0,
    "cp_asymptomatic": 1 if cp=="asymptomatic" else 0,
    "fbs": fbs,
    "restecg_normal": 1 if restecg=="normal" else 0,
    "restecg_st-t abnormality": 1 if restecg=="st-t abnormality" else 0,
    "restecg_left ventricular hypertrophy": 1 if restecg=="left ventricular hypertrophy" else 0,
    "exang": exang,
    "slope_upsloping": 1 if slope=="upsloping" else 0,
    "slope_flat": 1 if slope=="flat" else 0,
    "slope_downsloping": 1 if slope=="downsloping" else 0,
    "thal_normal": 1 if thal=="normal" else 0,
    "thal_fixed defect": 1 if thal=="fixed defect" else 0,
    "thal_reversable defect": 1 if thal=="reversable defect" else 0,
    "dataset_Hungary": 1 if dataset=="Hungary" else 0,
    "dataset_Switzerland": 1 if dataset=="Switzerland" else 0,
    "dataset_VA Long Beach": 1 if dataset=="VA Long Beach" else 0
}

input_df = pd.DataFrame([input_dict])
input_df = input_df.reindex(columns=model_columns, fill_value=0)
input_scaled = scaler.transform(input_df)

# --------------------------
# Prediction & Results
# --------------------------
if st.button("Predict"):
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]
    prob_percent = probability*100

    # Prediction card
    if prediction[0] == 1:
        st.markdown(f'<div style="background-color:#ffcccc;padding:15px;border-radius:10px;font-size:18px;">⚠️ Heart Disease Predicted</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="background-color:#ccffcc;padding:15px;border-radius:10px;font-size:18px;">✅ No Heart Disease</div>', unsafe_allow_html=True)

    # Animated Gauge
    st.markdown("### Heart Disease Probability")
    gauge_placeholder = st.empty()
    def get_color(val):
        if val < 30: return "#2ecc71"
        elif val < 70: return "#f1c40f"
        else: return "#e74c3c"

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

    # Probability Chart
    st.markdown("### Probability Chart")
    fig = go.Figure(go.Bar(
        x=["No Heart Disease", "Heart Disease"],
        y=[100-prob_percent, prob_percent],
        marker_color=["#2ecc71", "#e74c3c"]
    ))
    fig.update_layout(yaxis=dict(title='Probability (%)', range=[0,100]))
    st.plotly_chart(fig, use_container_width=True)

    # Risk Factors Summary
    st.markdown("### Key Risk Factors")
    risk_list = []
    if trestbps>140: risk_list.append("High Blood Pressure")
    if chol>240: risk_list.append("High Cholesterol")
    if oldpeak>2: risk_list.append("High ST Depression")
    if exang==1: risk_list.append("Exercise Induced Angina")
    if fbs==1: risk_list.append("High Fasting Blood Sugar")
    if ca>=2: risk_list.append("Multiple Major Vessels Affected")
    if len(risk_list)==0: risk_list.append("No major risk factors detected")
    st.write(", ".join(risk_list))
