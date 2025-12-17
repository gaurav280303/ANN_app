import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="Credit Risk Prediction", layout="centered")

# ================================
# LOAD MODEL & SCALER
# ================================
model = load_model("credit_ann_model.keras", compile=False)
scaler = joblib.load("scaler.pkl")

# ================================
# TITLE
# ================================
st.title("Credit Risk Prediction System")
st.write("Artificial Neural Network based finance risk model")

# ================================
# FEATURE NAMES (MUST MATCH TRAINING EXACTLY)
# ================================
FEATURE_COLUMNS = [
    "age",
    "monthly_income",
    "credit_score",
    "loan_amount",
    "loan_tenure",
    "employment_years",
    "debt_to_income_ratio",
    "num_active_loans",
    "past_default_count",
    "account_balance",
    "credit_utilization_ratio",
    "savings_amount"
]

# ================================
# USER INPUTS
# ================================
inputs = []

inputs.append(st.number_input("Age", min_value=18, max_value=70, value=30))
inputs.append(st.number_input("Monthly Income", min_value=0.0, value=50000.0))
inputs.append(st.number_input("Credit Score", min_value=300, max_value=900, value=650))
inputs.append(st.number_input("Loan Amount", min_value=0.0, value=200000.0))
inputs.append(st.number_input("Loan Tenure", min_value=0, value=36))
inputs.append(st.number_input("Employment Years", min_value=0, value=5))
inputs.append(st.number_input("Debt to Income Ratio", min_value=0.0, max_value=1.0, value=0.3))
inputs.append(st.number_input("Active Loans", min_value=0, value=1))
inputs.append(st.number_input("Past Default Count", min_value=0, value=0))
inputs.append(st.number_input("Account Balance", min_value=0.0, value=100000.0))
inputs.append(st.number_input("Credit Utilization Ratio", min_value=0.0, max_value=1.0, value=0.4))
inputs.append(st.number_input("Savings Amount", min_value=0.0, value=50000.0))

# ================================
# PREDICTION
# ================================
if st.button("Evaluate Credit Risk"):

    # 🔴 IMPORTANT FIX — USE DATAFRAME, NOT NUMPY
    input_df = pd.DataFrame([inputs], columns=FEATURE_COLUMNS)

    # Scale
    data_scaled = scaler.transform(input_df)

    # Predict
    probability = float(model.predict(data_scaled)[0][0])

    if probability >= 0.4:
        st.error(f"High Credit Risk — Probability: {probability:.2f}")
    else:
        st.success(f"Low Credit Risk — Probability: {probability:.2f}")
