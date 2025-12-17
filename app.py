import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Credit Risk Predictor", layout="centered")

st.title("Credit Risk Prediction")
st.write("ANN-based model")

# -----------------------------
# Load model & scaler
# -----------------------------
try:
    model = load_model("credit_ann_model.keras", compile=False)
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error("Failed to load model or scaler.")
    st.stop()

# -----------------------------
# HARD DEBUG (NO CRASH)
# -----------------------------
st.subheader("Debug Info (do not remove)")
st.write("Scaler expects features:", getattr(scaler, "n_features_in_", "UNKNOWN"))
st.write("Model input shape:", model.input_shape)

EXPECTED_FEATURES = getattr(scaler, "n_features_in_", None)

# -----------------------------
# Inputs (ALWAYS 12)
# -----------------------------
st.subheader("Customer Inputs")

inputs = [
    st.number_input("Age", 18, 70, 30),
    st.number_input("Monthly Income", 0.0, value=50000.0),
    st.number_input("Credit Score", 300, 900, 650),
    st.number_input("Loan Amount", 0.0, value=200000.0),
    st.number_input("Loan Tenure (Months)", 0, value=36),
    st.number_input("Employment Years", 0, value=5),
    st.number_input("Debt-to-Income Ratio", 0.0, 1.0, 0.3),
    st.number_input("Active Loans", 0, value=1),
    st.number_input("Past Default Count", 0, value=0),
    st.number_input("Account Balance", 0.0, value=100000.0),
    st.number_input("Credit Utilization Ratio", 0.0, 1.0, 0.4),
    st.number_input("Savings Amount", 0.0, value=50000.0),
]

X = np.array(inputs).reshape(1, -1)
st.write("Input shape sent to scaler:", X.shape)

# -----------------------------
# Prediction (SAFE)
# -----------------------------
if st.button("Predict"):

    if EXPECTED_FEATURES is not None and X.shape[1] != EXPECTED_FEATURES:
        st.error(
            f"❌ Feature mismatch: scaler expects {EXPECTED_FEATURES}, "
            f"but app sent {X.shape[1]}"
        )
        st.stop()

    try:
        X_scaled = scaler.transform(X)
        prob = float(model.predict(X_scaled)[0][0])

        if prob >= 0.4:
            st.error(f"High Credit Risk (Probability: {prob:.2f})")
        else:
            st.success(f"Low Credit Risk (Probability: {prob:.2f})")

    except Exception as e:
        st.error("Prediction failed.")
        st.write("Error details:", str(e))
