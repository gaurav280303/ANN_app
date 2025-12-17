import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Credit Risk Intelligence System",
    page_icon="📈",
    layout="centered"
)

# =========================================================
# CUSTOM CSS (Professional, not flashy)
# =========================================================
st.markdown("""
<style>
    .title {
        font-size: 44px;
        font-weight: 800;
        color: #F1F5F9;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 18px;
        color: #94A3B8;
        margin-bottom: 30px;
    }
    .section {
        font-size: 22px;
        font-weight: 600;
        margin-top: 30px;
        margin-bottom: 10px;
        color: #E5E7EB;
    }
    .result-box {
        background-color: #0F172A;
        padding: 25px;
        border-radius: 14px;
        margin-top: 25px;
        border-left: 6px solid #38BDF8;
    }
    .footer {
        margin-top: 50px;
        text-align: center;
        font-size: 14px;
        color: #94A3B8;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL & SCALER
# =========================================================
model = load_model("credit_ann_model.h5")
scaler = joblib.load("scaler.pkl")

# =========================================================
# SIDEBAR – PROFESSIONAL BRANDING
# =========================================================
st.sidebar.markdown("## 👤 Project Author")
st.sidebar.markdown("""
**Gaurav Singh**  
MBA – Artificial Intelligence & Data Science  

---
**Domain**  
Finance · Credit Risk · Banking  

---
**Tech Stack**  
ANN · Machine Learning  
Streamlit · Deployment
""")

st.sidebar.info(
    "This application demonstrates an end-to-end "
    "Artificial Neural Network system used for "
    "credit risk assessment in the finance industry."
)

# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="title">Credit Risk Intelligence System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">ANN-based decision support tool for financial institutions</div>',
    unsafe_allow_html=True
)

# =========================================================
# IMPORTANT: FEATURE ORDER MUST MATCH TRAINING
# =========================================================
st.markdown('<div class="section">Customer Financial Profile</div>', unsafe_allow_html=True)

# 🔒 These are CORE finance features used universally
FEATURES = [
    "Age",
    "Monthly Income",
    "Credit Score",
    "Loan Amount",
    "Loan Tenure (Months)",
    "Employment Experience (Years)",
    "Debt-to-Income Ratio",
    "Number of Active Loans",
    "Past Default Count",
    "Account Balance",
    "Credit Utilization Ratio",
    "Savings Amount"
]

inputs = []

# =========================================================
# INPUT UI (Sliders + Numbers)
# =========================================================
for feature in FEATURES:

    if feature == "Age":
        value = st.slider(feature, 18, 70, 30)

    elif feature == "Credit Score":
        value = st.slider(feature, 300, 900, 650)

    elif "Ratio" in feature:
        value = st.slider(feature, 0.0, 1.0, 0.3)

    elif "Tenure" in feature or "Experience" in feature:
        value = st.slider(feature, 0, 360, 60)

    else:
        value = st.number_input(feature, min_value=0.0, value=0.0)

    inputs.append(value)

# =========================================================
# PREDICTION
# =========================================================
if st.button("📊 Evaluate Credit Risk", use_container_width=True):

    data = np.array(inputs).reshape(1, -1)
    data_scaled = scaler.transform(data)

    probability = float(model.predict(data_scaled)[0][0])

    risk_label = "High Credit Risk" if probability >= 0.4 else "Low Credit Risk"
    risk_color = "#EF4444" if probability >= 0.4 else "#22C55E"

    st.markdown(f"""
    <div class="result-box">
        <h2 style="color:{risk_color};">
            {risk_label}
        </h2>
        <p style="font-size:20px;">
            Risk Probability: <b>{probability:.2f}</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer">
Developed by <b>Gaurav Singh</b> · Finance ANN Project · Streamlit Deployment
</div>
""", unsafe_allow_html=True)
