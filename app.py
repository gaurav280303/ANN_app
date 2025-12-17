import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Credit Risk Intelligence System",
    page_icon="📊",
    layout="centered"
)

# =====================================================
# STYLING (Professional, not flashy)
# =====================================================
st.markdown("""
<style>
.title {
    font-size: 40px;
    font-weight: 800;
    color: #F8FAFC;
}
.subtitle {
    font-size: 17px;
    color: #94A3B8;
    margin-bottom: 25px;
}
.section {
    font-size: 22px;
    font-weight: 600;
    margin-top: 25px;
    color: #E5E7EB;
}
.result-box {
    background-color: #0F172A;
    padding: 22px;
    border-radius: 12px;
    margin-top: 25px;
}
.footer {
    margin-top: 40px;
    text-align: center;
    font-size: 14px;
    color: #94A3B8;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL & SCALER (ALIGNED – FIXED)
# =====================================================
model = load_model("credit_ann_model.keras", compile=False)
scaler = joblib.load("scaler.pkl")

# =====================================================
# SIDEBAR (Professional identity)
# =====================================================
st.sidebar.markdown("## 👤 Project Owner")
st.sidebar.markdown("""
**Gaurav Singh**  
MBA – Artificial Intelligence & Data Science  

**Domain:** Finance / Credit Risk  
**Model:** Artificial Neural Network  

This application demonstrates a  
**production-ready ANN pipeline**.
""")

# =====================================================
# HEADER
# =====================================================
st.markdown('<div class="title">Credit Risk Intelligence System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">ANN-based decision support for financial institutions</div>',
    unsafe_allow_html=True
)

# =====================================================
# INPUT FEATURES (MUST MATCH TRAINING ORDER)
# =====================================================
st.markdown('<div class="section">Customer Financial Profile</div>', unsafe_allow_html=True)

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

for feature in FEATURES:

    if feature == "Age":
        value = st.slider(feature, 18, 70, 30)

    elif feature == "Credit Score":
        value = st.slider(feature, 300, 900, 650)

    elif "Ratio" in feature:
        value = st.slider(feature, 0.0, 1.0, 0.3)

    elif "Tenure" in feature:
        value = st.slider(feature, 0, 360, 60)

    elif "Experience" in feature:
        value = st.slider(feature, 0, 40, 5)

    else:
        value = st.number_input(feature, min_value=0.0, value=0.0)

    inputs.append(value)

# =====================================================
# PREDICTION
# =====================================================
if st.button("📊 Evaluate Credit Risk", use_container_width=True):

    data = np.array(inputs).reshape(1, -1)
    data_scaled = scaler.transform(data)

    probability = float(model.predict(data_scaled)[0][0])

    if probability >= 0.4:
        label = "High Credit Risk"
        color = "#EF4444"
    else:
        label = "Low Credit Risk"
        color = "#22C55E"

    st.markdown(f"""
    <div class="result-box">
        <h2 style="color:{color};">{label}</h2>
        <p style="font-size:20px;">
            Risk Probability: <b>{probability:.2f}</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<div class="footer">
Developed by <b>Gaurav Singh</b> · Finance ANN · Streamlit Deployment
</div>
""", unsafe_allow_html=True)
