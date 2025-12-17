import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# -----------------------------
# Load trained model & scaler
# -----------------------------
model = load_model("credit_ann_model.h5")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="ANN Credit Risk Prediction", layout="centered")

st.title("ANN Credit Risk Prediction")
st.write("Enter customer feature values to predict risk")

# Number of features expected by the model
num_features = scaler.n_features_in_

st.subheader("Input Features")

input_values = []

for i in range(num_features):
    value = st.number_input(
        label=f"Feature {i+1}",
        value=0.0,
        step=0.1
    )
    input_values.append(value)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    data = np.array(input_values).reshape(1, -1)
    data_scaled = scaler.transform(data)

    probability = model.predict(data_scaled)[0][0]

    decision = "YES" if probability >= 0.4 else "NO"

    st.subheader("Prediction Result")
    st.write(f"**Probability:** {probability:.3f}")
    st.write(f"**Decision:** {decision}")
