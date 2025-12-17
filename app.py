import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Bank Marketing Prediction",
    layout="centered"
)

st.title("Bank Marketing Subscription Prediction")
st.write("ANN-based decision support system")

# ======================================================
# LOAD MODEL & PREPROCESSOR
# ======================================================
model = load_model("bank_marketing_ann.keras", compile=False)
preprocessor = joblib.load("preprocessor.pkl")

# ======================================================
# USER INPUTS (RAW DATASET FEATURES)
# ======================================================
st.subheader("Customer Information")

age = st.number_input("Age", min_value=18, max_value=100, value=35)

job = st.selectbox(
    "Job",
    [
        'admin.', 'blue-collar', 'entrepreneur', 'housemaid',
        'management', 'retired', 'self-employed', 'services',
        'student', 'technician', 'unemployed', 'unknown'
    ]
)

marital = st.selectbox(
    "Marital Status",
    ['married', 'single', 'divorced']
)

education = st.selectbox(
    "Education",
    [
        'basic.4y', 'basic.6y', 'basic.9y',
        'high.school', 'professional.course',
        'university.degree', 'unknown'
    ]
)

default = st.selectbox("Credit Default", ['no', 'yes', 'unknown'])
housing = st.selectbox("Housing Loan", ['no', 'yes', 'unknown'])
loan = st.selectbox("Personal Loan", ['no', 'yes', 'unknown'])

campaign = st.number_input("Number of Contacts in Current Campaign", min_value=1, value=1)
pdays = st.number_input("Days Since Last Contact (999 = never)", value=999)
previous = st.number_input("Number of Previous Contacts", min_value=0, value=0)

euribor3m = st.number_input("Euribor 3 Month Rate", value=4.0)
nr_employed = st.number_input("Number of Employees", value=5000.0)

# ======================================================
# BUILD INPUT DATAFRAME (MUST MATCH TRAINING COLUMNS)
# ======================================================
input_df = pd.DataFrame([{
    'age': age,
    'job': job,
    'marital': marital,
    'education': education,
    'default': default,
    'housing': housing,
    'loan': loan,
    'campaign': campaign,
    'pdays': pdays,
    'previous': previous,
    'euribor3m': euribor3m,
    'nr.employed': nr_employed
}])

# ======================================================
# PREDICTION
# ======================================================
if st.button("Predict Subscription"):

    # Preprocess input
    X_processed = preprocessor.transform(input_df)

    # Predict probability
    probability = float(model.predict(X_processed)[0][0])

    if probability >= 0.5:
        st.success(f"Customer is likely to subscribe (Probability: {probability:.2f})")
    else:
        st.error(f"Customer is unlikely to subscribe (Probability: {probability:.2f})")

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.caption("Developed by Gaurav Singh | ANN • Machine Learning • Bank Marketing")
