import streamlit as st
import pandas as pd
import joblib

# Load the trained model pipeline
model = joblib.load("model.joblib")

st.title("🏦 Credit Risk Prediction App")

st.write("This app predicts the **Probability of Default (PD)** for a loan applicant.")

# Input fields
EXT_SOURCE_1 = st.number_input("External Score 1", 0.0, 1.0, 0.5)
EXT_SOURCE_2 = st.number_input("External Score 2", 0.0, 1.0, 0.5)
EXT_SOURCE_3 = st.number_input("External Score 3", 0.0, 1.0, 0.5)

DAYS_BIRTH = st.number_input("Age (in days, negative)", value=-12000)
DAYS_EMPLOYED = st.number_input("Employment days (negative)", value=-2000)

AMT_INCOME_TOTAL = st.number_input("Income", value=300000.0)
AMT_CREDIT = st.number_input("Credit Amount", value=600000.0)
AMT_ANNUITY = st.number_input("Annuity (EMI)", value=25000.0)
AMT_GOODS_PRICE = st.number_input("Goods Price", value=550000.0)

NAME_INCOME_TYPE = st.selectbox(
    "Income Type",
    ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed"]
)

NAME_EDUCATION_TYPE = st.selectbox(
    "Education Type",
    ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"]
)

NAME_FAMILY_STATUS = st.selectbox(
    "Family Status",
    ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"]
)

NAME_HOUSING_TYPE = st.selectbox(
    "Housing Type",
    ["House / apartment", "With parents", "Municipal apartment", "Rented apartment", "Office apartment"]
)

REGION_RATING_CLIENT = st.selectbox("Region Rating", [1, 2, 3])
FLAG_DOCUMENT_3 = st.selectbox("Document 3 Provided?", [0, 1])

# Predict button
if st.button("Predict Credit Risk"):
    input_data = pd.DataFrame([{
        "EXT_SOURCE_1": EXT_SOURCE_1,
        "EXT_SOURCE_2": EXT_SOURCE_2,
        "EXT_SOURCE_3": EXT_SOURCE_3,
        "DAYS_BIRTH": DAYS_BIRTH,
        "DAYS_EMPLOYED": DAYS_EMPLOYED,
        "AMT_INCOME_TOTAL": AMT_INCOME_TOTAL,
        "AMT_CREDIT": AMT_CREDIT,
        "AMT_ANNUITY": AMT_ANNUITY,
        "AMT_GOODS_PRICE": AMT_GOODS_PRICE,
        "NAME_INCOME_TYPE": NAME_INCOME_TYPE,
        "NAME_EDUCATION_TYPE": NAME_EDUCATION_TYPE,
        "NAME_FAMILY_STATUS": NAME_FAMILY_STATUS,
        "NAME_HOUSING_TYPE": NAME_HOUSING_TYPE,
        "REGION_RATING_CLIENT": REGION_RATING_CLIENT,
        "FLAG_DOCUMENT_3": FLAG_DOCUMENT_3
    }])

    probability = model.predict_proba(input_data)[0][1]

    st.success(f"Probability of Default: {probability:.2%}")
