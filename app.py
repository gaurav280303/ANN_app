import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load trained pipeline
# -----------------------------
model = joblib.load("model.joblib")

# -----------------------------
# LOCKED FEATURE LIST (must match training)
# -----------------------------
FEATURES = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "REGION_RATING_CLIENT",
    "FLAG_DOCUMENT_3",
]

st.set_page_config(page_title="Credit Risk ANN Pipeline")

st.title("🏦 Credit Risk Decision Support System")

st.write("Predict **Probability of Default (PD)** using an ANN-based credit risk pipeline.")

# -----------------------------
# INPUTS (UI)
# -----------------------------
inputs = {}

inputs["EXT_SOURCE_1"] = st.number_input("External Source 1", 0.0, 1.0, 0.5)
inputs["EXT_SOURCE_2"] = st.number_input("External Source 2", 0.0, 1.0, 0.5)
inputs["EXT_SOURCE_3"] = st.number_input("External Source 3", 0.0, 1.0, 0.5)

inputs["DAYS_BIRTH"] = st.number_input("Age (days, negative)", value=-12000)
inputs["DAYS_EMPLOYED"] = st.number_input("Employment days (negative)", value=-2000)

inputs["AMT_INCOME_TOTAL"] = st.number_input("Total Income", value=300000.0)
inputs["AMT_CREDIT"] = st.number_input("Credit Amount", value=600000.0)
inputs["AMT_ANNUITY"] = st.number_input("Loan Annuity", value=25000.0)
inputs["AMT_GOODS_PRICE"] = st.number_input("Goods Price", value=550000.0)

inputs["NAME_INCOME_TYPE"] = st.selectbox(
    "Income Type",
    ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed"]
)

inputs["NAME_EDUCATION_TYPE"] = st.selectbox(
    "Education Type",
    [
        "Secondary / secondary special",
        "Higher education",
        "Incomplete higher",
        "Lower secondary",
        "Academic degree",
    ]
)

inputs["NAME_FAMILY_STATUS"] = st.selectbox(
    "Family Status",
    ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"]
)

inputs["NAME_HOUSING_TYPE"] = st.selectbox(
    "Housing Type",
    [
        "House / apartment",
        "With parents",
        "Municipal apartment",
        "Rented apartment",
        "Office apartment",
    ]
)

inputs["REGION_RATING_CLIENT"] = st.selectbox("Region Rating", [1, 2, 3])
inputs["FLAG_DOCUMENT_3"] = st.selectbox("Document 3 Provided?", [0, 1])

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Evaluate Credit Risk"):
    # Build DataFrame in correct order
    input_df = pd.DataFrame([[inputs[col] for col in FEATURES]], columns=FEATURES)

    # Force numeric columns to numeric
    numeric_cols = [
        "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
        "DAYS_BIRTH", "DAYS_EMPLOYED",
        "AMT_INCOME_TOTAL", "AMT_CREDIT",
        "AMT_ANNUITY", "AMT_GOODS_PRICE",
        "REGION_RATING_CLIENT", "FLAG_DOCUMENT_3"
    ]

    input_df[numeric_cols] = input_df[numeric_cols].apply(pd.to_numeric)

    # Predict
    pd_value = model.predict_proba(input_df)[0][1]

    st.metric("Probability of Default (PD)", f"{pd_value:.2%}")

    if pd_value >= 0.5:
        st.error("⚠️ High Credit Risk")
    else:
        st.success("✅ Lower Credit Risk")
