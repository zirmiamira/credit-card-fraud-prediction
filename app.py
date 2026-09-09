import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Allow importing from src/
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from predict import predict_fraud

# Page configuration
 
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide"
)

st.title(" Credit Card Fraud Detection")
st.write(
    "Enter the transaction features below to predict whether "
    "the transaction is fraudulent."
)

st.info("The model uses XGBoost with an optimized threshold of 0.10.")

# Input form
 
st.subheader("Transaction data")

transaction = {}

# Time and Amount
col1, col2 = st.columns(2)

with col1:
    transaction["Time"] = st.number_input(
        "Time",
        value=0.0
    )

with col2:
    transaction["Amount"] = st.number_input(
        "Amount",
        min_value=0.0,
        value=100.0
    )


# V1 to V28
st.subheader("Anonymized features")

cols = st.columns(4)

for i in range(1, 29):
    with cols[(i - 1) % 4]:
        transaction[f"V{i}"] = st.number_input(
            f"V{i}",
            value=0.0,
            format="%.6f"
        )

# Prediction
 
if st.button(" Predict transaction", use_container_width=True):

    result = predict_fraud(transaction)

    probability = result["fraud_probability"]
    prediction = result["prediction"]

    st.divider()

    st.subheader("Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Fraud probability",
            f"{probability:.2%}"
        )

    with col2:
        if prediction == 1:
            st.error("FRAUD DETECTED")
        else:
            st.success("NORMAL TRANSACTION")

    st.progress(float(probability))