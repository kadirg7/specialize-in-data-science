# app.py
import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    return joblib.load("profit_prediction_model.joblib")

model = load_model()

st.title("Profit Prediction")
st.write("Enter costs to predict company profit.")

rd  = st.number_input("R&D Spend",        min_value=0.0, step=1000.0, format="%.2f")
adm = st.number_input("Administration",   min_value=0.0, step=1000.0, format="%.2f")
mkt = st.number_input("Marketing Spend",  min_value=0.0, step=1000.0, format="%.2f")

if st.button("Predict Profit"):
    X = pd.DataFrame([[rd, adm, mkt]],
                     columns=["R&D Spend", "Administration", "Marketing Spend"])
    y_pred = model.predict(X)[0]
    st.success(f"Predicted Profit: {y_pred:,.2f}")
