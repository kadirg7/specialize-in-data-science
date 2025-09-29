import streamlit as st
import pandas as pd
import joblib

FEATURES = [
    "enginesize", "curbweight", "horsepower",
    "carwidth", "carheight", "wheelbase",
    "boreratio", "citympg", "highwaympg"
]

model = st.cache_resource(joblib.load)("car_price_model.joblib")


st.title("Car Price Prediction")

inputs = {f: st.text_input(f, placeholder="") for f in FEATURES}

if st.button("Predict"):
    try:
        row = [float(str(inputs[f]).replace(",", ".").strip()) for f in FEATURES]
        X = pd.DataFrame([row], columns=FEATURES)
        y = model.predict(X)[0]
        st.success(f"Estimated Price: {y:,.2f}")
    except Exception:
        st.error("Please fill all fields with numeric values.")