import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression

import os, pathlib
os.environ["HOME"] = "/tmp"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
pathlib.Path("/tmp/.streamlit").mkdir(exist_ok=True)


model = joblib.load("sales_model.joblib")

st.set_page_config(page_title="Sales Prediction", page_icon="📈")
st.title("Sales Prediction")

tv = st.number_input("TV",        min_value=0.0, value=230.1, step=1.0)
radio = st.number_input("Radio",  min_value=0.0, value=37.8,  step=0.1)
newspaper = st.number_input("Newspaper", min_value=0.0, value=69.2, step=0.1)

if st.button("Predict"):
    X = pd.DataFrame([[tv, radio, newspaper]], columns=["TV","Radio","Newspaper"])
    y_hat = model.predict(X)[0]
    st.metric("Predicted Sales ", f"{y_hat:.2f}")
