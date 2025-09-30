import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Credit Score Predictor", layout="centered")
st.title("Credit Score Predictor")


model = joblib.load("rf_model.pkl")
feature_order = joblib.load("feature_order.pkl")
credit_mix_map = joblib.load("credit_mix_map.pkl")  


defaults = {
    "Annual_Income": 60000.0,
    "Monthly_Inhand_Salary": 3500.0,
    "Num_Bank_Accounts": 3.0,
    "Num_Credit_Card": 2.0,
    "Interest_Rate": 12.0,
    "Num_of_Loan": 1.0,
    "Delay_from_due_date": 5.0,
    "Num_of_Delayed_Payment": 1.0,
    "Credit_Mix": "Standard",     
    "Outstanding_Debt": 1200.0,
    "Credit_History_Age": 84.0,
    "Monthly_Balance": 500.0,
}

with st.form("form"):
    c1, c2 = st.columns(2)
    vals = {}

    num_left = ["Annual_Income","Num_Bank_Accounts","Interest_Rate",
                "Delay_from_due_date","Outstanding_Debt","Monthly_Balance"]

    for k in feature_order:
        if k == "Credit_Mix":
            label = st.selectbox("Credit Mix", list(credit_mix_map.keys()),
                                 index=list(credit_mix_map.keys()).index(defaults["Credit_Mix"]))
            vals[k] = credit_mix_map[label]  
        elif k in num_left:
            vals[k] = c1.number_input(k.replace("_"," "), value=float(defaults[k]))
        else:
            vals[k] = c2.number_input(k.replace("_"," "), value=float(defaults[k]))

    submitted = st.form_submit_button("Predict")

if submitted:
    row = np.array([[vals[col] for col in feature_order]], dtype=float)
    pred = model.predict(row)[0]   
    st.success(f"Predicted Credit Score: {pred}")