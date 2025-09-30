import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---- settings
st.set_page_config(page_title="Mobile Price Classification", layout="centered")
st.title("📱 Mobile Price Classification")

# ---- load trained pipeline (StandardScaler + LogisticRegression)
pipe = joblib.load("mobile_price.pkl")  # önce notebook'ta kaydettiğin dosya

FEATURES = [
    "battery_power","blue","clock_speed","dual_sim","fc","four_g","int_memory",
    "m_dep","mobile_wt","n_cores","pc","px_height","px_width","ram",
    "sc_h","sc_w","talk_time","three_g","touch_screen","wifi"
]
BINARY = {"blue","dual_sim","four_g","three_g","touch_screen","wifi"}

with st.form("form"):
    cols = st.columns(2)
    vals = {}
    for i, f in enumerate(FEATURES):
        c = cols[i % 2]
        if f in BINARY:
            vals[f] = c.selectbox(f, [0, 1], index=1 if f in {"wifi","three_g","four_g"} else 0)
        else:
            step = 1.0 if f != "m_dep" else 0.1
            vals[f] = c.number_input(f, value=0.0, step=step, format="%.3f")
    ok = st.form_submit_button("Predict")

if ok:
    X = pd.DataFrame([vals], columns=FEATURES)
    pred = int(pipe.predict(X)[0])
    proba = getattr(pipe, "predict_proba", lambda z: None)(X)
    labels = {0: "low", 1: "medium", 2: "high", 3: "very high"}

    st.success(f"Predicted price range: **{labels.get(pred, pred)}**")
    if proba is not None:
        st.write("Class probabilities:", {labels[i]: float(p) for i, p in enumerate(proba[0])})

st.caption("Model file: mobile_price.pkl (scaler + logistic regression pipeline)")
