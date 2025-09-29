import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from pathlib import Path


TICKER, START_DATE, END_DATE = "AAPL", "2012-01-23", "2025-09-26"

st.set_page_config(page_title="LSTM Stock Close", page_icon="📈")

from pathlib import Path
from tensorflow.keras.models import load_model

@st.cache_resource
def get_model():
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "lstm_stock_model.keras"   
    return load_model(str(model_path), compile=False)

model = get_model()

st.title("📈 LSTM Stock Close Prediction")
st.caption(
    f"Ticker: **{TICKER}** | Data: **{START_DATE} → {END_DATE}** · "
    "Inputs: Open, High, Low, Volume → Output: Close"
)

c1, c2 = st.columns(2)
with c1:
    open_p = st.number_input("Open",  value=177.08, step=0.01)
    low_p  = st.number_input("Low",   value=177.07, step=0.01)
with c2:
    high_p = st.number_input("High",  value=180.42, step=0.01)
    volume = st.number_input("Volume", value=74_919_600.0, step=10_000.0, format="%.0f")

if st.button("Predict"):
    X = np.array([[open_p, high_p, low_p, volume]], dtype=np.float32).reshape(1, 4, 1)
    y = float(model.predict(X, verbose=0)[0, 0])
    st.metric("Predicted Close", f"{y:.2f}")

st.caption("Keep input order: [Open, High, Low, Volume].")
