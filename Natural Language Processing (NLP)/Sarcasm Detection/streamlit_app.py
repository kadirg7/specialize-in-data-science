from pathlib import Path
import joblib
import streamlit as st

st.set_page_config(page_title="Sarcasm Detector", layout="centered")
st.title("🧠 Sarcasm Detector")

MODEL_PATH = Path(__file__).parent / "sarcasm_nb_pipeline.joblib"  
pipe = joblib.load(MODEL_PATH)

text = st.text_area("Enter a sentence:", height=120, placeholder="Type a headline...")
if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        st.write(pipe.predict([text])[0])  