import os, json, joblib, numpy as np, streamlit as st

st.set_page_config(page_title="Iris Classifier", layout="centered")
st.title("🌸 Iris Classifier")

@st.cache_resource
def load_model_and_meta():
    model = joblib.load("iris_knn.pkl") 
    meta = json.load(open("iris_meta.json")) if os.path.exists("iris_meta.json") else {}
    feats = meta.get("feature_order", [
        "sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"
    ])
    return model, feats

try:
    model, feats = load_model_and_meta()
except Exception as e:
    st.error(f"Model yüklenemedi: {e}")
    st.stop()

cols = st.columns(2)
defaults = [5.1, 3.5, 1.4, 0.2]
vals = [cols[i%2].number_input(feats[i].replace(" (cm)",""),
                               min_value=0.0, max_value=10.0,
                               value=defaults[i], step=0.1)
        for i in range(4)]

if st.button("Predict"):
    try:
        x = np.array(vals, dtype=float).reshape(1, -1)
        pred = model.predict(x)[0]                 
        label = str(pred)
        st.success(label)                          
    except Exception as e:
        st.error(f"Tahmin hatası: {e}")
