import json
import requests
import pandas as pd
import streamlit as st
from streamlit.logger import get_logger

BACKEND = "http://localhost:8000"
LOGGER = get_logger(__name__)

# (key, label, min, max, default, step) — ranges cover the Wine dataset
FEATURES = [
    ("alcohol", "Alcohol (%)", 11.0, 14.9, 13.2, 0.01),
    ("malic_acid", "Malic acid", 0.7, 5.8, 1.78, 0.01),
    ("ash", "Ash", 1.3, 3.3, 2.14, 0.01),
    ("alcalinity_of_ash", "Alcalinity of ash", 10.6, 30.0, 11.2, 0.1),
    ("magnesium", "Magnesium", 70.0, 162.0, 100.0, 1.0),
    ("total_phenols", "Total phenols", 0.98, 3.88, 2.65, 0.01),
    ("flavanoids", "Flavanoids", 0.34, 5.08, 2.76, 0.01),
    ("nonflavanoid_phenols", "Nonflavanoid phenols", 0.13, 0.66, 0.26, 0.01),
    ("proanthocyanins", "Proanthocyanins", 0.41, 3.58, 1.28, 0.01),
    ("color_intensity", "Color intensity", 1.28, 13.0, 4.38, 0.01),
    ("hue", "Hue", 0.48, 1.71, 1.05, 0.01),
    ("od280_od315_of_diluted_wines", "OD280/OD315", 1.27, 4.0, 3.4, 0.01),
    ("proline", "Proline", 270.0, 1690.0, 1050.0, 10.0),
]


def backend_online():
    try:
        return requests.get(BACKEND, timeout=3).status_code == 200
    except requests.ConnectionError as e:
        LOGGER.error(e)
        return False


def run():
    st.set_page_config(page_title="Wine Classifier Dashboard", page_icon="🍷", layout="wide")

    with st.sidebar:
        if backend_online():
            st.success("Backend online ✅")
            try:
                info = requests.get(f"{BACKEND}/model-info", timeout=3).json()
                st.caption(f"Model: {info['model']} · Test accuracy: {info['test_accuracy']}")
            except Exception as e:
                LOGGER.error(e)
        else:
            st.error("Backend offline 😱 — start it with `uvicorn main:app --reload` in backend/src")

        mode = st.radio("Input method", ["Sliders", "Upload JSON"])
        features = {}
        if mode == "Sliders":
            st.info("Configure wine features")
            for key, label, lo, hi, default, step in FEATURES:
                features[key] = st.slider(label, lo, hi, default, step)
        else:
            uploaded = st.file_uploader("Upload test prediction file", type=["json"])
            if uploaded:
                data = json.load(uploaded)
                features = data.get("input_test", data)
                st.write("Preview file")
                st.json(features)

        predict_button = st.button("Predict", type="primary")

    st.write("# Wine Cultivar Prediction 🍷")
    st.write("Predicts which of three cultivars a wine comes from, using 13 chemical measurements.")

    if predict_button:
        if not features:
            st.toast(":red[Upload a JSON file first]", icon="⚠️")
            return
        try:
            with st.spinner("Predicting..."):
                response = requests.post(f"{BACKEND}/predict", json=features, timeout=10)
            if response.status_code == 200:
                result = response.json()
                probs = result["probabilities"]
                col1, col2 = st.columns(2)
                col1.success(f"Predicted cultivar: **{result['class_name']}**")
                col2.metric("Confidence", f"{max(probs.values()):.0%}")
                st.subheader("Class probabilities")
                st.bar_chart(pd.DataFrame({"probability": probs}))
            elif response.status_code == 422:
                st.error("Invalid input — the backend rejected these values:")
                st.json(response.json())
            else:
                st.toast(f":red[Server returned {response.status_code}]", icon="🔴")
        except requests.ConnectionError as e:
            LOGGER.error(e)
            st.toast(":red[Backend offline. Start the FastAPI server and refresh.]", icon="🔴")


if __name__ == "__main__":
    run()
