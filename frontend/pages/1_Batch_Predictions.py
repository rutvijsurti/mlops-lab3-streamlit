import requests
import pandas as pd
import streamlit as st

BACKEND = "http://localhost:8000"

st.set_page_config(page_title="Batch Predictions", page_icon="📊", layout="wide")
st.write("# Batch Predictions 📊")
st.write("Upload a CSV with the 13 wine feature columns to predict every row at once.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.write(f"Loaded {len(df)} rows")
    st.dataframe(df)

    if st.button("Predict all rows", type="primary"):
        predictions = []
        try:
            with st.spinner("Predicting..."):
                for _, row in df.iterrows():
                    r = requests.post(f"{BACKEND}/predict", json=row.astype(float).to_dict(), timeout=10)
                    predictions.append(r.json()["class_name"] if r.status_code == 200 else f"error {r.status_code}")
        except requests.ConnectionError:
            st.error("Backend offline 😱 — start the FastAPI server first.")
            st.stop()

        results = df.copy()
        results.insert(0, "prediction", predictions)
        st.subheader("Results")
        st.dataframe(results)
        st.subheader("Prediction counts")
        st.bar_chart(results["prediction"].value_counts())
        st.download_button("Download results CSV", results.to_csv(index=False),
                           file_name="predictions.csv", mime="text/csv")
