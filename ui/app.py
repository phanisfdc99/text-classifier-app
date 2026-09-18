import os
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://backend:8000")

st.set_page_config(page_title="Text Classifier", page_icon="📧")
st.title("📧 Text / Email Classifier")
st.write("Enter some text (e.g. an email body) and classify it.")

text_input = st.text_area("Input text", height=180, placeholder="Paste an email or any text here...")

if st.button("Classify"):
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Classifying..."):
            try:
                response = requests.post(f"{API_URL}/classify", json={"text": text_input}, timeout=30)
                response.raise_for_status()
                result = response.json()
                st.success(f"**Label:** {result['label']}  |  **Confidence:** {result['score']}")
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach the classification API: {e}")

st.caption(f"Backend API: {API_URL}")
