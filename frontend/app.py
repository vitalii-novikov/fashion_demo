import streamlit as st
import requests
import os
from PIL import Image
import io

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Fashion Style Classifier", layout="centered")
st.title("👗 Fashion Style Classifier")


# Simple in-memory storage for last embedding
if "last_embedding" not in st.session_state:
    st.session_state.last_embedding = None


# --- IMAGE UPLOAD AND CLASSIFICATION ---
st.header("1. Upload an image")

uploaded = st.file_uploader("Upload your clothing image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", width=300)

    if st.button("Classify"):
        with st.spinner("Classifying..."):
            files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
            r = requests.post(f"{API_URL}/classify", files=files)

        if r.status_code != 200:
            st.error(f"Error: {r.text}")
        else:
            result = r.json()

            # Save embedding for recommendations
            st.session_state.last_embedding = result.get("embedding")

            # UI for main style only
            st.subheader("Detected main style:")
            st.success(f"🎯 {result['main_style']}")

            # Expandable full result
            with st.expander("Show full model output"):
                st.json(result)


# --- RECOMMENDATIONS ---
st.header("2. Get recommendations")

k = st.number_input("Number of recommendations:", min_value=1, max_value=30, value=5)

if st.button("Get recommendations"):
    if st.session_state.last_embedding is None:
        st.warning("Please classify an image first.")
    else:
        req = {
            "embedding": st.session_state.last_embedding,
            "k": k
        }

        with st.spinner("Searching for similar items..."):
            r = requests.post(f"{API_URL}/recommend", json=req)

        if r.status_code != 200:
            st.error(f"Error: {r.text}")
        else:
            recs = r.json()["recommendations"]

            st.subheader("Recommended items:")
            for item in recs:
                st.markdown(f"### **{item['name']}**")
                st.image(item["url"], width=250)
                st.write(f"Distance: `{item['distance']:.4f}`")
                st.markdown("---")
