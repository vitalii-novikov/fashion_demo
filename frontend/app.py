import streamlit as st
import requests
import os
from PIL import Image
import io
import random

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Fashion Style Classifier", layout="wide")
st.title("👗 Fashion Style Classifier")


# Store last embedding
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

            st.session_state.last_embedding = result.get("embedding")

            st.subheader("Detected main style:")
            st.success(f"🎯 {result['main_style']}")

            # Full model output
            with st.expander("Show full model output"):
                st.json(result)


# --- RECOMMENDATIONS ---
st.header("2. Get recommendations")

k = st.number_input("Number of recommendations:", min_value=1, max_value=50, value=8)

# Randomness slider: 0 = no random, 1 = fully random
randomness = st.slider("Randomness level", 0.0, 1.0, 0.0)

# Fetch dataset size from API (cached)
@st.cache_data
def fetch_dataset_size():
    try:
        r = requests.get(f"{API_URL}/dataset_size")
        return r.json().get("size", 50000)
    except:
        return 50000  # fallback if API not responding

dataset_size = fetch_dataset_size()


if st.button("Get recommendations"):
    if st.session_state.last_embedding is None:
        st.warning("Please classify an image first.")
    else:

        # Compute d using the user formula
        d = int(k * (1 + 5 * randomness))
        d = min(dataset_size, d)
        d = max(d, k)

        st.info(f"Fetching top {d} candidates (dataset size = {dataset_size})")

        req = {
            "embedding": st.session_state.last_embedding,
            "k": d
        }

        with st.spinner("Searching for similar items..."):
            r = requests.post(f"{API_URL}/recommend", json=req)

        if r.status_code != 200:
            st.error(f"Error: {r.text}")
        else:
            candidates = r.json()["recommendations"]

            # Pick n items:
            if randomness == 0:
                # Deterministic: take first n
                recs = candidates[:k]
            else:
                # Random selection of n from d candidates
                recs = random.sample(candidates, min(k, len(candidates)))

            # Render items in a 4-column grid
            st.subheader("Recommended items:")

            # Custom card CSS (same as before)
            st.markdown("""
                <style>
                    .item-card {
                        background-color: #f2f2f2;
                        border-radius: 10px;
                        padding: 10px;
                        margin: 5px;
                        text-align: center;
                        box-shadow: 0 0 5px rgba(0,0,0,0.1);
                    }
                    .item-name {
                        font-weight: bold;
                        margin-top: 8px;
                        font-size: 16px;
                    }
                    .item-distance {
                        color: gray;
                        font-size: 14px;
                    }
                    img {
                        border-radius: 6px;
                    }
                </style>
            """, unsafe_allow_html=True)

            cols = st.columns(4)

            for idx, item in enumerate(recs):
                col = cols[idx % 4]
                with col:
                    st.markdown(
                        f"""
                        <div class="item-card">
                            <img src="{item['url']}" width="200">
                            <div class="item-name">{item['name']}</div>
                            <div class="item-distance">distance: {item['distance']:.4f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
