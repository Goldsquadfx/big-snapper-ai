import streamlit as st
from PIL import Image
from transformers import pipeline
import random

# --- Page setup ---
st.set_page_config(page_title="Big Snapper AI Analyzer", layout="centered")
st.title("📈 Big Snapper – AI Chart Analyzer")
st.write("Upload a trading chart image to get an instant trade suggestion.")

# --- File uploader ---
uploaded = st.file_uploader("Upload your chart (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Chart", use_column_width=True)
    st.write("🔍 Analyzing your chart...")

    # --- Vision model ---
    classifier = pipeline("image-classification", model="microsoft/resnet-50")
    results = classifier(image)

    label = results[0]["label"].lower()
    conf = round(results[0]["score"] * 100, 2)

    # --- Basic trade decision ---
    if any(word in label for word in ["up", "bull", "trend", "ascending"]):
        direction = "BUY"
    elif any(word in label for word in ["down", "bear", "decline", "descending"]):
        direction = "SELL"
    else:
        direction = "WAIT / NEUTRAL"

    # --- Generate fake numeric levels (example only) ---
    current_price = random.uniform(1.2000, 1.3000)
    if direction == "BUY":
        tp = current_price * 1.005   # +0.5%
        sl = current_price * 0.9975  # -0.25%
    elif direction == "SELL":
        tp = current_price * 0.995
        sl = current_price * 1.0025
    else:
        tp = sl = current_price

    # --- Display output ---
    st.markdown(f"""
    ### 🧭 Trade Suggestion
    **Signal:** {direction}  
    **Confidence:** {conf}%  
    **Entry:** {current_price:.4f}  
    **Take Profit:** {tp:.4f}  
    **Stop Loss:** {sl:.4f}  
    **Reasoning:** Detected pattern hinting *{label}*
    """)
