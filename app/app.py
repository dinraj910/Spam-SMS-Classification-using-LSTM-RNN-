"""
Spam SMS Classification — Streamlit Inference App
===================================================
Loads a pre-trained LSTM model and Keras tokenizer to classify
SMS messages as SPAM or HAM in real time.

Author : Dinraj
Stack  : TensorFlow / Keras · Streamlit · Python
"""

import os
import pickle
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ──────────────────────────────────────────────
# CONSTANTS (must match training configuration)
# ──────────────────────────────────────────────
MAX_LEN = 40
THRESHOLD = 0.5

# ──────────────────────────────────────────────
# PATHS  (relative to project root)
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "spam_lstm_model.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "model", "tokenizer.pkl")


# ──────────────────────────────────────────────
# LOAD ARTIFACTS  (cached so they load only once)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def load_artifacts():
    """Load the trained LSTM model and fitted tokenizer from disk."""
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as fh:
        tokenizer = pickle.load(fh)
    return model, tokenizer


# ──────────────────────────────────────────────
# PREPROCESSING
# ──────────────────────────────────────────────
def preprocess(text: str, tokenizer) -> np.ndarray:
    """
    Replicate the exact preprocessing pipeline used during training:
    1. Tokenize (word-level, lowercased, OOV-aware)
    2. Pad / truncate to MAX_LEN (post)
    """
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")
    return padded


# ──────────────────────────────────────────────
# INFERENCE
# ──────────────────────────────────────────────
def classify(text: str, model, tokenizer) -> tuple[str, str]:
    """
    Run inference and return a (label, confidence_level) tuple.

    Confidence levels (qualitative only — no raw probabilities exposed):
        • High   — model is very confident
        • Medium — model is moderately confident
        • Low    — prediction is borderline
    """
    padded = preprocess(text, tokenizer)
    prob = float(model.predict(padded, verbose=0)[0][0])

    label = "SPAM" if prob >= THRESHOLD else "HAM"

    # Qualitative confidence bucket
    distance = abs(prob - THRESHOLD)
    if distance >= 0.35:
        confidence = "High"
    elif distance >= 0.15:
        confidence = "Medium"
    else:
        confidence = "Low"

    return label, confidence


# ──────────────────────────────────────────────
# STREAMLIT UI
# ──────────────────────────────────────────────
def main():
    # ── Page config ──────────────────────────
    st.set_page_config(
        page_title="SMS Spam Classifier",
        page_icon="📩",
        layout="centered",
    )

    # ── Custom styles ────────────────────────
    st.markdown(
        """
        <style>
        .result-box {
            padding: 1.2rem 1.5rem;
            border-radius: 0.6rem;
            text-align: center;
            margin-top: 1rem;
        }
        .spam-box {
            background-color: #fde8e8;
            border: 2px solid #e53e3e;
        }
        .ham-box {
            background-color: #e6ffed;
            border: 2px solid #38a169;
        }
        .label-text {
            font-size: 2rem;
            font-weight: 700;
            margin: 0;
        }
        .confidence-text {
            font-size: 0.95rem;
            margin-top: 0.3rem;
            color: #555;
        }
        .app-footer {
            text-align: center;
            color: #888;
            font-size: 0.8rem;
            margin-top: 3rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Header ───────────────────────────────
    st.title("📩 SMS Spam Classifier")
    st.caption(
        "Powered by a single-layer LSTM trained on the UCI SMS Spam Collection. "
        "Enter any SMS message below to check whether it is **spam** or **ham**."
    )

    # ── Load model + tokenizer ───────────────
    model, tokenizer = load_artifacts()

    # ── Input area ───────────────────────────
    sms_input = st.text_area(
        "Enter SMS message",
        height=120,
        placeholder="Type or paste an SMS message here …",
    )

    classify_btn = st.button("Classify", type="primary", use_container_width=True)

    # ── Classification logic ─────────────────
    if classify_btn:
        # Guard: empty / whitespace-only input
        cleaned = sms_input.strip()
        if not cleaned:
            st.warning("Please enter a valid SMS message before classifying.")
            return

        with st.spinner("Analyzing …"):
            label, confidence = classify(cleaned, model, tokenizer)

        # Render result
        if label == "SPAM":
            st.markdown(
                '<div class="result-box spam-box">'
                '<p class="label-text">🚨 SPAM</p>'
                f'<p class="confidence-text">Confidence: {confidence}</p>'
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-box ham-box">'
                '<p class="label-text">✅ HAM</p>'
                f'<p class="confidence-text">Confidence: {confidence}</p>'
                "</div>",
                unsafe_allow_html=True,
            )

    # ── Sidebar — project context ────────────
    with st.sidebar:
        st.header("About")
        st.markdown(
            """
            **Model:** LSTM (Many-to-One)  
            **Embedding:** Trainable, 64-dim  
            **Vocabulary:** 8 000 words  
            **Max length:** 40 tokens  
            **Class imbalance:** Handled via class weights  
            **Evaluation:** Precision · Recall · F1-score  
            """
        )
        st.divider()
        st.markdown(
            """
            **How it works**  
            1. Your SMS text is tokenized using the same word-level tokenizer from training.  
            2. The token sequence is padded to 40 tokens.  
            3. The LSTM model outputs a binary prediction — SPAM or HAM.  
            """
        )
        st.divider()
        st.markdown(
            """
            **Try these examples**  
            - *Congratulations! You've won a £1000 gift card. Call now!*  
            - *Hey, are we still meeting at 5pm today?*  
            - *URGENT! Your account has been compromised. Click here to verify.*  
            - *Can you pick up some milk on the way home?*  
            """
        )

    # ── Footer ───────────────────────────────
    st.markdown(
        '<p class="app-footer">'
        "Built with TensorFlow &amp; Streamlit · SMS Spam Classification (LSTM)"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
