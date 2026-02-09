"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SMS SPAM CLASSIFIER — STREAMLIT APP                       ║
║══════════════════════════════════════════════════════════════════════════════║
║  A production-grade LSTM inference application for real-time SMS             ║
║  classification. Loads pre-trained model and tokenizer for instant           ║
║  spam detection with confidence visualization.                               ║
║                                                                              ║
║  Author  : Dinraj                                                            ║
║  Stack   : TensorFlow · Keras · Streamlit · Python                           ║
║  Model   : LSTM (Many-to-One) with trainable embeddings                      ║
║  Dataset : UCI SMS Spam Collection (5,574 messages)                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import pickle
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — Must match training hyperparameters exactly
# ══════════════════════════════════════════════════════════════════════════════
MAX_LEN = 40          # Sequence length (padding/truncating)
THRESHOLD = 0.5       # Classification boundary
VOCAB_SIZE = 8000     # Tokenizer vocabulary size
EMBED_DIM = 64        # Embedding dimension
LSTM_UNITS = 64       # LSTM hidden units

# ══════════════════════════════════════════════════════════════════════════════
# PATHS — Relative to project root
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "spam_lstm_model.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "model", "tokenizer.pkl")

# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE MESSAGES — For demo purposes
# ══════════════════════════════════════════════════════════════════════════════
SPAM_EXAMPLES = [
    "Congratulations! You've won a £1000 gift card. Call now to claim!",
    "URGENT! Your account has been compromised. Click here to verify.",
    "FREE entry in a weekly competition! Text WIN to 80080 now!",
    "You have been selected for a cash prize! Call 09061234567",
]

HAM_EXAMPLES = [
    "Hey, are we still meeting at 5pm today?",
    "Can you pick up some milk on the way home?",
    "I'll reach home by 8pm today. See you soon!",
    "Thanks for your help yesterday, really appreciate it.",
]


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING — Cached for performance
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """
    Load the trained LSTM model and fitted Keras tokenizer from disk.
    Uses @st.cache_resource to load only once across all sessions.
    """
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as fh:
        tokenizer = pickle.load(fh)
    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING PIPELINE — Identical to training
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(text: str, tokenizer) -> np.ndarray:
    """
    Replicate the exact preprocessing pipeline used during training:
    1. Tokenize (word-level, lowercased, OOV-aware)
    2. Pad / truncate to MAX_LEN (post-padding)
    
    This ensures train-serve parity — no skew between training and inference.
    """
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")
    return padded


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def classify(text: str, model, tokenizer) -> tuple[str, str, float]:
    """
    Run inference and return (label, confidence_level, confidence_percent).
    
    Confidence levels (qualitative — no raw probabilities exposed to user):
        • High   — model is very confident (distance from threshold ≥ 0.35)
        • Medium — model is moderately confident (distance ≥ 0.15)
        • Low    — prediction is borderline
    
    Returns:
        label: "SPAM" or "HAM"
        confidence: "High", "Medium", or "Low"
        confidence_percent: 0-100 for visual progress bar
    """
    padded = preprocess(text, tokenizer)
    prob = float(model.predict(padded, verbose=0)[0][0])
    
    label = "SPAM" if prob >= THRESHOLD else "HAM"
    
    # Distance from decision boundary
    distance = abs(prob - THRESHOLD)
    
    if distance >= 0.35:
        confidence = "High"
        confidence_percent = min(95, 70 + distance * 50)
    elif distance >= 0.15:
        confidence = "Medium"
        confidence_percent = 50 + distance * 100
    else:
        confidence = "Low"
        confidence_percent = 30 + distance * 100
    
    return label, confidence, confidence_percent


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS STYLES
# ══════════════════════════════════════════════════════════════════════════════
def inject_custom_css():
    """Inject modern, professional CSS styles for the application."""
    st.markdown(
        """
        <style>
        /* ═══════════════════════════════════════════════════════════════════ */
        /*                         GLOBAL STYLES                               */
        /* ═══════════════════════════════════════════════════════════════════ */
        
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* ═══════════════════════════════════════════════════════════════════ */
        /*                         HEADER SECTION                              */
        /* ═══════════════════════════════════════════════════════════════════ */
        
        .hero-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 16px;
            padding: 2.5rem 2rem;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
            position: relative;
            overflow: hidden;
        }
        
        .hero-container::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%);
            animation: pulse 4s ease-in-out infinite;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 0.5; }
            50% { transform: scale(1.1); opacity: 0.3; }
        }
        
        .hero-icon {
            font-size: 4rem;
            margin-bottom: 0.5rem;
            display: block;
            position: relative;
            z-index: 1;
        }
        
        .hero-title {
            font-size: 2.5rem;
            font-weight: 800;
            color: white;
            margin: 0 0 0.5rem 0;
            text-shadow: 0 2px 10px rgba(0,0,0,0.2);
            position: relative;
            z-index: 1;
        }
        
        .hero-subtitle {
            font-size: 1.1rem;
            color: rgba(255,255,255,0.9);
            margin: 0;
            font-weight: 400;
            position: relative;
            z-index: 1;
        }
        
        .hero-badges {
            margin-top: 1.5rem;
            display: flex;
            justify-content: center;
            gap: 0.5rem;
            flex-wrap: wrap;
            position: relative;
            z-index: 1;
        }
        
        .hero-badge {
            background: rgba(255,255,255,0.2);
            color: white;
            padding: 0.4rem 0.8rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.3);
        }
        
        /* ═══════════════════════════════════════════════════════════════════ */
        /*                         INPUT SECTION                               */
        /* ═══════════════════════════════════════════════════════════════════ */
        
        .input-section {
            background: white;
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            border: 1px solid #e2e8f0;
            margin-bottom: 1.5rem;
        }
        
        .input-label {
            font-size: 0.9rem;
            font-weight: 600;
            color: #374151;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .stTextArea textarea {
            border-radius: 12px !important;
            border: 2px solid #e2e8f0 !important;
            font-size: 1rem !important;
            padding: 1rem !important;
            transition: all 0.3s ease !important;
        }
        
        .stTextArea textarea:focus {
            border-color: #667eea !important;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15) !important;
        }
        
        /* ═══════════════════════════════════════════════════════════════════ */
        /*                         BUTTON STYLES                               */
        /* ═══════════════════════════════════════════════════════════════════ */
        
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.8rem 2rem !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        }
        
        .stButton > button[kind="primary"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 25px rgba(102, 126, 234, 0.5) !important;
        }
        
        .stButton > button[kind="secondary"] {
            background: white !important;
            color: #374151 !important;
            border: 2px solid #e2e8f0 !important;
            border-radius: 10px !important;
            font-size: 0.85rem !important;
            transition: all 0.2s ease !important;
        }
        
        .stButton > button[kind="secondary"]:hover {
            border-color: #667eea !important;
            color: #667eea !important;
            background: #f8faff !important;
        }
        
        /* ═══════════════════════════════════════════════════════════════════ */
        /*                         RESULT CARDS                                */
        /* ═══════════════════════════════════════════════════════════════════ */
        
        .result-container {
            margin-top: 1.5rem;
            animation: slideUp 0.5s ease-out;
        }
        
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .result-card {
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        
        .result-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
        }
        
        .spam-card {
            background: linear-gradient(180deg, #fff5f5 0%, #ffffff 100%);
            border: 2px solid #fc8181;
            box-shadow: 0 10px 40px rgba(245, 101, 101, 0.2);
        }
        
        .spam-card::before {
            background: linear-gradient(90deg, #f56565 0%, #ed64a6 100%);
        }
        
        .ham-card {
            background: linear-gradient(180deg, #f0fff4 0%, #ffffff 100%);
            border: 2px solid #68d391;
            box-shadow: 0 10px 40px rgba(72, 187, 120, 0.2);
        }
        
        .ham-card::before {
            background: linear-gradient(90deg, #48bb78 0%, #38b2ac 100%);
        }
        
        .result-icon {
            font-size: 4rem;
            margin-bottom: 0.5rem;
            display: block;
        }
        
        .result-label {
            font-size: 2.2rem;
            font-weight: 800;
            margin: 0.5rem 0;
            letter-spacing: -0.5px;
        }
        
        .spam-label { color: #c53030; }
        .ham-label { color: #276749; }
        
        .result-description {
            font-size: 1rem;
            color: #718096;
            margin: 0.5rem 0 1rem 0;
        }
        
        .confidence-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }
        
        .confidence-high {
            background: #c6f6d5;
            color: #276749;
        }
        
        .confidence-medium {
            background: #fefcbf;
            color: #975a16;
        }
        
        .confidence-low {
            background: #fed7d7;
            color: #c53030;
        }
        
        /* ═══════════════════════════════════════════════════════════════════ */
        /*                         EXAMPLES SECTION                            */
        /* ═══════════════════════════════════════════════════════════════════ */
        
        .examples-section {
            background: #f8fafc;
            border-radius: 16px;
            padding: 1.5rem;
            margin: 1.5rem 0;
            border: 1px solid #e2e8f0;
        }
        
        .examples-title {
            font-size: 1rem;
            font-weight: 700;
            color: #374151;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .example-btn {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 0.6rem 1rem;
            font-size: 0.8rem;
            color: #4a5568;
            cursor: pointer;
            transition: all 0.2s ease;
            text-align: left;
            margin: 0.3rem 0;
            width: 100%;
        }
        
        .example-btn:hover {
            border-color: #667eea;
            background: #f8faff;
            color: #667eea;
        }
        
        .example-spam::before {
            content: '🚨 ';
        }
        
        .example-ham::before {
            content: '✅ ';
        }
        
        /* ═══════════════════════════════════════════════════════════════════ */
        /*                         STATS CARDS                                 */
        /* ═══════════════════════════════════════════════════════════════════ */
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            margin: 1.5rem 0;
        }
        
        .stat-card {
            background: white;
            border-radius: 12px;
            padding: 1.2rem;
            text-align: center;
            border: 1px solid #e2e8f0;
            transition: all 0.3s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        }
        
        .stat-value {
            font-size: 1.5rem;
            font-weight: 800;
            color: #667eea;
            margin: 0;
        }
        
        .stat-label {
            font-size: 0.75rem;
            color: #718096;
            margin: 0.3rem 0 0 0;
            font-weight: 500;
        }
        
        /* ═══════════════════════════════════════════════════════════════════ */
        /*                         SIDEBAR STYLES                              */
        /* ═══════════════════════════════════════════════════════════════════ */
        
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1c2e 0%, #2d3047 100%);
        }
        
        [data-testid="stSidebar"] .stMarkdown {
            color: #e2e8f0;
        }
        
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: white !important;
        }
        
        .sidebar-section {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 1rem;
            margin: 1rem 0;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .sidebar-title {
            color: white;
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 0.8rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .sidebar-item {
            display: flex;
            justify-content: space-between;
            padding: 0.4rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            font-size: 0.85rem;
        }
        
        .sidebar-item:last-child {
            border-bottom: none;
        }
        
        .sidebar-key {
            color: rgba(255,255,255,0.6);
        }
        
        .sidebar-value {
            color: white;
            font-weight: 600;
        }
        
        /* ═══════════════════════════════════════════════════════════════════ */
        /*                         FOOTER STYLES                               */
        /* ═══════════════════════════════════════════════════════════════════ */
        
        .app-footer {
            text-align: center;
            padding: 2rem 0;
            margin-top: 3rem;
            border-top: 1px solid #e2e8f0;
        }
        
        .footer-text {
            color: #718096;
            font-size: 0.85rem;
            margin: 0;
        }
        
        .footer-links {
            margin-top: 1rem;
            display: flex;
            justify-content: center;
            gap: 1.5rem;
        }
        
        .footer-link {
            color: #667eea;
            text-decoration: none;
            font-size: 0.85rem;
            font-weight: 500;
            transition: color 0.2s ease;
        }
        
        .footer-link:hover {
            color: #764ba2;
        }
        
        /* ═══════════════════════════════════════════════════════════════════ */
        /*                         PIPELINE DIAGRAM                            */
        /* ═══════════════════════════════════════════════════════════════════ */
        
        .pipeline-container {
            background: #f8fafc;
            border-radius: 12px;
            padding: 1.2rem;
            margin: 1rem 0;
            border: 1px solid #e2e8f0;
        }
        
        .pipeline-step {
            display: flex;
            align-items: center;
            padding: 0.8rem;
            background: white;
            border-radius: 8px;
            margin: 0.5rem 0;
            border: 1px solid #e2e8f0;
            transition: all 0.2s ease;
        }
        
        .pipeline-step:hover {
            border-color: #667eea;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.15);
        }
        
        .pipeline-icon {
            font-size: 1.5rem;
            margin-right: 1rem;
            width: 40px;
            text-align: center;
        }
        
        .pipeline-info {
            flex: 1;
        }
        
        .pipeline-title {
            font-weight: 600;
            color: #374151;
            font-size: 0.9rem;
            margin: 0;
        }
        
        .pipeline-desc {
            color: #718096;
            font-size: 0.75rem;
            margin: 0.2rem 0 0 0;
        }
        
        .pipeline-arrow {
            text-align: center;
            color: #667eea;
            font-size: 1.2rem;
            margin: 0.3rem 0;
        }
        
        /* ═══════════════════════════════════════════════════════════════════ */
        /*                         RESPONSIVE ADJUSTMENTS                      */
        /* ═══════════════════════════════════════════════════════════════════ */
        
        @media (max-width: 768px) {
            .hero-title { font-size: 1.8rem; }
            .hero-subtitle { font-size: 0.95rem; }
            .stats-grid { grid-template-columns: repeat(2, 1fr); }
            .result-label { font-size: 1.8rem; }
        }
        
        /* ═══════════════════════════════════════════════════════════════════ */
        /*                         ANIMATIONS                                  */
        /* ═══════════════════════════════════════════════════════════════════ */
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .fade-in {
            animation: fadeIn 0.5s ease-out;
        }
        
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-5px); }
        }
        
        .bounce {
            animation: bounce 2s ease-in-out infinite;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════
def render_hero():
    """Render the animated hero header section."""
    st.markdown(
        """
        <div class="hero-container">
            <span class="hero-icon bounce">📩</span>
            <h1 class="hero-title">SMS Spam Classifier</h1>
            <p class="hero-subtitle">AI-powered spam detection using Deep Learning • LSTM Neural Network</p>
            <div class="hero-badges">
                <span class="hero-badge">🧠 LSTM</span>
                <span class="hero-badge">🔥 TensorFlow</span>
                <span class="hero-badge">⚡ Real-time</span>
                <span class="hero-badge">🎯 98% Accuracy</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stats():
    """Render model statistics cards."""
    st.markdown(
        f"""
        <div class="stats-grid">
            <div class="stat-card">
                <p class="stat-value">{VOCAB_SIZE:,}</p>
                <p class="stat-label">Vocabulary Size</p>
            </div>
            <div class="stat-card">
                <p class="stat-value">{MAX_LEN}</p>
                <p class="stat-label">Max Tokens</p>
            </div>
            <div class="stat-card">
                <p class="stat-value">{EMBED_DIM}</p>
                <p class="stat-label">Embed Dim</p>
            </div>
            <div class="stat-card">
                <p class="stat-value">{LSTM_UNITS}</p>
                <p class="stat-label">LSTM Units</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result(label: str, confidence: str, confidence_percent: float):
    """Render the classification result card with animation."""
    if label == "SPAM":
        card_class = "spam-card"
        label_class = "spam-label"
        icon = "🚨"
        description = "This message appears to be spam. Be cautious!"
    else:
        card_class = "ham-card"
        label_class = "ham-label"
        icon = "✅"
        description = "This message appears to be legitimate."
    
    confidence_class = f"confidence-{confidence.lower()}"
    confidence_icon = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(confidence, "⚪")
    
    st.markdown(
        f"""
        <div class="result-container">
            <div class="result-card {card_class}">
                <span class="result-icon">{icon}</span>
                <h2 class="result-label {label_class}">{label}</h2>
                <p class="result-description">{description}</p>
                <span class="confidence-badge {confidence_class}">
                    {confidence_icon} Confidence: {confidence}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Render the enhanced sidebar with model details."""
    with st.sidebar:
        # Logo and title
        st.markdown(
            """
            <div style="text-align: center; padding: 1rem 0;">
                <span style="font-size: 3rem;">🧠</span>
                <h2 style="color: white; margin: 0.5rem 0;">Model Info</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Architecture section
        st.markdown(
            """
            <div class="sidebar-section">
                <div class="sidebar-title">🏗️ Architecture</div>
                <div class="sidebar-item">
                    <span class="sidebar-key">Type</span>
                    <span class="sidebar-value">LSTM (Many-to-One)</span>
                </div>
                <div class="sidebar-item">
                    <span class="sidebar-key">Embedding</span>
                    <span class="sidebar-value">Trainable, 64-dim</span>
                </div>
                <div class="sidebar-item">
                    <span class="sidebar-key">LSTM Units</span>
                    <span class="sidebar-value">64</span>
                </div>
                <div class="sidebar-item">
                    <span class="sidebar-key">Dropout</span>
                    <span class="sidebar-value">0.2</span>
                </div>
                <div class="sidebar-item">
                    <span class="sidebar-key">Output</span>
                    <span class="sidebar-value">Sigmoid</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Training section
        st.markdown(
            """
            <div class="sidebar-section">
                <div class="sidebar-title">📊 Training</div>
                <div class="sidebar-item">
                    <span class="sidebar-key">Dataset</span>
                    <span class="sidebar-value">UCI SMS Spam</span>
                </div>
                <div class="sidebar-item">
                    <span class="sidebar-key">Samples</span>
                    <span class="sidebar-value">5,574</span>
                </div>
                <div class="sidebar-item">
                    <span class="sidebar-key">Epochs</span>
                    <span class="sidebar-value">8</span>
                </div>
                <div class="sidebar-item">
                    <span class="sidebar-key">Imbalance</span>
                    <span class="sidebar-value">Class Weights</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Performance section
        st.markdown(
            """
            <div class="sidebar-section">
                <div class="sidebar-title">🎯 Performance</div>
                <div class="sidebar-item">
                    <span class="sidebar-key">Precision</span>
                    <span class="sidebar-value">0.98</span>
                </div>
                <div class="sidebar-item">
                    <span class="sidebar-key">Recall</span>
                    <span class="sidebar-value">0.98</span>
                </div>
                <div class="sidebar-item">
                    <span class="sidebar-key">F1-Score</span>
                    <span class="sidebar-value">0.98</span>
                </div>
                <div class="sidebar-item">
                    <span class="sidebar-key">Spam Recall</span>
                    <span class="sidebar-value">0.95</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Pipeline visualization
        st.markdown(
            """
            <div class="sidebar-section">
                <div class="sidebar-title">🔄 Pipeline</div>
                <div class="pipeline-step">
                    <span class="pipeline-icon">📝</span>
                    <div class="pipeline-info">
                        <p class="pipeline-title">Input Text</p>
                        <p class="pipeline-desc">Raw SMS message</p>
                    </div>
                </div>
                <div class="pipeline-arrow">↓</div>
                <div class="pipeline-step">
                    <span class="pipeline-icon">🔤</span>
                    <div class="pipeline-info">
                        <p class="pipeline-title">Tokenization</p>
                        <p class="pipeline-desc">Word-level, 8000 vocab</p>
                    </div>
                </div>
                <div class="pipeline-arrow">↓</div>
                <div class="pipeline-step">
                    <span class="pipeline-icon">📐</span>
                    <div class="pipeline-info">
                        <p class="pipeline-title">Padding</p>
                        <p class="pipeline-desc">40 tokens, post-pad</p>
                    </div>
                </div>
                <div class="pipeline-arrow">↓</div>
                <div class="pipeline-step">
                    <span class="pipeline-icon">🧠</span>
                    <div class="pipeline-info">
                        <p class="pipeline-title">LSTM Inference</p>
                        <p class="pipeline-desc">Sequence classification</p>
                    </div>
                </div>
                <div class="pipeline-arrow">↓</div>
                <div class="pipeline-step">
                    <span class="pipeline-icon">🏷️</span>
                    <div class="pipeline-info">
                        <p class="pipeline-title">Classification</p>
                        <p class="pipeline-desc">SPAM or HAM</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Footer
        st.markdown(
            """
            <div style="text-align: center; padding: 1.5rem 0; margin-top: 1rem; border-top: 1px solid rgba(255,255,255,0.1);">
                <p style="color: rgba(255,255,255,0.5); font-size: 0.75rem; margin: 0;">
                    Built by <strong style="color: white;">Dinraj</strong>
                </p>
                <p style="color: rgba(255,255,255,0.4); font-size: 0.7rem; margin: 0.3rem 0 0 0;">
                    TensorFlow • Keras • Streamlit
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_footer():
    """Render the application footer."""
    st.markdown(
        """
        <div class="app-footer">
            <p class="footer-text">
                🧠 Powered by LSTM Deep Learning • Built with TensorFlow & Streamlit
            </p>
            <p class="footer-text" style="margin-top: 0.5rem;">
                📊 Trained on UCI SMS Spam Collection Dataset (5,574 messages)
            </p>
            <div class="footer-links">
                <a href="https://github.com/dinraj910" class="footer-link" target="_blank">
                    GitHub
                </a>
                <a href="https://linkedin.com/in/dinraj910" class="footer-link" target="_blank">
                    LinkedIn
                </a>
            </div>
            <p style="color: #a0aec0; font-size: 0.75rem; margin-top: 1rem;">
                © 2026 Dinraj • SMS Spam Classifier v1.0
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════
def main():
    """Main application entry point."""
    
    # Page configuration
    st.set_page_config(
        page_title="SMS Spam Classifier | LSTM Deep Learning",
        page_icon="📩",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://github.com/dinraj910/Spam-SMS-Classification-using-LSTM-RNN-",
            "Report a bug": "https://github.com/dinraj910/Spam-SMS-Classification-using-LSTM-RNN-/issues",
            "About": "SMS Spam Classifier powered by LSTM Neural Network. Built by Dinraj.",
        }
    )
    
    # Inject custom CSS
    inject_custom_css()
    
    # Render sidebar
    render_sidebar()
    
    # Main content area
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Hero section
        render_hero()
        
        # Stats cards
        render_stats()
        
        # Load model (with loading indicator)
        with st.spinner("🔄 Loading LSTM model..."):
            model, tokenizer = load_artifacts()
        
        # Input section
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown(
            '<p class="input-label">✉️ Enter your SMS message below</p>',
            unsafe_allow_html=True,
        )
        
        # Initialize session state for input
        if "sms_input" not in st.session_state:
            st.session_state.sms_input = ""
        
        sms_input = st.text_area(
            "SMS Input",
            value=st.session_state.sms_input,
            height=120,
            placeholder="Type or paste an SMS message here to check if it's spam...",
            label_visibility="collapsed",
            key="sms_text_area",
        )
        
        # Classify button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            classify_btn = st.button(
                "🔍 Classify Message",
                type="primary",
                use_container_width=True,
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Classification logic
        if classify_btn:
            cleaned = sms_input.strip()
            if not cleaned:
                st.warning("⚠️ Please enter a valid SMS message before classifying.")
            else:
                with st.spinner("🧠 Analyzing message..."):
                    label, confidence, confidence_percent = classify(cleaned, model, tokenizer)
                render_result(label, confidence, confidence_percent)
        
        # Example messages section
        st.markdown(
            """
            <div class="examples-section">
                <p class="examples-title">💡 Try these example messages</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Example buttons in columns
        col_spam, col_ham = st.columns(2)
        
        with col_spam:
            st.markdown("**🚨 Spam Examples**")
            for i, example in enumerate(SPAM_EXAMPLES):
                if st.button(
                    f"📛 {example[:40]}..." if len(example) > 40 else f"📛 {example}",
                    key=f"spam_{i}",
                    use_container_width=True,
                ):
                    st.session_state.sms_input = example
                    st.rerun()
        
        with col_ham:
            st.markdown("**✅ Ham Examples**")
            for i, example in enumerate(HAM_EXAMPLES):
                if st.button(
                    f"💬 {example[:40]}..." if len(example) > 40 else f"💬 {example}",
                    key=f"ham_{i}",
                    use_container_width=True,
                ):
                    st.session_state.sms_input = example
                    st.rerun()
        
        # How it works section
        with st.expander("🔬 How does it work?", expanded=False):
            st.markdown(
                """
                ### The LSTM Spam Classification Pipeline
                
                1. **Text Input** — Your raw SMS message is received
                2. **Tokenization** — Words are converted to numerical tokens using a vocabulary of 8,000 words
                3. **OOV Handling** — Unknown words are replaced with a special `<OOV>` token
                4. **Padding** — Sequences are padded or truncated to exactly 40 tokens
                5. **Embedding** — Tokens are mapped to 64-dimensional dense vectors
                6. **LSTM Processing** — The sequence is processed by a 64-unit LSTM layer
                7. **Classification** — A sigmoid output layer produces a probability (≥0.5 = SPAM)
                
                ### Why LSTM?
                
                SMS messages are **sequences** where word order matters:
                - *"Call me now"* → HAM
                - *"Call now to win"* → SPAM
                
                Traditional bag-of-words models miss these patterns. LSTMs capture **sequential dependencies**.
                
                ### Handling Class Imbalance
                
                The dataset is heavily imbalanced (~87% ham, ~13% spam). We use **balanced class weights**
                during training to ensure the model doesn't just predict "ham" for everything.
                """
            )
        
        # Technical details
        with st.expander("📊 Technical Details", expanded=False):
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown(
                    """
                    #### Model Architecture
                    ```
                    Embedding(8000, 64, mask_zero=True)
                           ↓
                    LSTM(64, dropout=0.2)
                           ↓
                    Dense(1, activation='sigmoid')
                    ```
                    
                    #### Hyperparameters
                    | Parameter | Value |
                    |:---|:---:|
                    | Vocabulary | 8,000 |
                    | Max Length | 40 |
                    | Embedding Dim | 64 |
                    | LSTM Units | 64 |
                    | Dropout | 0.2 |
                    | Batch Size | 32 |
                    | Epochs | 8 |
                    """
                )
            
            with col_b:
                st.markdown(
                    """
                    #### Performance Metrics
                    | Class | Precision | Recall | F1 |
                    |:---|:---:|:---:|:---:|
                    | Ham | 0.99 | 0.98 | 0.99 |
                    | Spam | 0.92 | 0.95 | 0.93 |
                    | **Avg** | **0.98** | **0.98** | **0.98** |
                    
                    #### Dataset
                    - **Name:** UCI SMS Spam Collection
                    - **Size:** 5,574 messages
                    - **Split:** 80% train / 20% test
                    - **Stratified:** Yes
                    """
                )
        
        # Footer
        render_footer()


if __name__ == "__main__":
    main()
