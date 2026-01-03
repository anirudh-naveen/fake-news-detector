"""
Fake News Detection - Web Application
Streamlit-based web interface for fake news detection
"""

import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import config
from predict import load_model, predict_news

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 2rem 0;
    }
    .real-news {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .fake-news {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📰 Fake News Detector</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("About")
    st.write("""
    This application uses a deep learning model to classify news articles as **REAL** or **FAKE**.
    
    **Model Architecture:**
    - LSTM + CNN neural network
    - GloVe word embeddings
    - Trained on 3,000 news articles
    
    **How it works:**
    1. Enter a news headline or article text
    2. The model analyzes the text
    3. Get instant prediction with confidence score
    """)
    
    st.markdown("---")
    st.header("Model Info")
    st.write(f"**Max Length:** {config.MAX_LENGTH} words")
    st.write(f"**Embedding Dim:** {config.EMBEDDING_DIM}")
    st.write("**Accuracy:** ~74% (test set)")

# Load model (cached)
@st.cache_resource
def get_model():
    """Load model and tokenizer (cached for performance)"""
    return load_model()

# Main content
st.header("Enter News Text")

# Text input
text_input = st.text_area(
    "Paste news headline or article text here:",
    height=150,
    placeholder="Example: Scientists discover new planet with potential for life..."
)

# Prediction button
if st.button("🔍 Analyze News", type="primary", use_container_width=True):
    if not text_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Loading model and analyzing..."):
            # Load model
            model, tokenizer = get_model()
            
            if model is None or tokenizer is None:
                st.error("Error: Model not found. Please train the model first by running: `python detector.py`")
            else:
                # Make prediction
                label, confidence = predict_news(text_input, model, tokenizer)
                
                # Display result
                st.markdown("---")
                st.header("Prediction Result")
                
                # Confidence bar
                st.progress(confidence if label == "REAL" else (1 - confidence))
                
                # Result box
                if label == "REAL":
                    st.markdown(f"""
                    <div class="prediction-box real-news">
                        <h2 style="color: #28a745; text-align: center;">✓ This news appears to be REAL</h2>
                        <p style="text-align: center; font-size: 1.2rem;">
                            Confidence: <strong>{confidence*100:.2f}%</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box fake-news">
                        <h2 style="color: #dc3545; text-align: center;">✗ This news appears to be FAKE</h2>
                        <p style="text-align: center; font-size: 1.2rem;">
                            Confidence: <strong>{(1-confidence)*100:.2f}%</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Additional info
                with st.expander("📊 Detailed Information"):
                    st.write(f"**Raw Prediction Score:** {confidence:.4f}")
                    st.write(f"**Threshold:** 0.5 (≥0.5 = REAL, <0.5 = FAKE)")
                    st.write(f"**Text Length:** {len(text_input)} characters")
                    st.write(f"**Word Count:** {len(text_input.split())} words")

# Auto-fill text area if example selected
if 'example_text' in st.session_state:
    text_input = st.text_area(
        "Paste news headline or article text here:",
        value=st.session_state.example_text,
        height=150
    )
    del st.session_state.example_text

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>Built with TensorFlow, Keras, and Streamlit</p>
    <p>Model trained on fake news dataset with LSTM-CNN architecture</p>
</div>
""", unsafe_allow_html=True)
