import streamlit as st
import pickle as pkl
from sklearn.feature_extraction.text import TfidfVectorizer

# Page configuration
st.set_page_config(page_title="Emotion Insight", page_icon="💭", layout="centered")

# Load ML components
classifier = pkl.load(open('model.pkl', 'rb'))
vectorizer = pkl.load(open('scaler.pkl', 'rb'))

# Emoji icon
icon_url = "https://cdn-icons-png.flaticon.com/512/4712/4712027.png"

# UI Styling with bright and English-style color palette
st.markdown("""
    <style>
    html, body, .stApp {
        background: linear-gradient(135deg, #fdfbfb, #ebedee);
        font-family: 'Segoe UI', sans-serif;
        color: #333;
    }

    .main-header {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        color: #4a148c;
        margin-bottom: 0.2rem;
    }

    .main-header img {
        height: 45px;
        margin-right: 10px;
        margin-bottom: 5px;
        vertical-align: middle;
    }

    .subtext {
        text-align: center;
        font-size: 1.15rem;
        color: #6a1b9a;
        margin-bottom: 2rem;
        font-weight: 500;
    }

    .input-label {
        font-size: 1.3rem;
        color: #00695c;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .stTextInput input {
        border-radius: 12px;
        background-color: #ffffff;
        border: 2px solid #64b5f6;
        padding: 12px;
        font-size: 1rem;
        color: #1a237e;
        font-weight: 500;
    }

    .stButton > button {
        background-image: linear-gradient(to right, #64b5f6, #2196f3);
        color: white;
        font-weight: 600;
        border-radius: 25px;
        padding: 10px 25px;
        border: none;
        font-size: 1rem;
        transition: all 0.3s ease-in-out;
    }

    .stButton > button:hover {
        background-image: linear-gradient(to right, #1976d2, #1e88e5);
        box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
    }

    .footer {
        margin-top: 3rem;
        text-align: center;
        color: #37474f;
        font-size: 0.95rem;
    }

    .marquee {
        overflow: hidden;
        white-space: nowrap;
    }

    .marquee span {
        display: inline-block;
        animation: marqueeAnim 16s linear infinite;
        font-style: italic;
        font-weight: 700;
        font-family: 'Verdana', sans-serif;
        color: #c62828;
    }

    @keyframes marqueeAnim {
        0% { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown(f'<div class="main-header"><img src="{icon_url}" />Emotion Insight</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Reveal the tone behind your message using machine learning</div>', unsafe_allow_html=True)

# Input area
st.markdown('<div class="input-label">Write your message below:</div>', unsafe_allow_html=True)
user_text = st.text_input(label="", placeholder="Example: This app just made my day!", key="input_text")

# Prediction logic
if st.button("✨ Reveal Emotion"):
    if user_text.strip() == "":
        st.warning("🚫 Please enter a message to analyze.")
    else:
        transformed_text = vectorizer.transform([user_text]).toarray()
        emotion = classifier.predict(transformed_text)

        if emotion[0] == 0:
            st.error("😢 The sentiment appears **Negative**.", icon="❌")
        else:
            st.success("😄 The sentiment appears **Positive**.", icon="✅")

# Footer with scrolling name
st.markdown("""
    <div class="footer">
        <div class="marquee">
            <span>Developed with ❤️ by Poonam Bhatt • Developed with ❤️ by Poonam Bhatt • </span>
        </div>
    </div>
""", unsafe_allow_html=True)
