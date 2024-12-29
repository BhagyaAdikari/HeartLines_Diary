# Core Packages
import streamlit as st

# EDA Packages
import pandas as pd
import numpy as np

# Model Loading
import joblib

# Load the pre-trained emotion classification model
@st.cache_resource
def load_model():
    return joblib.load("models/emotion_classifier_pipe_lr.pkl")  # Replace with your actual model file

# Main Function
def main():
    st.title("Emotion Classifier App")
    menu = ["Home", "Monitor", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Load Model
    model = load_model()

    if choice == "Home":
        st.subheader("Home - Emotion in Text")

        with st.form(key='myform'):
            raw_text = st.text_area("Type Here")
            submit_button = st.form_submit_button("Analyze")

        if submit_button:
            if raw_text.strip():
                # Make Prediction
                prediction = model.predict([raw_text])
                emotion = prediction[0]
                st.success(f"Detected Emotion: **{emotion}**")
            else:
                st.warning("Please enter some text!")

    elif choice == "Monitor":
        st.subheader("Monitor App")
        st.write("Monitoring functionality coming soon!")

    else:
        st.subheader("About")
        st.write("""
        This app classifies the emotion in a given text using a machine learning model.
        Built with Streamlit and Python.
        """)

if __name__ == '__main__':
    main()
