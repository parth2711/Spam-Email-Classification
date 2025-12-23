import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.set_page_config(page_title="Spam Email Classifier",page_icon="📧",layout="centered")

#loading the model and scaler
with open('model.pkl','rb') as file:
    model=pickle.load(file)
with open('tfidf.pkl','rb') as file2:
    tfidf=pickle.load(file2)

#proceeding with ui
st.markdown("""
    <h1 style='text-align: center;'>📧 Spam Email Classification</h1>
    <p style='text-align: center; color: gray;'>
    Detect if a message is spam or not using SVM.
    </p>
    """,unsafe_allow_html=True)

st.divider()

with st.expander("ℹ️ About this app"):
    st.write(
        """
        This app uses a **Linear Support Vector Machine (SVM)** trained on a real world SMS spam dataset.

        Model pipeline:

        -> Text preprocessing

        -> Data Splitting

        -> Vectorization

        -> Fitting Linear Support Vector Classifier

        The model predicts whether a given message is **Spam** or **Not Spam (Ham)**.
        """
    )

st.subheader("Enter message received in mail")

message = st.text_area(label="Paste the email/SMS content below",height=180)

messagetfidf=tfidf.transform([message])
    
pred=model.predict(messagetfidf)[0]

st.divider()
st.subheader("Prediction Result")

if pred=='spam':
    st.error("This message is SPAM") 
    st.markdown("<p style='color:red;'>Be cautious before clicking links or sharing personal information.</p>",unsafe_allow_html=True)
else:
    st.success("This message is NOT SPAM (ham)")
    st.markdown("<p style='color:green;'>This message appears to be safe.</p>",unsafe_allow_html=True)