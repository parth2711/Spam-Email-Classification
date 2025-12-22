import streamlit as st
import numpy as np
import pandas as pd
import pickle

#loading the model and scaler
with open('model.pkl','rb') as file:
    model=pickle.load(file)
with open('tfidf.pkl','rb') as file2:
    tfidf=pickle.load(file2)

#proceeding with ui
st.title('Spam Email Classification')

st.markdown("""
### About the Model
This app uses a Support Vector Machine model (Linear Support Vector Classifier) trained on a spam messages dataset.
""")

st.subheader("Enter message received in mail")

message=st.text_input('Message')

messagetfidf=tfidf.transform([message])
    
pred=model.predict(messagetfidf)[0]

st.divider()
st.subheader("Prediction Result")

if pred=='spam':
    st.error("This message is SPAM") 
else:
    st.success("This message is NOT SPAM (ham)")