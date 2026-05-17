import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
import numpy as np
st.set_page_config(page_title="Smart Categorizer", page_icon="🏦", layout="centered")

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

category_mapping = {
    0: "Charity & Donations ",
    1: "Entertainment & Recreation ",
    2: "Financial Services ",
    3: "Food & Dining ",
    4: "Government & Legal ",
    5: "Healthcare & Medical ",
    6: "Income ",
    7: "Shopping & Retail ",
    8: "Transportation ",
    9: "Utilities & Services "
}

@st.cache_resource 
def load_assets():
    lr_model = joblib.load('logistic_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return lr_model, vectorizer

model, tfidf = load_assets()

def clean_text(text):
    text = text.lower()
    for ch in string.punctuation:
        text = text.replace(ch, '')
    words = text.split()
    clean_words = [word for word in words if word not in stop_words]
    return ' '.join(clean_words)


with st.sidebar:
    st.header(" About the Model")
    st.write("This AI model uses Natural Language Processing (NLP) and Logistic Regression to classify bank transactions.")
    st.markdown("---")
    st.info("**Try testing these:** \n- `AMZN Mktp US` \n- `UBER TRIP SF` \n- `APOLLO CLINIC` \n- `PAYROLL REF`")


st.title(" Smart Transaction Categorizer")
st.markdown("Enter a raw bank transaction description below to predict its category instantly.")
st.markdown("---")

user_input = st.text_input("Transaction Description", placeholder="e.g., NETFLIX.COM PAYMENT")


if st.button(" Categorize Transaction", use_container_width=True):
    if user_input:
        with st.spinner("Analyzing data..."):
            cleaned_input = clean_text(user_input)
            vectorized_input = tfidf.transform([cleaned_input])
            
            
            prediction_code = model.predict(vectorized_input)[0]
            predicted_name = category_mapping.get(prediction_code, "Unknown Category")
            
            
            probabilities = model.predict_proba(vectorized_input)[0]
            confidence = np.max(probabilities) * 100
            
        st.markdown("###  Analysis Result")
        
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"**Category:**\n### {predicted_name}")
            
        with col2:
            st.info(f"**AI Confidence:**\n### {confidence:.2f}%")
      
        st.progress(int(confidence))
        
    else:
        st.warning(" Please enter a transaction description.")