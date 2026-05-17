from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import string
import nltk
from nltk.corpus import stopwords
import numpy as np

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

model = joblib.load('logistic_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Transaction(BaseModel):
    description: str

def clean_text(text):
    text = text.lower()
    for ch in string.punctuation:
        text = text.replace(ch, '')
    words = text.split()
    clean_words = [word for word in words if word not in stop_words]
    return ' '.join(clean_words)

@app.post("/predict")
def predict_category(transaction: Transaction):
    cleaned_input = clean_text(transaction.description)
    vectorized_input = tfidf.transform([cleaned_input])
    prediction_code = int(model.predict(vectorized_input)[0])
    probabilities = model.predict_proba(vectorized_input)[0]
    confidence = float(np.max(probabilities) * 100)
    return {
        "success": True,
        "predicted_category_name": category_mapping.get(prediction_code, "Unknown"),
        "confidence_score": round(confidence, 2)
    }