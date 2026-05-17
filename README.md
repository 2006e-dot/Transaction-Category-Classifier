# Transaction Category Classifier

An NLP + Deep Learning based web application that predicts transaction categories from financial transaction descriptions.

Built using TensorFlow/Keras for model training and FastAPI for backend deployment.

---

# Features

- Financial transaction category prediction
- NLP preprocessing pipeline
- TF-IDF vectorization
- Deep Learning + Logistic Regression models
- FastAPI backend integration
- Simple frontend interface using HTML
- Real-time prediction support

---

# Tech Stack

- Python
- TensorFlow / Keras
- Scikit-learn
- FastAPI
- HTML
- NumPy
- Pandas

---

# Project Structure

```bash
.
├── 1.ipynb
├── api.py
├── app.py
├── index.html
├── logistic_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Dataset Access

The dataset used in this project is hosted on Hugging Face because GitHub has strict file size limitations for large datasets.

## Official Dataset Link

https://huggingface.co/datasets/mitulshah/transaction-categorization

---

# How to Access the Dataset

## Step 1: Install Required Libraries

Run the following command in terminal:

```bash
pip install datasets huggingface_hub
```

---

## Step 2: Generate Hugging Face Access Token

1. Create/Login to your Hugging Face account.
2. Open:

https://huggingface.co/settings/tokens

3. Click on `New Token`
4. Give the token any name.
5. Select `Read` permission.
6. Generate the token.
7. Copy the generated token.

---

## Step 3: Login Using Token

Run the following command:

```bash
huggingface-cli login
```

Terminal will ask:

```text
Enter your token:
```

Paste your Hugging Face token and press Enter.

---

## Step 4: Download Dataset

```python
from datasets import load_dataset

dataset = load_dataset("mitulshah/transaction-categorization")
```

---

## Step 5: Convert Dataset to CSV

```python
dataset["train"].to_csv("transactions.csv")
```

After running the code, the CSV dataset file will automatically be generated inside the project directory.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/2006-dot/Transaction-Category-Classifier.git
cd Transaction-Category-Classifier
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the FastAPI Server

Start the backend server using:

```bash
uvicorn api:app --reload
```

The API server will start locally.

---

# Running Frontend

Open `index.html` in your browser.

The frontend communicates with the FastAPI backend for transaction category predictions.

---

# Model Workflow

1. Load financial transaction dataset
2. Apply NLP preprocessing
3. Convert text using TF-IDF vectorization
4. Train classification models
5. Predict transaction categories
6. Serve predictions using FastAPI

---

# Future Improvements

- Transformer-based NLP models
- Better UI/UX
- Docker deployment
- Cloud hosting
- Improved model accuracy
- Authentication system

---
