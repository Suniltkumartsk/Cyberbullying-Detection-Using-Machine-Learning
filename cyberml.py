# ==========================================
# CYBERBULLYING DETECTION USING MACHINE LEARNING
# ==========================================

import pandas as pd
import numpy as np
import re

# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv("cyber.csv")

print("Dataset Info:\n")
print(df.info())

print("\nMissing Values:\n")
print(df.isnull().sum())

# Drop missing values
df.dropna(inplace=True)

# =========================
# 2. Text Cleaning Function
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['tweet_text'] = df['tweet_text'].apply(clean_text)

# =========================
# 3. Features & Labels
# =========================
X = df['tweet_text']
y = df['cyberbullying_type']

print("\nUnique Classes:\n", y.unique())

# =========================
# 4. TF-IDF Vectorization
# =========================
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    stop_words='english'
)

X = vectorizer.fit_transform(X)

# =========================
# 5. Train-Test Split
# =========================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 6. Model Training
# =========================
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=1000,
    solver='lbfgs',
   
)

model.fit(X_train, y_train)

# =========================
# 7. Model Evaluation
# =========================
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

y_pred = model.predict(X_test)

print("\n===================================")
print("Model Performance")
print("===================================")

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# =========================
# 8. User Input Prediction
# =========================
print("\n===================================")
print("Cyberbullying Detection System Ready")
print("Type 'exit' to quit")
print("===================================")

while True:
    user_input = input("\nEnter a sentence: ")

    if user_input.lower() == "exit":
        print("Exiting program...")
        break

    cleaned = clean_text(user_input)
    vectorized_input = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized_input)[0]

    print("Predicted Class:", prediction)
