import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
print("Current working directory:", os.getcwd())

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")


fake_df["label"] = 0  # 0 for Fake
true_df["label"] = 1  # 1 for Real


df = pd.concat([fake_df, true_df], ignore_index=True)


df = df.sample(frac=1).reset_index(drop=True)


print(df.head())
print(df["label"].value_counts())

df = df.drop(columns=["date", "subject"], errors='ignore')

df["content"] = df["title"] + " " + df["text"]

X = df["content"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

from sklearn.linear_model import PassiveAggressiveClassifier


model = PassiveAggressiveClassifier(max_iter=1000)

model.fit(X_train_tfidf, y_train)


y_pred = model.predict(X_test_tfidf)


acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc * 100:.2f}%")


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

import pickle

# Save the model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save the TF-IDF vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved successfully.")
