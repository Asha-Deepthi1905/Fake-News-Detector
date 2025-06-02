import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.title("📰 Fake News Detector")

st.markdown("This app predicts whether a news article is real or fake and explains the key influencing words.")

st.sidebar.header("🔄 Batch Prediction")
uploaded_file = st.sidebar.file_uploader("Upload CSV with 'text' column", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "text" in df.columns:
        st.sidebar.success("File uploaded successfully")
        df["Prediction"] = model.predict(vectorizer.transform(df["text"]))
        df["Prediction"] = df["Prediction"].map({0: "Fake", 1: "Real"})
        st.subheader("Predictions for Uploaded File")
        st.dataframe(df)
    else:
        st.sidebar.error("CSV must contain a 'text' column.")

st.subheader("🔍 Predict Single News Article")
user_input = st.text_area("Enter news content here:")

if st.button("Predict") and user_input:
    input_vector = vectorizer.transform([user_input])
    prediction = model.predict(input_vector)[0]
    label = "Real" if prediction == 1 else "Fake"
    st.success(f"🧠 Prediction: {label}")


    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(input_vector.toarray()).flatten()[::-1]
    top_n = 10
    top_words = feature_array[tfidf_sorting][:top_n]
    st.info(f"Top indicative words: {', '.join(top_words)}")


st.sidebar.markdown("---")
if st.sidebar.checkbox("Show Model Performance on Test Data"):
    
    X_test = pickle.load(open("X_test.pkl", "rb"))
    y_test = pickle.load(open("y_test.pkl", "rb"))
    y_pred = model.predict(X_test)

    st.subheader("📈 Model Performance")
    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.text("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)


with st.expander("🧠 Show Top Predictive Words (from trained model)"):
    feature_names = vectorizer.get_feature_names_out()
    weights = model.coef_[0]

    # Top 20 features pushing toward Fake (label 0)
    top_fake_indices = np.argsort(weights)[:20]
    top_fake_words = [(feature_names[i], weights[i]) for i in top_fake_indices]

    # Top 20 features pushing toward Real (label 1)
    top_real_indices = np.argsort(weights)[-20:]
    top_real_words = [(feature_names[i], weights[i]) for i in reversed(top_real_indices)]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("🔥 **Top words predicting Fake news**")
        for word, weight in top_fake_words:
            st.write(f"{word}: {weight:.4f}")

    with col2:
        st.markdown("✅ **Top words predicting Real news**")
        for word, weight in top_real_words:
            st.write(f"{word}: {weight:.4f}")
st.warning(
    "⚠️ Disclaimer: This Fake News Detector is a demo built on a limited dataset. "
    "Predictions may not be fully accurate, especially for real-world events. "
    "Always verify information from trusted sources."
)

