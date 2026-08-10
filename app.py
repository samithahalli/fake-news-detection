import streamlit as st
import joblib
import re
import string


# Load trained model and TF-IDF vectorizer
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")


# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Page configuration
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)


# Title
st.title("📰 Fake News Detection")
st.write(
    "Enter a news article or headline below to check whether "
    "the model predicts it as real or fake."
)


# User input
news_text = st.text_area(
    "Enter news content:",
    height=250,
    placeholder="Paste a news article or headline here..."
)


# Prediction button
if st.button("Check News", type="primary"):

    if not news_text.strip():
        st.warning("Please enter some news content first.")

    else:
        # Clean the input
        cleaned_text = clean_text(news_text)

        # Convert text into TF-IDF features
        text_vector = vectorizer.transform([cleaned_text])

        # Make prediction
        prediction = model.predict(text_vector)[0]

        # Get prediction probability
        probability = model.predict_proba(text_vector)[0]

        if prediction == 1:
            confidence = probability[1] * 100

            st.success("🟢 REAL NEWS")
            st.write(f"Confidence: **{confidence:.2f}%**")

        else:
            confidence = probability[0] * 100

            st.error("🔴 FAKE NEWS")
            st.write(f"Confidence: **{confidence:.2f}%**")


st.divider()

st.caption(
    "Machine Learning Model: TF-IDF + Logistic Regression"
)