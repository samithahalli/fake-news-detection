import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib


# 1. Load datasets
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")


# 2. Add labels
fake["label"] = 0
true["label"] = 1


# 3. Combine datasets
data = pd.concat([fake, true], ignore_index=True)


# 4. Combine title and article text
data["content"] = data["title"].fillna("") + " " + data["text"].fillna("")


# 5. Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


data["content"] = data["content"].apply(clean_text)


# 6. Features and target
X = data["content"]
y = data["label"]


# 7. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 8. Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer(
    max_features=50000,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# 9. Train Logistic Regression model
model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)


# 10. Evaluate model
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Training Completed!")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 11. Save model and vectorizer
joblib.dump(model, "model/fake_news_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("\nModel saved to: model/fake_news_model.pkl")
print("Vectorizer saved to: model/tfidf_vectorizer.pkl")