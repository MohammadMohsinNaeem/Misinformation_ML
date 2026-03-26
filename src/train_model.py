import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from preprocess import load_dataset


def train():

    print("Loading dataset...")

    df = load_dataset()

    X = df["content"]
    y = df["label"]

    print("Splitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("Vectorizing text...")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.7,
        ngram_range=(1, 2)
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training model...")

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train_vec, y_train)

    print("Evaluating model...")

    predictions = model.predict(X_test_vec)

    print(classification_report(y_test, predictions))

    print("Saving model...")

    os.makedirs("models", exist_ok=True)

    with open("models/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model training complete")


if __name__ == "__main__":
    train()