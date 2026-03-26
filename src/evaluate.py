import pickle
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from preprocess import load_dataset


def evaluate():

    df = load_dataset()

    X = df["content"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with open("models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    X_test_vec = vectorizer.transform(X_test)

    predictions = model.predict(X_test_vec)

    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    evaluate()
