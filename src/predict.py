import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def predict(text):

    # Load vectorizer
    with open("models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    # Load trained model
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    # Load dataset for evidence retrieval
    fake = pd.read_csv("data/Fake.csv")
    real = pd.read_csv("data/True.csv")

    real["label"] = 1
    fake["label"] = 0

    dataset = pd.concat([fake, real])

    # Prepare text field
    dataset["content"] = dataset["title"] + " " + dataset["text"]

    # Vectorize dataset
    dataset_vectors = vectorizer.transform(dataset["content"])

    # Vectorize input text
    input_vector = vectorizer.transform([text])

    # Prediction
    prediction = model.predict(input_vector)
    probability = model.predict_proba(input_vector)

    fake_score = probability[0][0]
    real_score = probability[0][1]

    print("\nPrediction Result")
    print("------------------")

    if prediction[0] == 0:
        label = "Fake News"
        confidence = fake_score
    else:
        label = "Real News"
        confidence = real_score

    print("Label:", label)
    print("Confidence:", round(confidence * 100, 2), "%")

    # Risk scoring
    if confidence > 0.85:
        risk = "HIGH"
    elif confidence > 0.65:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    print("Risk Level:", risk)

    # Retrieve related articles
    print("\nRelated Real Articles For Verification:\n")

    similarities = cosine_similarity(input_vector, dataset_vectors)

    top_indices = similarities.argsort()[0][-10:]

    shown = 0

    for i in reversed(top_indices):

        if dataset.iloc[i]["label"] == 1:  # show real news only

            print("-", dataset.iloc[i]["title"])
            shown += 1

        if shown == 5:
            break


if __name__ == "__main__":

    user_input = input("\nEnter a news headline or claim:\n")

    predict(user_input)