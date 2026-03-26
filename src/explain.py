import pickle
import numpy as np

def explain():

    # Load vectorizer
    with open("models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    # Load model
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    # Get feature names (words)
    feature_names = vectorizer.get_feature_names_out()

    # Model coefficients
    coefficients = model.coef_[0]

    # Top words indicating FAKE news
    top_fake_indices = np.argsort(coefficients)[:20]

    # Top words indicating REAL news
    top_real_indices = np.argsort(coefficients)[-20:]

    print("\nTop words associated with FAKE news:\n")
    for i in top_fake_indices:
        print(feature_names[i])

    print("\nTop words associated with REAL news:\n")
    for i in reversed(top_real_indices):
        print(feature_names[i])


if __name__ == "__main__":
    explain()