import pandas as pd
import re


def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)     # remove links
    text = re.sub(r"[^a-zA-Z ]", "", text)  # remove punctuation/numbers
    text = re.sub(r"\s+", " ", text)

    return text


def load_dataset():

    fake = pd.read_csv("data/Fake.csv")
    real = pd.read_csv("data/True.csv")

    fake["label"] = 0
    real["label"] = 1

    df = pd.concat([fake, real], axis=0)

    df["content"] = df["title"] + " " + df["text"]

    df["content"] = df["content"].apply(clean_text)

    df = df[["title", "content", "label"]]

    return df