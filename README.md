# Misinformation Detection System (Version 1)

## Overview

This project implements a **machine learning pipeline for detecting misinformation in news headlines or textual claims**. The system classifies input text as **Fake News or Real News**, provides a **confidence score**, and retrieves **related real articles for human verification**.

The goal of the project is to demonstrate a **Trust & Safety–oriented moderation assistant**, where machine learning helps identify suspicious claims while still allowing humans to verify evidence.

---

## Features

* Text-based misinformation detection
* Risk scoring based on prediction confidence
* Retrieval of related real news articles for human verification
* Explainable prediction pipeline
* Modular Python project structure suitable for ML workflows

---

## System Architecture

Input Claim
↓
Text Cleaning
↓
TF-IDF Vectorization (Unigrams + Bigrams)
↓
Logistic Regression Classifier
↓
Prediction + Confidence Score
↓
Evidence Retrieval (Cosine Similarity)

---

## Example Output

Input:

```
Scientists confirm aliens built pyramids
```

Output:

```
Prediction Result
------------------
Label: Fake News
Confidence: 92.7 %
Risk Level: HIGH

Related Real Articles For Verification:

- Archaeologists explain how ancient Egyptians built pyramids
- Scientists debunk alien pyramid conspiracy theory
- Researchers dismiss viral pyramid hoax
```

---

## Project Structure

```
misinformation_ml/
│
├── data/
│   ├── Fake.csv
│   └── True.csv
│
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── requirements.txt
└── README.md
```

---

## Technologies Used

* Python
* pandas
* scikit-learn
* TF-IDF text vectorization
* Logistic Regression
* Cosine similarity for article retrieval

Relevant concepts include:

* Natural Language Processing (NLP)
* Supervised Machine Learning
* Text Feature Engineering
* Explainable AI for Trust & Safety systems

---

## Dataset

The model is trained using the **Fake and Real News Dataset** from Kaggle.

The dataset contains labeled news articles classified as **fake or real**, which are used to train the classifier.

---

## Installation

Clone the repository:

```
git clone https://github.com/yourusername/misinformation-detection-ml.git
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Training the Model

Run the training script:

```
python src/train_model.py
```

This will:

* load the dataset
* preprocess text
* train the classifier
* save the trained model and vectorizer

Saved files:

```
models/model.pkl
models/vectorizer.pkl
```

---

## Running Predictions

Run:

```
python src/predict.py
```

Enter a news headline or claim when prompted.

The system will return:

* predicted label
* confidence score
* risk level
* related real articles for verification

---

## Limitations (Version 1)

This version has several limitations:

* The model relies on **TF-IDF keyword matching**, which may not capture semantic meaning.
* Evidence retrieval uses **cosine similarity**, which may return loosely related articles.
* The dataset contains **historical news articles**, which may not reflect modern language patterns.
* The model does not perform **true fact verification**, only classification based on textual patterns.

---

## Future Improvements

Planned improvements include:

* Semantic retrieval using sentence embeddings
* Integration of external knowledge sources such as Wikipedia
* Improved misinformation detection using transformer-based models
* Support for multimodal misinformation (images and videos)

---

## Author

This project was developed as part of a **machine learning portfolio focused on Trust & Safety systems**, demonstrating how ML can assist in identifying and investigating misinformation.
"# misinformation-detection-ml" 
"# Misinformation_ML" 
