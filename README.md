# Spam-Detection-using-NLP
## 📌 Overview

This project implements a Spam Detection System using Natural Language Processing (NLP) techniques.
It processes raw text messages, applies NLP transformations, and classifies them into Spam or Ham (Not Spam) with high accuracy.

## 📂 Project Structure
```
├── Spam Detection.ipynb   # Jupyter Notebook with full implementation
├── README.md              # Documentation
└── dataset               # Dataset 
```
## 🔑 Key Features

Text Preprocessing:

Lowercasing

Removing punctuation, numbers, and special characters

Stopword removal

Tokenization and stemming/lemmatization

Feature Engineering:

Bag of Words (BoW)

TF-IDF Vectorization

Modeling:

Trains and evaluates NLP-based classifiers (e.g., Multinomial Naive Bayes).

Performance evaluated using Accuracy, Precision, Recall, and F1-score.

Visualization:

Word clouds for spam vs ham messages

Message length distributions

Spam/ham class imbalance analysis

## 📊 Dataset

Dataset used: SMS Spam Collection Dataset

Contains ~5,574 labeled SMS messages:

ham → Non-spam messages

spam → Spam messages

## ⚙️ Tech Stack

Programming Language: Python

Libraries:

pandas, numpy → Data handling

matplotlib, seaborn, wordcloud → Visualization

scikit-learn → TF-IDF, Train/Test split, Model training

nltk / re → Text preprocessing

## 🚀 How to Run

Clone the repository:
```
git clone https://github.com/your-username/spam-detection-nlp.git
cd spam-detection-nlp
```

Install dependencies:
```
pip install -r requirements.txt

```
Launch Jupyter Notebook:

jupyter notebook "Spam Detection.ipynb"


Run all cells to train and evaluate the NLP spam detector.

## 📈 Results

Achieved ~97–99% accuracy on test data.

Naive Bayes with TF-IDF performed the best.

Word cloud visualizations clearly highlight common spam keywords like “free, win, offer, click”.

## 🔮 Future Work

Integrate deep learning (LSTMs, BiLSTMs, or Transformers like BERT).

Deploy the model as a web app using Flask or Streamlit.

Extend the dataset to cover emails, social media messages, etc.
