import os
import re
import html
import string
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def verify():
    print("--- Model Verification Script ---")

    # Load models
    spam_model = joblib.load("models/spam_model.pkl")
    spam_vec = joblib.load("models/spam_vectorizer.pkl")
    tweet_model = joblib.load("models/tweet_model.pkl")
    tweet_vec = joblib.load("models/tweet_vectorizer.pkl")
    tweet_sel = joblib.load("models/tweet_selector.pkl")

    # 1. Verify Spam Classifier
    print("Verifying Spam Classifier...")
    df_spam = pd.read_csv("data/sms.tsv", sep='\t', header=None, names=['label', 'message']).dropna()
    df_spam['message_clean'] = df_spam['message'].apply(lambda x: x.lower().translate(str.maketrans('', '', string.punctuation)))
    
    _, X_test_s, _, y_test_s = train_test_split(
        df_spam['message_clean'], df_spam['label'], test_size=0.2, random_state=42, stratify=df_spam['label']
    )
    
    X_test_vec_s = spam_vec.transform(X_test_s)
    preds_s = spam_model.predict(X_test_vec_s)
    acc_s = accuracy_score(y_test_s, preds_s)
    print(f"Spam Classifier Test Accuracy: {acc_s * 100:.2f}%")
    assert acc_s >= 0.95, f"Spam accuracy {acc_s:.4f} is below 95% target!"

    # 2. Verify Tweet Classifier
    print("Verifying Tweet Classifier...")
    df_tweets = pd.read_csv("data/labeled_data.csv").dropna(subset=['tweet'])
    
    lemmatizer = WordNetLemmatizer()
    def clean_tweet(text):
        text = html.unescape(text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'^RT\s+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.lower().strip()
        tokens = word_tokenize(text)
        cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if len(word) > 1]
        return " ".join(cleaned_tokens)

    df_tweets['tweet_clean'] = df_tweets['tweet'].apply(clean_tweet)
    
    _, X_test_t, _, y_test_t = train_test_split(
        df_tweets['tweet_clean'], df_tweets['class'], test_size=0.05, random_state=28, stratify=df_tweets['class']
    )
    
    X_test_vec_t = tweet_vec.transform(X_test_t)
    X_test_sel_t = tweet_sel.transform(X_test_vec_t)
    preds_t = tweet_model.predict(X_test_sel_t)
    acc_t = accuracy_score(y_test_t, preds_t)
    print(f"Tweet Classifier Test Accuracy: {acc_t * 100:.2f}%")
    assert acc_t >= 0.92, f"Tweet accuracy {acc_t:.4f} is below 92% target!"

    print("\n[SUCCESS] Both model accuracies meet their respective targets!")

if __name__ == "__main__":
    verify()
