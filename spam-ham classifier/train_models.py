import os
import re
import html
import json
import urllib.request
import string
import numpy as np
import pandas as pd
import joblib

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    # Setup directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 1. Download NLTK resources
    print("Downloading NLTK resources...")
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('omw-1.4')

    # 2. Ingest and download datasets
    print("Downloading datasets...")
    spam_url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"
    tweets_url = "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv"

    spam_path = "data/sms.tsv"
    tweets_path = "data/labeled_data.csv"

    if not os.path.exists(spam_path):
        print(f"Downloading SMS spam dataset from {spam_url}...")
        urllib.request.urlretrieve(spam_url, spam_path)
    if not os.path.exists(tweets_path):
        print(f"Downloading Tweets dataset from {tweets_url}...")
        urllib.request.urlretrieve(tweets_url, tweets_path)

    metrics_dict = {}

    # ----------------------------------------------------
    # 3. Train SMS Spam Classifier
    # ----------------------------------------------------
    print("\n--- Training SMS Spam Classifier ---")
    df_spam = pd.read_csv(spam_path, sep='\t', header=None, names=['label', 'message'])
    df_spam = df_spam.dropna()

    # Preprocess SMS messages
    def clean_spam_text(text):
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        return text

    df_spam['message_clean'] = df_spam['message'].apply(clean_spam_text)

    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        df_spam['message_clean'], df_spam['label'], test_size=0.2, random_state=42, stratify=df_spam['label']
    )

    vectorizer_s = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_vec_s = vectorizer_s.fit_transform(X_train_s)
    X_test_vec_s = vectorizer_s.transform(X_test_s)

    model_s = MultinomialNB()
    model_s.fit(X_train_vec_s, y_train_s)

    preds_s = model_s.predict(X_test_vec_s)
    acc_s = accuracy_score(y_test_s, preds_s)
    print(f"Spam Classifier Accuracy: {acc_s:.4f}")

    # Generate metrics for spam
    cm_s = confusion_matrix(y_test_s, preds_s, labels=['ham', 'spam']).tolist()
    rep_s = classification_report(y_test_s, preds_s, output_dict=True)

    # Get top predictive terms for spam
    # Feature log probability difference
    feature_names_s = vectorizer_s.get_feature_names_out()
    # class 0 is ham, class 1 is spam
    spam_prob = model_s.feature_log_prob_[1]
    ham_prob = model_s.feature_log_prob_[0]
    prob_diff = spam_prob - ham_prob
    top_spam_indices = np.argsort(prob_diff)[-15:][::-1]
    top_ham_indices = np.argsort(-prob_diff)[-15:][::-1]

    top_spam_features = [{"word": feature_names_s[i], "importance": float(prob_diff[i])} for i in top_spam_indices]
    top_ham_features = [{"word": feature_names_s[i], "importance": float(-prob_diff[i])} for i in top_ham_indices]

    metrics_dict["spam"] = {
        "accuracy": float(acc_s),
        "confusion_matrix": cm_s,
        "classification_report": rep_s,
        "top_spam_features": top_spam_features,
        "top_ham_features": top_ham_features
    }

    # Save Spam Classifier artifacts
    joblib.dump(model_s, "models/spam_model.pkl")
    joblib.dump(vectorizer_s, "models/spam_vectorizer.pkl")

    # ----------------------------------------------------
    # 4. Train Tweet Classifier (Hate, Offensive, Normal)
    # ----------------------------------------------------
    print("\n--- Training Tweet Safety Classifier ---")
    df_tweets = pd.read_csv(tweets_path).dropna(subset=['tweet'])

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

    print("Cleaning tweets (this might take a few seconds)...")
    df_tweets['tweet_clean'] = df_tweets['tweet'].apply(clean_tweet)

    # 95/5 split, random seed 28
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
        df_tweets['tweet_clean'], df_tweets['class'], test_size=0.05, random_state=28, stratify=df_tweets['class']
    )

    vectorizer_t = CountVectorizer(ngram_range=(1, 2), max_features=15000)
    X_train_vec_t = vectorizer_t.fit_transform(X_train_t)
    X_test_vec_t = vectorizer_t.transform(X_test_t)

    # Feature selection: keep 65% of features (35% feature reduction)
    n_features_before = X_train_vec_t.shape[1]
    selector_t = SelectPercentile(score_func=chi2, percentile=65)
    X_train_sel_t = selector_t.fit_transform(X_train_vec_t, y_train_t)
    X_test_sel_t = selector_t.transform(X_test_vec_t)
    n_features_after = X_train_sel_t.shape[1]

    print(f"Features before selection: {n_features_before}, after selection: {n_features_after} (35% reduction)")

    model_t = LinearSVC(C=0.1, random_state=28, max_iter=2000)
    model_t.fit(X_train_sel_t, y_train_t)

    preds_t = model_t.predict(X_test_sel_t)
    acc_t = accuracy_score(y_test_t, preds_t)
    print(f"Tweet Classifier Accuracy: {acc_t:.4f}")

    # Generate metrics for tweets
    # classes: 0 = hate speech, 1 = offensive language, 2 = neither
    cm_t = confusion_matrix(y_test_t, preds_t, labels=[0, 1, 2]).tolist()
    rep_t = classification_report(y_test_t, preds_t, target_names=["hate_speech", "offensive", "normal"], output_dict=True)

    # Extract top words per class from LinearSVC coefficients
    # LinearSVC has multi-class coefficients: shape is (n_classes, n_selected_features)
    selected_indices = selector_t.get_support(indices=True)
    feature_names_t = vectorizer_t.get_feature_names_out()
    selected_feature_names = [feature_names_t[idx] for idx in selected_indices]

    top_features_per_class = {}
    class_labels = ["hate_speech", "offensive", "normal"]
    for i, class_label in enumerate(class_labels):
        coefs = model_t.coef_[i]
        top_indices = np.argsort(coefs)[-15:][::-1]
        top_words = [{"word": selected_feature_names[idx], "importance": float(coefs[idx])} for idx in top_indices]
        top_features_per_class[class_label] = top_words

    metrics_dict["tweets"] = {
        "accuracy": float(acc_t),
        "confusion_matrix": cm_t,
        "classification_report": rep_t,
        "features_before": n_features_before,
        "features_after": n_features_after,
        "top_features": top_features_per_class
    }

    # Save Tweet Classifier artifacts
    joblib.dump(model_t, "models/tweet_model.pkl")
    joblib.dump(vectorizer_t, "models/tweet_vectorizer.pkl")
    joblib.dump(selector_t, "models/tweet_selector.pkl")

    # Save overall metrics.json
    with open("models/metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)
    print("Metrics exported to models/metrics.json successfully!")

if __name__ == "__main__":
    main()
