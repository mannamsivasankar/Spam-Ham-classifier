import os
import re
import html
import string
import json
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# Load models and assets
models_dir = "models"
spam_model = joblib.load(os.path.join(models_dir, "spam_model.pkl"))
spam_vectorizer = joblib.load(os.path.join(models_dir, "spam_vectorizer.pkl"))

tweet_model = joblib.load(os.path.join(models_dir, "tweet_model.pkl"))
tweet_vectorizer = joblib.load(os.path.join(models_dir, "tweet_vectorizer.pkl"))
tweet_selector = joblib.load(os.path.join(models_dir, "tweet_selector.pkl"))

# Load pre-calculated metrics
with open(os.path.join(models_dir, "metrics.json"), "r") as f:
    model_metrics = json.load(f)

# Tweet Preprocessing (must match train_models.py)
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

# SMS Preprocessing
def clean_spam_text(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return text

# Softmax helper for pseudo-probabilities from decision_function
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    return jsonify(model_metrics)

@app.route("/api/classify/spam", methods=["POST"])
def classify_spam():
    data = request.get_json() or {}
    message = data.get("message", "")
    if not message.strip():
        return jsonify({"error": "Empty message"}), 400

    cleaned = clean_spam_text(message)
    vec = spam_vectorizer.transform([cleaned])
    prediction = spam_model.predict(vec)[0]
    
    # Get probability using predict_proba
    probs = spam_model.predict_proba(vec)[0]
    # classes: ham is idx 0, spam is idx 1
    confidence = float(probs[1] if prediction == "spam" else probs[0])

    # Find which words contributed to the prediction
    # By analyzing features present in the text and their log probabilities
    feature_names = spam_vectorizer.get_feature_names_out()
    word_indices = vec.nonzero()[1]
    contributing_words = []
    
    # Class 0: ham, Class 1: spam
    spam_feat_prob = spam_model.feature_log_prob_[1]
    ham_feat_prob = spam_model.feature_log_prob_[0]
    prob_diff = spam_feat_prob - ham_feat_prob

    for idx in word_indices:
        word = feature_names[idx]
        diff = float(prob_diff[idx])
        contributing_words.append({
            "word": word,
            "spam_factor": diff # positive means more spammy, negative means more hammy
        })
        
    # Sort by absolute impact
    contributing_words = sorted(contributing_words, key=lambda x: abs(x["spam_factor"]), reverse=True)

    return jsonify({
        "message": message,
        "class": prediction,
        "confidence": confidence,
        "words_contributed": contributing_words[:10]
    })

@app.route("/api/classify/tweet", methods=["POST"])
def classify_tweet():
    data = request.get_json() or {}
    tweet = data.get("tweet", "")
    if not tweet.strip():
        return jsonify({"error": "Empty tweet text"}), 400

    cleaned = clean_tweet(tweet)
    vec = tweet_vectorizer.transform([cleaned])
    vec_sel = tweet_selector.transform(vec)
    
    # LinearSVC predict
    prediction_idx = int(tweet_model.predict(vec_sel)[0])
    
    # Pseudo probabilities using softmax on decision function
    decision_scores = tweet_model.decision_function(vec_sel)[0]
    probs = softmax(decision_scores)
    confidence = float(probs[prediction_idx])
    
    classes_map = {0: "hate_speech", 1: "offensive", 2: "normal"}
    prediction_label = classes_map[prediction_idx]

    # Find contributing words
    feature_names = tweet_vectorizer.get_feature_names_out()
    selected_indices = tweet_selector.get_support(indices=True)
    selected_feature_names = [feature_names[idx] for idx in selected_indices]
    
    word_indices = vec_sel.nonzero()[1]
    contributing_words = []
    
    for idx in word_indices:
        word = selected_feature_names[idx]
        # Get weight of word for the predicted class
        weight = float(tweet_model.coef_[prediction_idx][idx])
        contributing_words.append({
            "word": word,
            "weight": weight
        })

    # Sort by weight
    contributing_words = sorted(contributing_words, key=lambda x: x["weight"], reverse=True)

    return jsonify({
        "tweet": tweet,
        "class": prediction_label,
        "confidence": confidence,
        "all_confidences": {
            "hate_speech": float(probs[0]),
            "offensive": float(probs[1]),
            "normal": float(probs[2])
        },
        "words_contributed": contributing_words[:10]
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
