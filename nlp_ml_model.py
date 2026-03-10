"""
Optimized NLP topic prediction with confidence-aware classification.
Uses TF-IDF + LinearSVC as primary classifier, with a cosine-similarity
fallback when the classifier's decision score is low.
"""

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Load models (at module level – cached by Streamlit via st.cache_resource
# wrapper in app.py; standalone scripts just import normally)
# ---------------------------------------------------------------------------
_model = joblib.load("models/text_model.pkl")
_vectorizer = joblib.load("models/vectorizer.pkl")
_training_data = joblib.load("models/training_data.pkl")

# Pre-compute TF-IDF matrix for training texts (used in fallback)
_train_tfidf = _vectorizer.transform(_training_data["text"])


def predict_topic(text: str) -> str:
    """Predict the learning topic for a user query.

    Strategy:
        1. Classify using LinearSVC and check its decision-function confidence.
        2. If confidence is high (margin ≥ 0.3), return the SVC prediction.
        3. Otherwise, fall back to cosine-similarity against all training
           examples and return the category of the closest match.
    """
    if not text or not text.strip():
        return "Machine Learning"  # safe default

    text = text.strip().lower()
    text_vec = _vectorizer.transform([text])

    # ── Primary: SVC prediction ──
    prediction = _model.predict(text_vec)[0]

    # ── Confidence check via decision function ──
    try:
        decision = _model.decision_function(text_vec)
        # For multi-class, decision is a 2-D array; get max margin
        confidence = float(np.max(decision))
    except AttributeError:
        confidence = 1.0  # model has no decision_function → trust it

    if confidence >= 0.3:
        return prediction

    # ── Fallback: cosine similarity ──
    similarities = cosine_similarity(text_vec, _train_tfidf).flatten()
    best_idx = int(np.argmax(similarities))
    return _training_data["label"].iloc[best_idx]
