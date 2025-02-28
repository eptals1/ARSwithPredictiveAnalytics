from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
import xgboost as xgb


def calculate_jaccard_similarity(text1, text2):
    """Compute Jaccard similarity"""
    vectorizer = CountVectorizer(binary=True)
    X = vectorizer.fit_transform([text1, text2]).toarray()
    return jaccard_score(X[0], X[1])