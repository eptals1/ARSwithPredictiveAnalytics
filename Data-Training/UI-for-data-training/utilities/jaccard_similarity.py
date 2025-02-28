from sklearn.metrics import jaccard_score
from sklearn.feature_extraction.text import CountVectorizer


def calculate_jaccard_similarity(text1, text2):
    vectorizer = CountVectorizer(binary=True)
    X = vectorizer.fit_transform([text1, text2]).toarray()
    return jaccard_score(X[0], X[1])

# if __name__ == "__main__":
#     text1 = "resume1"
#     text2 = "job1"
#     print(calculate_jaccard_similarity(text1, text2))