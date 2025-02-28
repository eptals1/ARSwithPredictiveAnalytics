from sklearn.metrics import jaccard_score
from sklearn.feature_extraction.text import CountVectorizer


def calculate_jaccard_similarity(text1, text2):
    vectorizer = CountVectorizer(binary=True)
    X = vectorizer.fit_transform([text1, text2]).toarray()
    return jaccard_score(X[0], X[1])

# def calculate_jaccard_similarity(text1, text2):
#     set1 = set(text1.split())  
#     set2 = set(text2.split())  
#     intersection = set1.intersection(set2)  
#     union = set1.union(set2)  
#     jaccard_score = len(intersection) / len(union) if len(union) > 0 else 0
#     return jaccard_score, intersection  


# if __name__ == "__main__":
#     text1 = "resume1"
#     text2 = "job1"
#     print(calculate_jaccard_similarity(text1, text2))