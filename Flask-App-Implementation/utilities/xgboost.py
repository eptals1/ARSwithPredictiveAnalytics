from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from utilities.text_extraction import extract_text

def calculate_resume_similarities(job_path, resume_paths):
    # Extract text from job description
    job_text = extract_text(job_path)

    # Extract text from all resumes
    resume_texts = [extract_text(resume) for resume in resume_paths]

    # Remove empty resumes
    valid_resumes = [(path, text) for path, text in zip(resume_paths, resume_texts) if text]
    if not valid_resumes:
        return {"resume_comparisons": []}

    resume_paths, resume_texts = zip(*valid_resumes)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([job_text] + list(resume_texts))

    # Compute cosine similarity between job description and resumes
    job_vec = tfidf_matrix[0]  # First document (job description)
    resume_vecs = tfidf_matrix[1:]  # Remaining (resumes)
    tfidf_similarities = cosine_similarity(job_vec, resume_vecs).flatten()

    # Compute Jaccard similarity
    def jaccard_similarity(text1, text2):
        set1, set2 = set(text1.split()), set(text2.split())
        return len(set1 & set2) / len(set1 | set2)

    jaccard_similarities = np.array([jaccard_similarity(job_text, resume) for resume in resume_texts])

    # Combine TF-IDF and Jaccard scores (weighted)
    combined_scores = 0.6 * tfidf_similarities + 0.4 * jaccard_similarities

    # Sort resumes by combined similarity
    ranked_indices = np.argsort(-combined_scores)
    ranked_resumes = [
        {"resume_path": resume_paths[i], "score": combined_scores[i]}
        for i in ranked_indices
    ]

    return {"resume_comparisons": ranked_resumes}
