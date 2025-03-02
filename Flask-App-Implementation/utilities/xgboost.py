import pickle
import numpy as np
from utilities.text_extraction import extract_text
from utilities.jaccard_similarity_scoring import calculate_resume_similarities
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Load pre-trained models
with open("models/xgboost_model.pkl", "rb") as file:
    xgb_model = pickle.load(file)

with open("models/vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

with open("models/label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)

def predict_suitability(resume_paths, job_path):
    """Predicts suitability of resumes using the trained XGBoost model."""
    
    # Extract text from job description
    job_text = extract_text(job_path)
    
    # Extract text from resumes
    resume_texts = [extract_text(resume) for resume in resume_paths]
    
    # Ensure all extracted texts are valid
    valid_resumes = [(path, text) for path, text in zip(resume_paths, resume_texts) if text]
    if not valid_resumes:
        return ["Unknown"] * len(resume_paths)  # Return 'Unknown' if no valid resumes

    resume_paths, resume_texts = zip(*valid_resumes)

    # TF-IDF Transformation (Use Pre-trained Vectorizer)
    job_vector = vectorizer.transform([job_text])
    resume_vectors = vectorizer.transform(resume_texts)

    # Compute Jaccard Similarity
    jaccard_similarities = np.array([
        calculate_resume_similarities(job_path, [resume])["resume_comparisons"][0]["score"]
        for resume in resume_paths
    ]).reshape(-1, 1)  # Convert to column format

    # Combine TF-IDF & Jaccard Scores
    X_features = np.hstack((resume_vectors.toarray(), jaccard_similarities))

    # Predict using XGBoost
    predictions = xgb_model.predict(X_features)

    # Decode Labels from LabelEncoder
    decoded_predictions = label_encoder.inverse_transform(predictions)

    return decoded_predictions.tolist()  # Return predictions as a list
