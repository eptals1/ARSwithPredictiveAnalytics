import os
import xgboost as xgb
import pandas as pd
from utilities.pre_processing import extract_text_from_file, preprocess_text

def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets"""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def calculate_resume_similarities(job_path, resume_paths):
    """Calculate Jaccard similarity between job requirements and multiple resumes"""
    job_text = extract_text_from_file(job_path)
    job_tokens = set(preprocess_text(job_text))
    
    resume_data = []
    
    for resume_path in resume_paths:
        resume_text = extract_text_from_file(resume_path)
        resume_tokens = set(preprocess_text(resume_text))
        similarity = jaccard_similarity(resume_tokens, job_tokens)

        resume_data.append({
            'resume_name': os.path.basename(resume_path),
            'jaccard_score': similarity  # Keep as float (0-1) for XGBoost input
        })
    
    return pd.DataFrame(resume_data)

def rank_resumes_xgboost(job_path, resume_paths, model_path="xgboost_ranking_model.json"):
    """
    Rank resumes based on Jaccard similarity using a trained XGBoost ranking model.
    """
    # Step 1: Compute Jaccard similarity
    df = calculate_resume_similarities(job_path, resume_paths)
    
    # Step 2: Load trained XGBoost model
    model = xgb.Booster()
    model.load_model(model_path)
    
    # Step 3: Prepare data for prediction
    dmatrix = xgb.DMatrix(df[['jaccard_score']])  # Use Jaccard scores as input
    
    # Step 4: Predict ranking scores
    df["ranking_score"] = model.predict(dmatrix)
    
    # Step 5: Sort by ranking score (higher is better)
    df = df.sort_values(by="ranking_score", ascending=False)
    
    return df[["resume_name", "jaccard_score", "ranking_score"]].reset_index(drop=True)

