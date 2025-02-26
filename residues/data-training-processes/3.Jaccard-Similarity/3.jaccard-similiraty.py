import os
from PyPDF2 import PdfReader
from typing import List, Dict, Set
from collections import Counter

def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Calculate Jaccard similarity between two sets"""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a single PDF file"""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {str(e)}")
        return ""

def preprocess_text(text: str) -> List[str]:
    """Preprocess text by lowercasing and splitting into words"""
    return text.lower().split()

def calculate_jaccard_similarities(folder_path: str) -> Dict[str, Dict[str, float]]:
    """Calculate Jaccard similarities between all resumes in a folder"""
    resume_texts = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            text = extract_text_from_pdf(file_path)
            resume_texts[filename] = set(preprocess_text(text))

    similarities = {}
    for filename1 in resume_texts:
        similarities[filename1] = {}
        for filename2 in resume_texts:
            if filename1 != filename2:
                similarity = jaccard_similarity(resume_texts[filename1], resume_texts[filename2])
                similarities[filename1][filename2] = similarity

    return similarities

def calculate_resume_job_match(folder_path: str, job_requirement_text: str) -> Dict[str, float]:
    """Calculate Jaccard similarities between resumes and a job requirement"""
    job_req_words = set(preprocess_text(job_requirement_text))
    matches = {}
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            resume_text = extract_text_from_pdf(file_path)
            resume_words = set(preprocess_text(resume_text))
            similarity = jaccard_similarity(resume_words, job_req_words)
            matches[filename] = similarity
    
    return matches

def main():
    folder_path = "dataset/1.raw/linked-in"
    
    # Example job requirement text - replace this with actual job requirement
    job_requirement = """
    Required Skills:
    - Python programming
    - Machine learning
    - Data analysis
    - SQL databases
    - Communication skills
    """
    
    matches = calculate_resume_job_match(folder_path, job_requirement)
    
    print("Resume matches with job requirement:")
    # Sort resumes by match score in descending order
    sorted_matches = dict(sorted(matches.items(), key=lambda x: x[1], reverse=True))
    for resume, score in sorted_matches.items():
        print(f"{resume}: {score:.4f}")

if __name__ == "__main__":
    main()
