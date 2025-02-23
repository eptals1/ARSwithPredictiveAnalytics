import os
from utilities.pre_processing import extract_text_from_file, preprocess_text
from transformers import pipeline

# Initialize the NER pipeline
ner_pipeline = pipeline('ner', model='xlm-roberta-large-finetuned-conll03-english',
 tokenizer='xlm-roberta-large-finetuned-conll03-english')

# Extract Entities
def extract_entities(text):
    ner_results = ner_pipeline(text)
    entities = set()  # Use set to avoid duplicates
    
    for entity in ner_results:
        if entity['score'] > 0.7:  # Filter low-confidence entities
            entities.add(entity['word'].strip())
    
    return entities

def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets"""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def calculate_resume_similarities(job_path, resume_paths):
    """Calculate Jaccard similarity between job requirements and multiple resumes"""
    # Extract and preprocess job requirements
    job_text = extract_text_from_file(job_path)
    job_tokens = set(preprocess_text(job_text))
    
    # Calculate similarities for each resume
    for resume_path in resume_paths:
        resume_text = extract_text_from_file(resume_path)
        resume_tokens = set(preprocess_text(resume_text))
        similarity = jaccard_similarity(resume_tokens, job_tokens)
        
        similarities.append({
            'resume_name': os.path.basename(resume_path),
            'similarity': round(similarity * 100, 2)  # Convert to percentage
        })
    
    # Sort by similarity score in descending order
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    return similarities
