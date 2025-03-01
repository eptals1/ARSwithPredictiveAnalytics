import os
from transformers import pipeline
from utilities.pre_processing import extract_text_from_file, preprocess_text

# Load RoBERTa NER model
ner_pipeline = pipeline('ner', model='xlm-roberta-large-finetuned-conll03-english',
                         tokenizer='xlm-roberta-large-finetuned-conll03-english')


# Function to extract named entities
def extract_entities(text):
    """Extract named entities using RoBERTa NER"""
    ner_results = ner_pipeline(text)
    entities = set()  # Use a set to remove duplicates
    
    for entity in ner_results:
        if entity['score'] > 0.7:  # Filter low-confidence entities
            entities.add(entity['word'].strip())
    
    return entities

# Function to calculate Jaccard similarity
def jaccard_similarity(set1, set2):
    """Compute Jaccard similarity"""
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0

# Main function to process resumes
def calculate_resume_similarities(job_path, resume_paths):
    """Compute similarity scores for resumes against job description"""
    job_text = extract_text_from_file(job_path)
    job_entities = extract_entities(job_text)

    results = []

    for resume_path in resume_paths:
        resume_text = extract_text_from_file(resume_path)
        resume_entities = extract_entities(resume_text)
        similarity_score = jaccard_similarity(resume_entities, job_entities)

        # Find intersecting entities (common words between job and resume)
        intersecting_entities = list(resume_entities.intersection(job_entities))

        results.append({
            'filename': os.path.basename(resume_path),
            'similarity': round(similarity_score, 2),
            'resume_entities': list(resume_entities),
            'intersecting_entities': intersecting_entities  # Words in both job & resume
        })

    # Sort results by highest similarity score
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    return {
        'job_entities': list(job_entities),
        'resumes': results
    }
