import os
import torch
from transformers import RobertaTokenizerFast, RobertaForTokenClassification
from PyPDF2 import PdfReader
import re
from typing import List, Dict, Set
from docx import Document

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file"""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {str(e)}")
        return ""

def extract_text_from_docx(docx_path: str) -> str:
    """Extract text from a DOCX file"""
    try:
        doc = Document(docx_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        print(f"Error processing DOCX {docx_path}: {str(e)}")
        return ""

def extract_job_requirements(file_path: str) -> str:
    """Extract text from either PDF or DOCX file"""
    if file_path.lower().endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.lower().endswith('.docx'):
        return extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a PDF or DOCX file.")

def preprocess_text(text: str) -> List[str]:
    """Clean and tokenize text into words"""
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    # Split into words
    return text.strip().split()

def analyze_resume(model, tokenizer, text: str) -> Dict[str, List[str]]:
    """Analyze resume text using RoBERTa model"""
    # Tokenize the text
    tokens = preprocess_text(text)
    inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)

    # Convert predictions to labels
    predicted_labels = []
    word_ids = inputs.word_ids(batch_index=0)  # Fixed: specify batch_index
    
    previous_word_idx = None
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx == previous_word_idx:
            continue
        predicted_labels.append(model.config.id2label[predictions[0][idx].item()])
        previous_word_idx = word_idx

    # Group extracted information by label type
    extracted_info = {
        'GENDER': [],
        'AGE': [],
        'ADDRESS': [],
        'SKILL': [],
        'EDUCATION': [],
        'EXPERIENCE': [],
        'CERTIFICATION': [],
    }

    for token, label in zip(tokens[:len(predicted_labels)], predicted_labels):  # Fixed: ensure lengths match
        if label.startswith('B-'):
            category = label[2:]  # Remove 'B-' prefix
            if category in extracted_info:
                extracted_info[category].append(token)

    return extracted_info

def parse_job_requirements(text: str) -> Dict[str, List[str]]:
    """Parse job requirements using rule-based approach with improved filtering"""
    requirements = {
        'SKILL': [],
        'EDUCATION': [],
        'EXPERIENCE': [],
        'CERTIFICATION': []
    }
    
    # Convert to lowercase and split into lines
    lines = text.lower().split('\n')
    
    # Common requirement indicators
    skill_indicators = ['skills', 'proficient in', 'knowledge of', 'experience with', 'ability to', 'competencies']
    edu_indicators = ['education', 'degree', 'graduate', 'bachelor', 'master', 'academic']
    exp_indicators = ['experience', 'years', 'worked', 'work history']
    cert_indicators = ['certification', 'certificate', 'certified', 'license', 'qualified']
    
    # Words to exclude (common words, locations, etc.)
    exclude_words = {
        'the', 'and', 'for', 'with', 'has', 'must', 'will', 'can', 'should', 'city', 'region',
        'pasay', 'manila', 'philippines', 'metro', 'office', 'location', 'address',
        'required', 'preferred', 'minimum', 'maximum', 'least', 'more', 'than',
        'year', 'years', 'month', 'months', 'day', 'days'
    }
    
    current_category = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Determine category based on indicators
        if any(ind in line for ind in skill_indicators):
            current_category = 'SKILL'
        elif any(ind in line for ind in edu_indicators):
            current_category = 'EDUCATION'
        elif any(ind in line for ind in exp_indicators):
            current_category = 'EXPERIENCE'
        elif any(ind in line for ind in cert_indicators):
            current_category = 'CERTIFICATION'
            
        # Extract requirements if we're in a category
        if current_category:
            # Remove common bullet points and requirement phrases
            cleaned_line = re.sub(r'^[-•*]\s*', '', line)
            cleaned_line = re.sub(r'required|must have|should have|preferred', '', cleaned_line)
            # Split into words and clean up
            words = [word.strip('.,()') for word in cleaned_line.split()]
            # Filter out excluded words and short words
            words = [word for word in words if len(word) > 2 and word not in exclude_words]
            if words:
                requirements[current_category].extend(words)
    
    # Remove duplicates
    for category in requirements:
        requirements[category] = list(set(requirements[category]))
    
    return requirements

def calculate_match_score(resume_info: Dict[str, List[str]], job_requirements: Dict[str, List[str]]) -> Dict[str, float]:
    """Calculate match score between resume and job requirements"""
    scores = {}
    
    for category in job_requirements:
        if category not in resume_info:
            scores[category] = 0.0
            continue
            
        req_set = set(job_requirements[category])
        resume_set = set(resume_info[category])
        
        if not req_set:
            scores[category] = 1.0
        else:
            intersection = len(req_set.intersection(resume_set))
            scores[category] = intersection / len(req_set)
    
    # Calculate overall score (average of category scores)
    overall_score = sum(scores.values()) / len(scores)
    scores['overall'] = overall_score
    
    return scores

def main():
    # Load the trained model and tokenizer
    model_path = "output/models/roberta-resume-ner/checkpoint-40"
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
    model = RobertaForTokenClassification.from_pretrained(model_path)
    model.eval()

    # Get job requirements from file
    job_req_file = "dataset/job_requirements/admin assistant.docx"
    print("\nExtracting job requirements...")
    job_req_text = extract_job_requirements(job_req_file)
    
    # Parse job requirements using specialized parser
    print("Analyzing job requirements...")
    job_requirements = parse_job_requirements(job_req_text)
    
    print("\nJob Requirements Found:")
    for category, items in job_requirements.items():
        print(f"- {category}: {', '.join(items)}")

    # Process all resumes in the folder
    resume_folder = "dataset/1.raw/linked-in"
    results = {}

    for filename in os.listdir(resume_folder):
        if filename.lower().endswith('.pdf'):
            print(f"\nProcessing {filename}...")
            file_path = os.path.join(resume_folder, filename)
            
            # Extract and analyze text
            text = extract_text_from_pdf(file_path)
            resume_info = analyze_resume(model, tokenizer, text)
            
            # Calculate match scores
            scores = calculate_match_score(resume_info, job_requirements)
            results[filename] = {
                'scores': scores,
                'extracted_info': resume_info
            }

    # Display results sorted by overall score
    print("\nResults sorted by overall match score:")
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]['scores']['overall'],
        reverse=True
    )

    for filename, result in sorted_results:
        print(f"\n{filename}")
        print(f"Overall Score: {result['scores']['overall']:.2f}")
        print("Category Scores:")
        for category, score in result['scores'].items():
            if category != 'overall':
                print(f"- {category}: {score:.2f}")
        print("\nExtracted Information:")
        for category, items in result['extracted_info'].items():
            print(f"- {category}: {', '.join(items)}")

if __name__ == "__main__":
    main()
