import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import PyPDF2
import docx

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {str(e)}")
    return text

def extract_text_from_docx(docx_path):
    """Extract text from DOCX file"""
    text = ""
    try:
        doc = docx.Document(docx_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        print(f"Error reading DOCX {docx_path}: {str(e)}")
    return text

def extract_text_from_file(file_path):
    """Extract text from PDF or DOCX file"""
    if file_path.lower().endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.lower().endswith(('.docx', '.doc')):
        return extract_text_from_docx(file_path)
    return ""

def preprocess_text(text):
    """Preprocess text by tokenizing, removing stopwords, and lemmatizing"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens

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
    similarities = []
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
