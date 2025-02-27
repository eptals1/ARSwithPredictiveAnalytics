#________________________________________________
#                                                |
#                                                |
# Step 1. Initialize the Model and Tokenizer     |
#                                                |
#________________________________________________|

from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import pipeline

# Load pre-trained RoBERTa NER model
model_name = "xlm-roberta-large-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Create NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)


#________________________________________________
#                                                |
#                                                |
#        Step 2. Extract Entities                |
#                                                |
#________________________________________________|

def extract_entities(text):
    ner_results = ner_pipeline(text)
    named_entities = {
        ent['word']: ent['entity_group'] for ent in ner_results if ent['score'] > 0.5
    }
    return named_entities

#________________________________________________
#                                                |
#                                                |
#        Step 3. Compute NER Score               |
#                                                |
#________________________________________________|
def computer_ner_score(job_text, resume_text):
    # Extract entities from job text and resume text
    job_entities = extract_entities(job_text)
    resume_entities = extract_entities(resume_text)

    # Calculate matching entities
    matched_entities = sum(1 for entity in resume_entities if entity in job_entities)
    total_entities = len(job_entities) if job_entities else 1

    # Calculate similarity score as a percentage
    score = (matched_entities / total_entities) * 100 
    return round(score, 2), matched_entities, len(job_entities)

#________________________________________________
#                                                |
#                                                |
#           Step 4. Integrate with               |
#               Resume Similarity Calculation    |
#                                                |
#________________________________________________|

def analyze_resume_with_ner(job_requirement, resumes):
    results = []
    for resume_name, resume_text in resumes.items():
        ner_score, matched, total = computer_ner_score(job_requirement, resume_text)
        results.append({
            "resume_name" : resume_name,
            "ner_score" : ner_score,
            "matched_entities" : matched,
            "total_entities" : total
        })
    return results

#________________________________________________
#                                                |
#                                                |
#           Step 4. Sample Usage                 |
#                                                |
#________________________________________________|

job_description = """We are seeking a Data Scientist proficient in Python, Machine Learning, NLP, and TensorFlow."""
resumes = {
    "John_Doe.pdf": """John is a Data Scientist with experience in Python, Machine Learning, and NLP.""",
    "Jane_Smith.docx": """Jane has expertise in Java, Data Analysis, and Deep Learning."""
}

# Run the NER-based analysis
ner_results = analyze_resume_with_ner(job_description, resumes)

# Display results
for result in ner_results:
    print(f"{result['resume_name']}: {result['ner_score']}% match ({result['matched_entities']}/{result['total_entities']} entities)")
