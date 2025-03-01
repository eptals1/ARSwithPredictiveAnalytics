from flask import Flask, request, jsonify, render_template
import pickle
import fitz  # PyMuPDF for PDFs
import textract
import os
from docx import Document
import numpy as np
import math

app = Flask(__name__, template_folder="templates")

# Load trained models
with open('C:/Users/Acer/Desktop/Talaba,Ephraim/ARSwithPredictiveAnalytics/Data-Training/xgb_model.pkl', 'rb') as model_file:
    xgb_model = pickle.load(model_file)
with open('C:/Users/Acer/Desktop/Talaba,Ephraim/ARSwithPredictiveAnalytics/Data-Training/tfidf.pkl', 'rb') as tfidf_file:
    tfidf = pickle.load(tfidf_file)
with open('C:/Users/Acer/Desktop/Talaba,Ephraim/ARSwithPredictiveAnalytics/Data-Training/label_encoder.pkl', 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Preprocessing function
def preprocess_text(text):
    import re
    from nltk.corpus import stopwords
    from num2words import num2words
    import nltk
    nltk.download('stopwords')

    text = text.lower()
    text = re.sub(r'\d+', lambda x: num2words(x.group()), text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# Function to extract text from files
def extract_text(file):
    filename = file.filename.lower()
    file_path = os.path.join("uploads", filename)
    file.save(file_path)
    
    if filename.endswith(".pdf"):
        text = ""
        with fitz.open(file_path) as pdf:
            for page in pdf:
                text += page.get_text("text")
        return text

    elif filename.endswith(".docx"):
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    
    elif filename.endswith(".doc"):
        return textract.process(file_path).decode("utf-8")
    
    elif filename.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    
    return ""

# Serve HTML form
@app.route('/')
def home():
    return render_template("index.html")

# API Route for multiple resume predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'resumes' not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist("resumes")
    predictions = []

    for file in files:
        text = extract_text(file)
        if not text:
            predictions.append({"filename": file.filename, "prediction": "Error: Could not extract text"})
            continue

        processed_text = preprocess_text(text)
        features = tfidf.transform([processed_text]).toarray()
        predicted_label = xgb_model.predict(features)
        predicted_role = label_encoder.inverse_transform(predicted_label)[0]
        predictions.append({"filename": file.filename, "prediction": predicted_role})
        for prediction in predictions:
            if isinstance(prediction["prediction"], float) and math.isnan(prediction["prediction"]):
                prediction["prediction"] = None


    return jsonify({"predictions": predictions})

if __name__ == '__main__':
    app.run(debug=True)
