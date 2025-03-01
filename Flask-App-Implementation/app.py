from flask import Flask, render_template, request, jsonify
import numpy as np
import xgboost as xgb
from utilities.pre_processing import preprocess_text
from utilities.jaccard_xgboost_ranking import calculate_jaccard_similarity
from utilities.text_extraction import extract_text
import numpy as np
import joblib
from scipy.sparse import hstack


app = Flask(__name__)

#--------------------------
# Homepage
#--------------------------
@app.route('/')
def index():
    return render_template('index.html')

#--------------------------</>
# Resume Scoring
#--------------------------</>
# Load the XGBoost model
classifier_model = xgb.Booster()
classifier_model.load_model("Data-Training/models/xgboost_resume_classifier.json")

# Load the saved TF-IDF vectorizer
vectorizer = joblib.load("Data-Training/models/tfidf_vectorizer.pkl")

@app.route("/score-resumes", methods=["POST"])
def score_resumes():
    try:

        job_file = request.files.get("job_description")
        resumes = request.files.getlist("resumes")

        if not job_file or not resumes:
            return jsonify({"error": "Missing job description or resumes"}), 400

        # Extract & preprocess job description text
        job_text = extract_text(job_file)
        if not job_text:
            return jsonify({"error": "Could not extract text from job description"}), 400
        job_text = preprocess_text(job_text)

        scores = []

        for resume_file in resumes:
            resume_text = extract_text(resume_file)
            if not resume_text:
                jsonify({"error": "Could not extract text from resume"})
                print(f"Skipping {resume_file.filename}: Could not extract text")  # Debugging print
                # continue  # Skip if text extraction fails

            resume_text = preprocess_text(resume_text)

            # Convert job and resume into TF-IDF features
            job_tfidf = vectorizer.transform([job_text])
            resume_tfidf = vectorizer.transform([resume_text])

            # Combine job & resume features (horizontally stacked)
            feature_input = hstack([job_tfidf, resume_tfidf])

            # Convert to DMatrix for XGBoost
            dmatrix = xgb.DMatrix(feature_input)

            # Predict suitability using XGBoost
            prediction = classifier_model.predict(dmatrix)
            classification = "Suitable" if prediction[0] > 0.05 else "Not Suitable"

            print(f"Processed: {resume_file.filename} | XGBoost Prediction: {classification}")  # Debugging print

            scores.append({
                "resume": resume_file.filename,
                "classification": classification,
                "probability": float(prediction[0])  # Convert numpy float to Python float
            })

        if not scores:
            print("Error: No resumes processed")  # Debugging print
            return jsonify({"error": "No resumes processed"}), 400

        return jsonify({"success": True, "scores": scores})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

#------------------------------------------------
# Resume Analyzer
#------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True)
