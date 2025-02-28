from flask import Flask, render_template, request, jsonify
import numpy as np
import xgboost as xgb
from utilities.pre_processing import preprocess_text
from utilities.jaccard_xgboost_ranking import calculate_jaccard_similarity
from utilities.text_extraction import extract_text
import numpy as np

app = Flask(__name__)

# Load XGBoost models
ranking_model = xgb.Booster()
ranking_model.load_model("C:/Users/Acer/Desktop/Talaba,Ephraim/ARSwithPredictiveAnalytics/Data-Training/xgboost_ranking_model.json")

classifier_model = xgb.Booster()
classifier_model.load_model("C:/Users/Acer/Desktop/Talaba,Ephraim/ARSwithPredictiveAnalytics/Data-Training/xgboost_resume_classifier.json")  # Load the JSON model

# Sample job suggestions for rejected resumes
job_suggestions = [
    "Software Engineer", "Data Analyst", "IT Support", "System Administrator"
]   

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/score-resumes", methods=["POST"])
def score_resumes():
    job_file = request.files.get("job_description")
    resumes = request.files.getlist("resumes")

    if not job_file or not resumes:
        print("Error: Missing job description or resumes")  # Debugging print
        return jsonify({"error": "Missing job description or resumes"}), 400

    job_text = extract_text(job_file)
    if not job_text:
        print("Error: Could not extract text from job description")  # Debugging print
        return jsonify({"error": "Could not extract text from job description"}), 400

    job_text = preprocess_text(job_text)
    scores = []

    for resume_file in resumes:
        resume_text = extract_text(resume_file)
        if not resume_text:
            print(f"Skipping {resume_file.filename}: Could not extract text")
            continue  # Skip if text extraction fails

        resume_text = preprocess_text(resume_text)
        jaccard_score_value, common_words = calculate_jaccard_similarity(job_text, resume_text)
        # rank_score = np.random.uniform(0.1, 0.9)  # Placeholder for actual rank score
        # final_score = (jaccard_score_value + rank_score) / 2  # Averaging both scores
        final_score = jaccard_score_value
        classification = "Suitable" if jaccard_score_value > 0.50 else "Not Suitable"

        print(f"Processed: {resume_file.filename} | Score: {final_score}")  # Debugging print

        scores.append({
            "resume": resume_file.filename,
            "score": final_score,
            "jaccard_score": jaccard_score_value,
            "common_words": list(common_words),
            "classification": classification
        })

    if not scores:
        print("Error: No resumes processed")  # Debugging print
        return jsonify({"error": "No resumes processed"}), 400

    scores.sort(key=lambda x: x["score"], reverse=True)
    return jsonify({"success": True, "scores": scores})




# @app.route('/analyze-rejected', methods=['POST'])
# def analyze_rejected():
#     resumes = request.form.getlist("rejected_resumes")
#     analysis = []
    
#     for resume in resumes:
#         suggestion = np.random.choice(job_suggestions)
#         analysis.append({"resume": resume, "suggested_job": suggestion})
    
#     return jsonify(analysis)

if __name__ == '__main__':
    app.run(debug=True)
