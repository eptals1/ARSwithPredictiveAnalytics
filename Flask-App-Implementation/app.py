import pickle
import logging
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from utilities.text_extraction import extract_text
from utilities.tfidf_with_jaccard_ranking import rank_resumes


import os
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)

# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit to 16MB
FAILED_FOLDER = 'failed_resumes'
UPLOAD_FOLDER = 'uploads'

# Load pre-trained models and vectorizer
try:
    with open("models/xgboost_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open("models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("models/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    logging.info("Models loaded successfully.")
except Exception as e:
    logging.error(f"Error loading models: {e}")
    raise e

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/tfidf-with-jaccard-ranking", methods=["POST"])
def rank_resumes():
    try:
        logging.info("📥 Received request for TF-IDF + Jaccard ranking.")

        # Validate request
        if "job_requirement" not in request.files or "resumes" not in request.files:
            logging.warning("⚠️ Missing job description or resumes in request.")
            return jsonify({"success": False, "message": "Missing files"}), 400

        job_file = request.files["job_requirement"]
        resume_files = request.files.getlist("resumes")

        if job_file.filename == "":
            logging.warning("⚠️ Job description file is empty.")
            return jsonify({"success": False, "message": "Job description file is empty"}), 400

        if not resume_files:
            logging.warning("⚠️ No resumes uploaded.")
            return jsonify({"success": False, "message": "No resumes provided"}), 400

        # ✅ Extract job text
        job_text = extract_text(job_file)

        # ✅ Extract resume texts
        resume_texts = [extract_text(resume) for resume in resume_files]

        logging.debug(f"📄 Extracted job text: {job_text[:500]}")
        logging.debug(f"📂 Extracted {len(resume_texts)} resumes.")

        # ✅ Save job file temporarily
        job_filename = secure_filename(job_file.filename)
        job_path = os.path.join(UPLOAD_FOLDER, job_filename)
        job_file.save(job_path)
        logging.info(f"📄 Job file saved at: {job_path}")

        # ✅ Save resume files temporarily
        resume_paths = []
        for resume_file in resume_files:
            resume_filename = secure_filename(resume_file.filename)
            resume_path = os.path.join(UPLOAD_FOLDER, resume_filename)
            resume_file.save(resume_path)
            resume_paths.append(resume_path)

            logging.info(f"✅ Resume saved at: {resume_path}")

        # 🟢 Step 1: Rank Resumes (TF-IDF + Jaccard)
        ranking_results = rank_resumes(job_text, resume_texts)

        # Extract ranked resumes
        ranked_resumes = ranking_results.get("resume_comparisons", [])

        # 🟢 Step 2: Predict Suitability with XGBoost
        predictions = predict_suitability(resume_texts, job_text)  # ✅ Use extracted text, not paths

        final_results = []
        for r, p in zip(ranked_resumes, predictions):
            result = {
                "resume_path": r["resume_path"],
                "score": r["score"],
                "prediction": p
            }
            final_results.append(result)

            # 🚨 Move rejected resumes to `failed_resumes` folder
            if p < 0.5:  # Adjust threshold as needed
                failed_path = os.path.join(FAILED_FOLDER, os.path.basename(r["resume_path"]))
                shutil.move(r["resume_path"], failed_path)
                logging.warning(f"🚨 Moved to failed resumes: {failed_path}")

        # ✅ Cleanup: Delete job file after processing
        if os.path.exists(job_path):
            os.remove(job_path)

        for resume_path in resume_paths:
            if os.path.exists(resume_path):
                os.remove(resume_path)

        return jsonify({"success": True, "data": {"resumes": final_results}})

    except Exception as e:
        logging.error(f"🔥 ERROR: {str(e)}")
        return jsonify({"success": False, "message": "Internal Server Error"}), 500

#--------------------------------------
# XGBoost Prediction
#--------------------------------------

# Allowed file types
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/analyse-failed-resume", methods=["POST"])
def analyse_failed_resume():
    FAILED_FOLDER = "failed_resumes"

    if not os.path.exists(FAILED_FOLDER):
        return jsonify({"success": False, "message": "No failed resumes found."}), 400

    files = [f for f in os.listdir(FAILED_FOLDER) if allowed_file(f)]
    results = []

    if not files:
        return jsonify({"success": False, "message": "No valid resume files in failed folder."}), 400

    for file in files:
        file_path = os.path.join(FAILED_FOLDER, file)

        # Extract resume text
        resume_text = extract_text(file_path)

        # Transform text into TF-IDF features
        resume_tfidf = vectorizer.transform([resume_text])

        # Predict job role
        probabilities = model.predict_proba(resume_tfidf)[0]
        top_indices = np.argsort(probabilities)[::-1][:3]  # Get top 3 job roles
        top_roles = label_encoder.inverse_transform(top_indices)
        top_scores = probabilities[top_indices] * 100  # Convert to percentage

        # Generate skills and experience analysis
        analysis = analyze_resume(resume_text)

        # Store result
        results.append({
            "filename": file,
            "top_jobs": [{ "role": top_roles[i], "score": f"{top_scores[i]:.2f}%" } for i in range(len(top_roles))],
            "analysis": analysis
        })

    return jsonify({"success": True, "data": results})



def analyze_resume(text):
    """Basic resume analysis - Extracts key skills & experience."""
    skills = ["Python", "Java", "C++", "Machine Learning", "Data Analysis", "Sales", "Excel", "Project Management"]
    experience_keywords = ["years", "months", "developer", "manager", "engineer", "assistant"]
    
    detected_skills = [skill for skill in skills if skill.lower() in text.lower()]
    experience = [word for word in text.split() if word.lower() in experience_keywords]

    return {
        "skills": detected_skills or "Not detected",
        "experience": " ".join(experience) or "Not detected"
    }

if __name__ == "__main__":
    app.run(debug=True)
