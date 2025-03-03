import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from utilities.text_extraction import extract_text
from utilities.pre_processing import preprocess_text
from utilities.jaccard_similarity_scoring import calculate_resume_similarities
from utilities.xgboost import predict_suitability

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the directory exists

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/tfidf-with-jaccard-ranking", methods=["POST"])
def rank_resumes():
    try:
        logging.info("📥 Received request for TF-IDF + Jaccard ranking.")

        # Validate request
        if "job_requirement" not in request.files or "resumes" not in request.files:
            logging.warning("⚠️ Missing job description or resumes in request.")
            return jsonify({"success": False, "message": "Missing files"}), 400

        job_file = request.files["job_description"]
        resume_files = request.files.getlist("resumes[]")

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


if __name__ == "__main__":
    app.run(debug=True)
