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

@app.route("/rank-resume-using-tfidf-jaccard", methods=["POST"])
def rank_resume_using_tfidf_jaccard():
    try:
        print("\n📝 Received request!")

        if "job_description" not in request.files or "resumes[]" not in request.files:
            return jsonify({"success": False, "message": "Missing files"}), 400

        job_file = request.files["job_description"]
        resume_files = request.files.getlist("resumes[]")

        print(f"📂 Job File: {job_file.filename}")
        print(f"📂 Received {len(resume_files)} resumes")

        if not job_file or not resume_files:
            return jsonify({"success": False, "message": "Invalid file uploads"}), 400

        # ✅ Save job file temporarily
        job_filename = secure_filename(job_file.filename)
        job_path = os.path.join(UPLOAD_FOLDER, job_filename)
        job_file.save(job_path)

        print(f"✅ Job file saved at: {job_path}")

        # ✅ Save all resume files temporarily
        resume_paths = []
        for resume_file in resume_files:
            resume_filename = secure_filename(resume_file.filename)
            resume_path = os.path.join(UPLOAD_FOLDER, resume_filename)
            resume_file.save(resume_path)
            resume_paths.append(resume_path)

            print(f"✅ Resume saved at: {resume_path}")

        # 🟢 Step 1: Rank Resumes (TF-IDF + Jaccard)
        ranking_results = calculate_resume_similarities(job_path, resume_paths)

        # Extract ranked resumes
        ranked_resumes = ranking_results.get("resume_comparisons", [])

        # 🟢 Step 2: Predict Suitability with XGBoost
        predictions = predict_suitability([r["resume_path"] for r in ranked_resumes], job_path)

        # Merge ranking & prediction
        final_results = [
            {"resume_path": r["resume_path"], "score": r["score"], "prediction": p}
            for r, p in zip(ranked_resumes, predictions)
        ]

        # ✅ Cleanup: Delete files after processing
        if os.path.exists(job_path):
            os.remove(job_path)

        for resume_path in resume_paths:
            if os.path.exists(resume_path):
                os.remove(resume_path)

        return jsonify({"success": True, "data": {"resumes": final_results}})

    except Exception as e:
        print("🔥 ERROR:", str(e))
        import traceback
        print(traceback.format_exc())
        return jsonify({"success": False, "message": "Internal Server Error"}), 500


if __name__ == "__main__":
    app.run(debug=True)
