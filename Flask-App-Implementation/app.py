import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from utilities.text_extraction import extract_text
from utilities.pre_processing import preprocess_text
from utilities.jaccard_similarity_scoring import calculate_resume_similarities

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the directory exists

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/score-resume-using-ner", methods=["POST"])
def score_resume_using_ner():
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

        # 🟢 Call function using file paths (NOT preprocessed text)
        results_dict = calculate_resume_similarities(job_path, resume_paths)

        # Extract comparisons from results
        results = results_dict.get("resume_comparisons", [])

        # ✅ Cleanup: Delete files after processing
        if os.path.exists(job_path):
            os.remove(job_path)

        for resume_path in resume_paths:
            if os.path.exists(resume_path):
                os.remove(resume_path)

        return jsonify({"success": True, "data": {"resumes": results}})

    except Exception as e:
        print("🔥 ERROR:", str(e))
        import traceback
        print(traceback.format_exc())
        return jsonify({"success": False, "message": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(debug=True)
