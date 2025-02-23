from flask import Flask, render_template, request, jsonify
from flask_wtf.csrf import CSRFProtect
from werkzeug.utils import secure_filename
import os
from utilities.RoBERTa_NER import calculate_resume_similarities    
# from utilities.roBERTa_NER import recognize_entities, calculate_match_score
# from utilities.XGBoost import predict_job_fit

app = Flask(__name__)  
app.config['SECRET_KEY'] = os.urandom(24)  # Required for CSRF
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'Flask-App-Implementation/temps/uploads'
csrf = CSRFProtect(app)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/score-resume', methods=['POST'])
def score_resume():
    try:
        # Check if job description is uploaded
        if 'job_description' not in request.files:
            return jsonify({'success': False, 'message': 'No job description uploaded'}), 400
        
        job_file = request.files['job_description']
        if not job_file or not allowed_file(job_file.filename):
            return jsonify({'success': False, 'message': 'Invalid job description file'}), 400

        # Save job description
        job_filename = secure_filename(job_file.filename)
        job_path = os.path.join(app.config['UPLOAD_FOLDER'], job_filename)
        job_file.save(job_path)

        # Process resumes
        resume_files = request.files.getlist('resumes[]')
        if not resume_files:
            return jsonify({'success': False, 'message': 'No resumes uploaded'}), 400

        resume_paths = []
        for resume in resume_files:
            if resume and allowed_file(resume.filename):
                filename = secure_filename(resume.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                resume.save(filepath)
                resume_paths.append(filepath)

        if not resume_paths:
            return jsonify({'success': False, 'message': 'No valid resume files uploaded'}), 400

        # Calculate similarities using utils.py function
        similarities = calculate_resume_similarities(job_path, resume_paths)

        # Calculate entities using utils.py function
        # for resume_path in resume_paths:
        #     resume_text = extract_text_from_file(resume_path)
        #     entities = recognize_entities_for_each_resume(resume_text)
        #     match_score = calculate_match_score(resume_text, entities)

        # Clean up uploaded files
        try:
            os.remove(job_path)
            # for path in resume_paths:
            #     os.remove(path)

        except Exception as e:
            print(f"Error cleaning up files: {str(e)}")

        # Return analysis results
        analysis_result = {
            'success': True,
            'data': {
                'resumes': similarities,
                'similarity_score': similarities[0]['similarity'] if similarities else 0,  # Best match score
                # 'matching_skills': [match_score['matching_skills']],  # We don't have skill extraction in utils.py yet
                # 'missing_skills': [match_score['missing_skills']],    # We don't have skill extraction in utils.py yet
            }
        }

        return jsonify(analysis_result)

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)