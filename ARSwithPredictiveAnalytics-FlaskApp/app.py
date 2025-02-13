from flask import Flask, render_template, request, jsonify
from flask_wtf.csrf import CSRFProtect
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)  
app.config['SECRET_KEY'] = os.urandom(24)  # Required for CSRF
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
csrf = CSRFProtect(app)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/resume-matcher/analyze', methods=['POST'])
def analyze_resumes():
    try:
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

        # TODO: Implement your resume analysis logic here
        # For now, returning dummy data
        analysis_result = {
            'success': True,
            'data': {
                'similarity_score': 85,
                'matching_skills': ['Python', 'Flask', 'Web Development'],
                'missing_skills': ['Docker', 'AWS']
            }
        }

        return jsonify(analysis_result)

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)