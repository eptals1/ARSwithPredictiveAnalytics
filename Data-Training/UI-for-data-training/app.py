from flask import Flask, render_template, request, jsonify
from textract import process
from utilities.text_extractor import extract_text_from_docx, extract_text, extract_text_from_pdf, extract_text_with_textract
from utilities.folder_to_csv import process_files

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract_text_from_resumes', methods=['POST'])
def extract_text_from_resumes():
    try:
        resumes = request.files.getlist("resumes")
        if not resumes:
            print("Error: Missing resumes")
            return jsonify({"error":"Missing resumes"}), 400

        # resumes = process_files(resumes)
        for resume_file in resumes:
            resume_text = extract_text(resume_file)
            if not resume_text:
                return jsonify(f"Skipping {resume_file.filename}: Could not extract text")
                # continue  # Skip if text extraction fails

        resume_text = process(resume_folder)


        # Handle the POST request
        data = request.get_json()  # Get the JSON data from the request
        # Process the data as needed
        # Example: data = {'name': 'John', 'age': 30}
        return jsonify({'message': 'pre_processing', 'data': data})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)