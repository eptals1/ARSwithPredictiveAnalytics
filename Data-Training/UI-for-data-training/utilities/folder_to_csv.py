import pandas as pd
import os
from text_extractor import extract_text

def process_files(resume_folder, output_csv):
    """Process resumes, extract text, and save to CSV."""
    data = []

    for filename in os.listdir(resume_folder):
        file_path = os.path.join(resume_folder, filename)
        if os.path.isfile(file_path) and file_path.endswith((".pdf", ".docx", ".jpg", ".jpeg", ".png")):
            print(f"Extracting text from resume: {filename}")
            text = extract_text(file_path)
            if text:
                data.append({"Filename": filename, "Extracted Text": text})

    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Extraction complete. Data saved to {output_csv}")

# Example usage
resume_folder = "C:/Users/Acer/Desktop/Talaba,Ephraim/PECIT"  # Change this to your resume folder path
output_csv = "text-extracted-resumes.csv"  # Change this to your desired output CSV path
process_files(resume_folder, output_csv)
