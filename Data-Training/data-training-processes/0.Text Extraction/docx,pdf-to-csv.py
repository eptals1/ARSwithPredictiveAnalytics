import os
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from typing import List, Dict

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a single PDF file
    """
    try:
        # Create PDF reader object
        reader = PdfReader(pdf_path)
        
        # Extract text from all pages
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "page break"
        
        # Clean up extra whitespace
        #text = " ".join(text.split())
        return text
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {str(e)}")
        return ""

def extract_text_from_docx(docx_path: str) -> str:
    """
    Extract text from a single DOCX file
    """
    try:
        # Load the document
        doc = Document(docx_path)
        
        # Extract text from paragraphs
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + " "
        
        # Clean up extra whitespace
        #text = " ".join(text.split())
        return text
    except Exception as e:
        print(f"Error processing DOCX {docx_path}: {str(e)}")
        return ""

def process_document_folder(folder_path: str, output_csv: str) -> None:
    """
    Process all PDFs and DOCX files in a folder and save their text content to a CSV file
    
    Args:
        folder_path: Path to folder containing PDF and DOCX files
        output_csv: Path where the output CSV file should be saved
    """
    # Store results
    results: List[Dict] = []
    
    # Process each file in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        text_content = ""
        
        # Process based on file extension
        if filename.lower().endswith('.pdf'):
            text_content = extract_text_from_pdf(file_path)
        elif filename.lower().endswith(('.docx', '.doc')):
            text_content = extract_text_from_docx(file_path)
        else:
            continue
        
        # Add to results if text was extracted
        if text_content:
            results.append({
                'filename': filename,
                'text': text_content
            })
            print(f"Successfully processed: {filename}")
    
    # Convert to DataFrame and save to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"\nProcessed {len(results)} files and saved to {output_csv}")
    else:
        print("\nNo PDF or DOCX files found in the specified folder")

if __name__ == "__main__":
    # Example usage

    folder_path = "ARSwithPredictiveAnalytics-DataTraining/input/dataset/linked-in/job_requirements"  # Replace with your folder path
    output_csv = "ARSwithPredictiveAnalytics-DataTraining/output/dataset/linked-in/job_requirements/extracted-text-job-requirements.csv"  # Replace with desired output CSV path

    process_document_folder(folder_path, output_csv)