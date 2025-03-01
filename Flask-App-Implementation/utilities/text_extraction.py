import textract

# def extract_text_from_file(file_path):
#     """Extract text from a given file (PDF, DOC, DOCX)."""
#     try:
#         text = textract.process(file_path).decode("utf-8")
#         return text.strip()
#     except Exception as e:
#         print(f"Error extracting text: {str(e)}")
#         return ""

def extract_text(file_path):
    text = textract.process(file_path).decode("utf-8")
    return text
