import docx
import PyPDF2
import textract


def extract_text_from_docx(file):
    """Extract text from a DOCX file"""
    try:
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return None

def extract_text_from_pdf(file):
    """Extract text from a PDF file"""
    try:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text if text else None
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

def extract_text_with_textract(file):
    """Fallback method using textract"""
    try:
        return textract.process(file).decode("utf-8")
    except Exception as e:
        print(f"Error extracting text using textract: {e}")
        return None

def extract_text(file):
    """Extract text based on file type"""
    if file.filename.endswith(".docx"):
        text = extract_text_from_docx(file)
    elif file.filename.endswith(".pdf"):
        text = extract_text_from_pdf(file)
    else:
        print(f"Unsupported file format: {file.filename}")
        return None

    # Fallback to textract if needed
    if text is None:
        text = extract_text_with_textract(file)

    return text
