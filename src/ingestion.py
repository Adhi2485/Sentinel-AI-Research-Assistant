import fitz  # PyMuPDF
import re

def clean_text(text: str) -> str:
    """Basic text cleaning to remove extra whitespace and noise."""
    text = re.sub(r'\s+', ' ', text)
    # Remove extremely short lines that might be headers/footers
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    return text.strip()

def extract_text_from_pdf(pdf_path: str, filename: str) -> list[dict]:
    """
    Extracts text from a PDF document.
    Returns a list of dictionaries containing page-level text and metadata.
    """
    doc = fitz.open(pdf_path)
    extracted_data = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        cleaned_text = clean_text(text)
        
        if cleaned_text:
            extracted_data.append({
                "document_name": filename,
                "page_number": page_num + 1,
                "text": cleaned_text
            })
            
    doc.close()
    return extracted_data
