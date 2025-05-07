import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import os
import logging
from PIL import Image
import numpy as np

# Set up logging
logging.basicConfig(filename="../logs/pdf_processing.log", level=logging.INFO)

def is_text_based_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            first_page = pdf.pages[0]
            text = first_page.extract_text() or ""
            return len(text.strip()) > 50
    except Exception:
        return False

def enhance_image(image):
    # Enhance image for better OCR
    image = image.convert("L")
    image = np.array(image)
    image = (image - image.min()) * (255 / (image.max() - image.min()))
    return Image.fromarray(image.astype(np.uint8))

def extract_text_based_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        logging.info(f"Extracted text from {pdf_path}")
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {e}")
    return text

def extract_scanned_pdf(pdf_path):
    text = ""
    try:
        images = convert_from_path(pdf_path)
        for image in images:
            enhanced_image = enhance_image(image)
            page_text = pytesseract.image_to_string(enhanced_image, lang="eng")
            text += page_text + "\n"
        logging.info(f"Extracted text from {pdf_path} using OCR")
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {e}")
    return text

def clean_text(text):
    text = " ".join(text.split())
    text = text.replace("|", "").replace("~", "")
    text = "\n".join(line for line in text.splitlines() if not line.strip().isdigit())
    return text

def extract_text(pdf_path):
    if is_text_based_pdf(pdf_path):
        text = extract_text_based_pdf(pdf_path)
    else:
        text = extract_scanned_pdf(pdf_path)
    return clean_text(text)

def process_pdfs(input_dir="../legal_docs/", output_dir="../extracted_text/"):
    os.makedirs(output_dir, exist_ok=True)
    documents = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            text = extract_text(pdf_path)
            if text.strip():
                documents.append({"filename": filename, "content": text})
                output_path = os.path.join(output_dir, filename.replace(".pdf", ".txt"))
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(text)
            else:
                logging.warning(f"No text extracted from {filename}")
    return documents

if __name__ == "__main__":
    documents = process_pdfs()
    print(f"Processed {len(documents)} documents")