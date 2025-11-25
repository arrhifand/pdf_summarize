import fitz  # PyMuPDF

def extract_text_from_pdf(file):
    """
    Membaca PDF dan mengekstraksi teks secara lengkap.
    
    Parameter:
        file: file object dari Streamlit (UploadedFile)

    Return:
        text (str): hasil ekstraksi teks PDF
    """
    pdf_data = file.read()
    doc = fitz.open(stream=pdf_data, filetype="pdf")

    extracted_text = ""
    for page in doc:
        extracted_text += page.get_text()

    return extracted_text
