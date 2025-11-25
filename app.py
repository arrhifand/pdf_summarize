import streamlit as st
import pdfplumber
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# ============================================================
# 1. LOAD MODEL FLAN-T5
# ============================================================
@st.cache_resource
def load_model():
    model_path = "models/flan_t5_model"   # pastikan folder ini benar

    tokenizer = T5Tokenizer.from_pretrained(model_path)

    # Auto device placement (GPU/CPU)
    model = T5ForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto"
    )

    return tokenizer, model


# ============================================================
# 2. PDF TEXT EXTRACTION
# ============================================================
def extract_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text.strip()


# ============================================================
# 3. STREAMLIT UI
# ============================================================
st.set_page_config(
    page_title="PDF Putusan Summarizer (Flan-T5)",
    page_icon="📘",
    layout="wide"
)

st.title("📘 Putusan Perkara – Summarizer (Flan-T5)")
st.write("Upload file PDF putusan, dan sistem akan otomatis merangkum isi dokumen menggunakan model Flan-T5.")

uploaded_file = st.file_uploader("Upload PDF Putusan", type=["pdf"])

if uploaded_file:
    st.info("📄 Sedang mengekstraksi teks dari PDF...")

    pdf_text = extract_text(uploaded_file)

    st.subheader("Isi PDF (Preview)")
    st.text_area("Teks Original:", pdf_text[:2000] + " ...", height=250)

    tokenizer, model = load_model()

    if st.button("Generate Summary"):
        with st.spinner("🔄 Menghasilkan ringkasan..."):

            # Prefix khusus Flan-T5
            input_text = "summarize: " + pdf_text

            input_ids = tokenizer.encode(
                input_text,
                return_tensors="pt",
                max_length=2048,
                truncation=True
            )

            # GENERATE SUMMARY
            summary_ids = model.generate(
                input_ids,
                max_length=300,
                min_length=80,
                num_beams=4,
                length_penalty=1.0,
                early_stopping=True
            )

            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        st.subheader("🔍 Ringkasan Dokumen")
        st.write(summary)
