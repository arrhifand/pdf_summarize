import streamlit as st
import tempfile
import nltk
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from utils.pdf_reader import extract_text_from_pdf
from utils.summarize import load_flan_model, summarize_text
from utils.clustering import clustering_tfidf, clustering_sbert
# ============================================================
# PREPROCESSING: PDF EXTRACTOR
# ============================================================

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# ============================================================
# LOAD FLAN-T5 MODEL
# ============================================================

@st.cache_resource
def load_flan_model():
    MODEL_DIR = "model/flan_t5/"
    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
    return tokenizer, model


def summarize_text(text, tokenizer, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    input_ids = tokenizer.encode(
        "summarize: " + text,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    ).to(device)

    output = model.generate(
        input_ids,
        max_length=250,
        min_length=50,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    return summary


# ============================================================
# CLUSTERING MODULE
# ============================================================

def clustering_tfidf(sentences, n_clusters=3):
    vectorizer = TfidfVectorizer(stop_words='indonesian')
    X = vectorizer.fit_transform(sentences)

    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X)

    return labels


def clustering_sbert(sentences, n_clusters=3):
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = sbert.encode(sentences)

    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(embeddings)

    return labels


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Summarization & Content Analysis Putusan",
    layout="wide"
)

st.title("📘 Analisis Putusan Pengadilan – Summarization & Clustering")
st.write("Upload PDF putusan untuk diringkas dan dianalisis.")

uploaded_file = st.file_uploader("Upload PDF Putusan", type=["pdf"])

if uploaded_file is not None:

    # Extract text
    st.subheader("📄 Ekstraksi Teks dari PDF")
    extracted_text = extract_text_from_pdf(uploaded_file)
    st.text_area("Hasil Ekstraksi Teks:", extracted_text[:2000] + "...", height=200)

    # Load model
    tokenizer, model = load_flan_model()

    # Summarization
    st.subheader("📝 Ringkasan Putusan (FLAN-T5)")
    if st.button("Generate Summary"):
        summary = summarize_text(extracted_text, tokenizer, model)
        st.success("Ringkasan berhasil dibuat!")
        st.write(summary)

    # Clustering
    st.subheader("🔍 Content Analysis – Clustering Alasan Perceraian")

    nltk.download("punkt")
    sentences = nltk.sent_tokenize(extracted_text)

    algo = st.selectbox(
        "Pilih Metode Clustering:",
        ["TF-IDF + KMeans", "SBERT + Agglomerative"]
    )

    n_clusters = st.slider("Jumlah Cluster:", 2, 10, 3)

    if st.button("Proses Clustering"):

        if algo == "TF-IDF + KMeans":
            labels = clustering_tfidf(sentences, n_clusters=n_clusters)
        else:
            labels = clustering_sbert(sentences, n_clusters=n_clusters)

        st.success("Clustering selesai!")

        # Display grouped results
        for cluster_id in range(n_clusters):
            st.markdown(f"### 📌 Cluster {cluster_id}")
            cluster_sentences = [sent for sent, label in zip(sentences, labels) if label == cluster_id]

            for s in cluster_sentences:
                st.write("- " + s)
            st.write("\n")

