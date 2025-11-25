import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. TF-IDF + KMeans  (Algoritma 1)
# ============================================================

def clustering_tfidf(sentences, n_clusters=3):
    """
    Melakukan clustering kalimat menggunakan TF-IDF + KMeans.

    Parameter:
        sentences (list): daftar kalimat
        n_clusters (int): jumlah cluster

    Return:
        labels (list): nomor cluster setiap kalimat
    """
    vectorizer = TfidfVectorizer(stop_words='indonesian')
    X = vectorizer.fit_transform(sentences)

    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X)

    return labels


# ============================================================
# 2. SBERT Embedding + Agglomerative (Algoritma 2)
# ============================================================

def clustering_sbert(sentences, n_clusters=3):
    """
    Clustering berbasis makna (semantic clustering)
    menggunakan Sentence-BERT + Agglomerative Clustering.

    Parameter:
        sentences (list): daftar kalimat
        n_clusters (int): jumlah cluster

    Return:
        labels (list): cluster setiap kalimat
    """

    # SBERT Embeddings (model terbaik & ringan)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(sentences)

    # Agglomerative clustering
    cluster_model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = cluster_model.fit_predict(embeddings)

    return labels
