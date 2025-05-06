import streamlit as st
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

# ---------- Load Preprocessed Data ----------
@st.cache_resource
def load_data():
    with open("chunked_transcript.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [c["text"] for c in chunks]
    starts = [c["start"] for c in chunks]
    ends = [c["end"] for c in chunks]

    # TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

    # BM25
    tokenized_corpus = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    # FAISS (text semantic search)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_tensor=True).detach().cpu().numpy().astype("float32")
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings)

    return chunks, tfidf_vectorizer, tfidf_matrix, bm25, model, faiss_index

# Load everything once
chunks, tfidf_vectorizer, tfidf_matrix, bm25, embedding_model, faiss_index = load_data()

# ---------- Streamlit UI ----------
st.title("🎥 Video Question Answering (RAG System)")
st.write("Ask a natural language question. The system will return the most relevant video segment.")

query = st.text_input("Enter your question:")

if query:
    # ---------- FAISS Semantic Retrieval ----------
    q_emb = embedding_model.encode([query], convert_to_numpy=True).astype("float32")
    _, faiss_ids = faiss_index.search(q_emb, k=1)
    faiss_result = chunks[faiss_ids[0][0]]

    # ---------- TF-IDF ----------
    tfidf_query = tfidf_vectorizer.transform([query])
    tfidf_scores = np.dot(tfidf_matrix, tfidf_query.T).toarray().flatten()
    tfidf_idx = np.argmax(tfidf_scores)
    tfidf_result = chunks[tfidf_idx]

    # ---------- BM25 ----------
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_idx = np.argmax(bm25_scores)
    bm25_result = chunks[bm25_idx]

    # ---------- Display Results ----------
    st.subheader("Semantic - FAISS")
    st.markdown(f"*Timestamp:* {faiss_result['start']}s – {faiss_result['end']}s")
    st.markdown(f"*Transcript:* {faiss_result['text']}")

    # Embedded YouTube video
    YOUTUBE_URL = "https://www.youtube.com/embed/dARr3lGKwk8"
    start_time = int(faiss_result['start'])
    st.subheader("Video Segment")
    st.components.v1.html(f"""
        <iframe width="700" height="400"
            src="{YOUTUBE_URL}?start={start_time}&autoplay=1&modestbranding=1&rel=0"
            frameborder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowfullscreen>
        </iframe>
    """, height=420)

    # TF-IDF and BM25
    st.subheader("TF-IDF Result")
    st.markdown(f"*Timestamp:* {tfidf_result['start']}s – {tfidf_result['end']}s")
    st.markdown(f"*Transcript:* {tfidf_result['text']}")

    st.subheader("BM25 Result")
    st.markdown(f"*Timestamp:* {bm25_result['start']}s – {bm25_result['end']}s")
    st.markdown(f"*Transcript:* {bm25_result['text']}")

    # Fallback message
    if all(len(r["text"].strip()) == 0 for r in [faiss_result, tfidf_result, bm25_result]):
        st.warning("No relevant answer found in the video.")
