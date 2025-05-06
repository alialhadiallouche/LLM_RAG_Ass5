# evaluation.py
import time
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

# Load data and models (similar to your app.py)
def load_resources():
    with open("chunked_transcript.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    with open("gold_set_test.json", "r", encoding="utf-8") as f:
        gold_set = json.load(f)
    
    texts = [c["text"] for c in chunks]
    starts = [c["start"] for c in chunks]
    ends = [c["end"] for c in chunks]

    # Initialize all retrieval methods
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    
    tokenized_corpus = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_tensor=True).detach().cpu().numpy().astype("float32")
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings)
    
    return chunks, gold_set, {
        "tfidf": (tfidf_vectorizer, tfidf_matrix),
        "bm25": bm25,
        "faiss": (model, faiss_index)
    }

# Evaluation functions
def evaluate_retrieval(chunks, gold_set, resources):
    methods = ["TF-IDF", "BM25", "FAISS"]
    results = {
        "Method": [],
        "Accuracy": [],
        "False Positive Rate": [],
        "Avg Latency (ms)": []
    }
    
    # Process answerable questions
    for method in methods:
        correct = 0
        latencies = []
        
        for q in gold_set["answerable_questions"]:
            start_time = time.time()
            
            if method == "TF-IDF":
                vectorizer, matrix = resources["tfidf"]
                query_vec = vectorizer.transform([q["question"]])
                scores = np.dot(matrix, query_vec.T).toarray().flatten()
                best_idx = np.argmax(scores)
                
            elif method == "BM25":
                bm25 = resources["bm25"]
                scores = bm25.get_scores(q["question"].lower().split())
                best_idx = np.argmax(scores)
                
            else:  # FAISS
                model, index = resources["faiss"]
                q_emb = model.encode([q["question"]], convert_to_numpy=True).astype("float32")
                _, best_idx = index.search(q_emb, k=1)
                best_idx = best_idx[0][0]
            
            latency = (time.time() - start_time) * 1000
            latencies.append(latency)
            
            # Check if retrieved timestamp matches ground truth
            retrieved_chunk = chunks[best_idx]
            gt_start, gt_end = map(float, q["timestamp"].split(" - "))
            if (retrieved_chunk["start"] <= gt_start <= retrieved_chunk["end"]) or \
               (retrieved_chunk["start"] <= gt_end <= retrieved_chunk["end"]):
                correct += 1
        
        # Process unanswerable questions
        false_positives = 0
        for q in gold_set["unanswerable_questions"]:
            start_time = time.time()
            
            if method == "TF-IDF":
                vectorizer, matrix = resources["tfidf"]
                query_vec = vectorizer.transform([q["question"]])
                scores = np.dot(matrix, query_vec.T).toarray().flatten()
                best_idx = np.argmax(scores)
                
            elif method == "BM25":
                bm25 = resources["bm25"]
                scores = bm25.get_scores(q["question"].lower().split())
                best_idx = np.argmax(scores)
                
            else:  # FAISS
                model, index = resources["faiss"]
                q_emb = model.encode([q["question"]], convert_to_numpy=True).astype("float32")
                _, best_idx = index.search(q_emb, k=1)
                best_idx = best_idx[0][0]
            
            # Consider it a false positive if the top result has non-empty text
            if chunks[best_idx]["text"].strip():
                false_positives += 1
        
        # Store results
        results["Method"].append(method)
        results["Accuracy"].append(correct / len(gold_set["answerable_questions"]))
        results["False Positive Rate"].append(false_positives / len(gold_set["unanswerable_questions"]))
        results["Avg Latency (ms)"].append(np.mean(latencies))
    
    return pd.DataFrame(results)

# Replace the matplotlib visualization with this Streamlit version
def generate_report(df):
    st.write("\n### Evaluation Results")
    st.table(df)
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Accuracy on Answerable Questions**")
        st.bar_chart(df.set_index("Method")["Accuracy"])
    
    with col2:
        st.write("**False Positive Rate**")
        st.bar_chart(df.set_index("Method")["False Positive Rate"])
    
    with col3:
        st.write("**Average Latency (ms)**")
        st.bar_chart(df.set_index("Method")["Avg Latency (ms)"])


def analyze_failures(chunks, gold_set, resources):
    print("\n=== Failure Analysis ===")
    
    # Example: Analyze why some answerable questions failed
    method = "FAISS"  # Can change to analyze other methods
    model, index = resources["faiss"]
    
    for q in gold_set["answerable_questions"]:
        q_emb = model.encode([q["question"]], convert_to_numpy=True).astype("float32")
        _, best_idx = index.search(q_emb, k=1)
        retrieved_chunk = chunks[best_idx[0][0]]
        
        gt_start, gt_end = map(float, q["timestamp"].split(" - "))
        if not ((retrieved_chunk["start"] <= gt_start <= retrieved_chunk["end"]) or 
                (retrieved_chunk["start"] <= gt_end <= retrieved_chunk["end"])):
            print(f"\nQuestion: {q['question']}")
            print(f"Expected: {q['timestamp']} ({q['answer'][:50]}...)")
            print(f"Retrieved: {retrieved_chunk['start']}-{retrieved_chunk['end']} ({retrieved_chunk['text'][:50]}...)")
            print("Potential reason: Semantic mismatch between question phrasing and transcript segments")

if __name__ == "__main__":
    chunks, gold_set, resources = load_resources()
    results_df = evaluate_retrieval(chunks, gold_set, resources)
    generate_report(results_df)
    analyze_failures(chunks, gold_set, resources)
