# evaluation.py
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def generate_report(df):
    print("\n=== Evaluation Report ===")
    print(df.to_markdown(index=False))
    
    # Visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Accuracy plot
    ax1.bar(df["Method"], df["Accuracy"])
    ax1.set_title("Accuracy on Answerable Questions")
    ax1.set_ylim(0, 1)
    
    # False Positive plot
    ax2.bar(df["Method"], df["False Positive Rate"])
    ax2.set_title("False Positive Rate")
    ax2.set_ylim(0, 1)
    
    # Latency plot
    ax3.bar(df["Method"], df["Avg Latency (ms)"])
    ax3.set_title("Average Query Latency (ms)")
    
    plt.tight_layout()
    plt.savefig("evaluation_results.png")
    print("\nSaved visualization to evaluation_results.png")

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
