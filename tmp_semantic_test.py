import sys
import os
import json
import torch

# Ensure we're in the right directory
os.chdir(r"e:\Rohith\COLLEGE\Mini Project")

# Add to path so config and rag_ingestion can be imported
sys.path.append(os.getcwd())

import config
from rag_ingestion import initialize_chromadb_safe, load_embedding_model, query_chromadb, classify_query_type

print("="*80)
print("SEMANTIC SIMILARITY TESTS")
print("="*80)

# 1. Initialize DB and Model
print("Initializing ChromaDB...")
client = initialize_chromadb_safe(str(config.CHROMADB_DIR))
collection = client.get_collection(name=config.CHROMADB_COLLECTION)
print(f"Loaded collection with {collection.count()} documents.")

print("Loading embedding model...")
model = load_embedding_model(config.EMBEDDING_MODEL, device="auto")

# 2. Define Test Queries
queries = [
    "Who is the Head of the Computer Science department?",
    "Machine Learning course in semester 6",
    "Where did Dr. Jisha John get her Ph.D?", # Specific faculty detail
    "What are the subjects in S3?", # Semester query
    "how many credits for mini project?", # General academic query
]

for q in queries:
    print("\n" + "-"*80)
    print(f"QUERY: '{q}'")
    q_type = classify_query_type(q)
    print(f"ROUTED AS: {q_type}")

    # Generate query embedding
    q_emb = model.encode(
        [q], 
        convert_to_numpy=True, 
        normalize_embeddings=True
    ).tolist()

    # Query Chroma
    results = collection.query(
        query_embeddings=q_emb,
        n_results=3,
        where={"content_type": "profile"} if q_type == "faculty" else ({"content_type": "table"} if q_type in ["course", "timetable"] else None)
    )

    if not results['distances'] or not results['distances'][0]:
        print("NO RESULTS FOUND.")
        continue

    for j in range(len(results['distances'][0])):
        dist = results['distances'][0][j]
        doc = results['documents'][0][j]
        meta = results['metadatas'][0][j]
        
        print(f"\n[{j+1}] DISTANCE: {dist:.4f} | TYPE: {meta.get('content_type')} | FILE: {meta.get('source_file')}")
        
        # Print a short preview of the text
        preview = doc[:250].replace('\n', ' ')
        if len(doc) > 250:
            preview += "..."
        print(f"TEXT PREVIEW:\n{preview}")

print("\n" + "="*80)
