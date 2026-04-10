import chromadb
import sys
from sentence_transformers import SentenceTransformer

# Setup Chroma DB connection directly
DB_PATH = "E:/Rohith/COLLEGE/Mini Project/data/chroma_db"
client = chromadb.PersistentClient(path=DB_PATH)
try:
    collection = client.get_collection("mbcet_cse_knowledge")
    print("Successfully connected to ChromaDB collection: mbcet_cse_knowledge")
    print(f"Total entries: {collection.count()}")
except Exception as e:
    print(f"Failed to connect to collection: {e}")
    sys.exit(1)

# Load embedder locally
print("Loading model...")
model = SentenceTransformer("google/embeddinggemma-300m", device="cuda")
model.max_seq_length = 512

queries = [
    "What is the qualification of Dr. Jisha John?",
    "Which faculty member's email is jisha@mbcet.ac.in?",
    "Who are the assistant professors in CSE?",
    "What are the research areas of Mr. Binu D?",
    "Who teaches Artificial Intelligence?",
]

for query in queries:
    print("\n" + "="*80)
    print(f"TEST: {query}")
    print("="*80)
    
    # Generate embedding
    query_emb = model.encode(query, normalize_embeddings=True).tolist()
    
    # Query Chroma
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=3
    )
    
    for j in range(len(results["distances"][0])):
        dist = results["distances"][0][j]
        doc = results["documents"][0][j]
        meta = results["metadatas"][0][j]
        
        print(f"\n>> Rank {j+1} (Distance: {dist:.3f} L2)")
        print(f"   Source: {meta.get('source_file')}")
        
        snippet = doc[:200].replace('\n', ' ')
        if len(doc) > 200: snippet += '...'
        print(f"   Snippet: {snippet}")
