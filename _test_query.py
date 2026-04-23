import json
from rag_ingestion import query_chromadb_with_fallback, initialize_chromadb_safe, get_rag_collections
import config

client = initialize_chromadb_safe(str(config.CHROMADB_DIR))
collections = get_rag_collections(client, False, False)

def test_query(query):
    print(f"\n--- Query: {query} ---")
    results = query_chromadb_with_fallback(collections, query)
    
    docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    meta = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]
    
    for i, (d, dist, m, id) in enumerate(zip(docs, distances, meta, ids)):
        print(f"[{i+1}] ID: {id} | Distance: {dist:.4f}")
        print(f"    Source: {m.get('source_file')}")
        print(f"    Text: {d[:150]}...\n")

test_query("How are internal marks calculated?")
test_query("Who teaches Data Structures?")
test_query("What electives are available for S5CS?")
