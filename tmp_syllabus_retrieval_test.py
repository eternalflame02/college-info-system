import rag_ingestion as r
import config

queries = [
    "What are the electives for s7ct and explain syllabus in detail?",
    "I'm not a fan of theory subjects so which one should I pick from these electives?",
    "What type of subject is Artificial Intelligence? Theory based or application?",
    "What are common subjects for cs and ct in S6?",
    "I'm planning to move into a ui/ux field so what minor will be apt for that?",
]

print("Initializing ChromaDB...")
client = r.initialize_chromadb_safe(str(config.CHROMADB_DIR))
collection_map = r.get_rag_collections(client, recreate=False, create_missing=False)

print("Loading embedding model once...")
model = r.load_embedding_model(device="auto")
r.load_embedding_model = lambda device="auto": model

for i, q in enumerate(queries, 1):
    qtype = r.classify_query_type(q)
    results = r.query_chromadb_with_fallback(
        collection_map,
        q,
        query_type=qtype,
        enable_fallback=True,
        rerank_mixed=True,
    )
    print("\n" + "=" * 110)
    print(f"Q{i}: {q}")
    print(f"query_type={qtype} quality={results.get('quality')} best_distance={results.get('best_distance')} threshold={results.get('threshold_used')}")
    print(f"fallback_triggered={results.get('fallback_triggered', False)} filtered_count={results.get('filtered_count', 0)} type_mix={results.get('content_type_distribution', {})}")

    count = min(results.get('filtered_count', 0), 3)
    if count == 0:
        print("top_hits=[]")
        continue

    top_hits = []
    for idx in range(count):
        meta = results['metadatas'][0][idx] if results.get('metadatas') else {}
        dist = float(results['distances'][0][idx])
        top_hits.append({
            'rank': idx + 1,
            'distance': round(dist, 3),
            'content_type': meta.get('content_type'),
            'source_file': meta.get('source_file'),
            'section_hierarchy': meta.get('section_hierarchy'),
        })
    print(f"top_hits={top_hits}")
