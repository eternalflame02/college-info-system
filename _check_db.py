from rag_ingestion import initialize_chromadb_safe, get_rag_collections
import config

client = initialize_chromadb_safe(str(config.CHROMADB_DIR))
cols = get_rag_collections(client, False, False)
c = cols['legacy']

res = c.get(where={'source_file': 'data\\markdown\\pages\\cse_frequently_asked_questions.md'})
print(f"Found {len(res['ids'])} chunks using backslashes. IDs: {res['ids']}")

res2 = c.get(where={'source_file': 'data/markdown/pages/cse_frequently_asked_questions.md'})
print(f"Found {len(res2['ids'])} chunks using forward slashes. IDs: {res2['ids']}")

all_files = set(m.get('source_file') for m in c.get()['metadatas'] if m.get('source_file'))
print(f"Sample source_file values in DB: {list(all_files)[:5]}")
print(f"Does faq exist in source_files? {any('faq' in f or 'frequently' in f for f in all_files)}")
