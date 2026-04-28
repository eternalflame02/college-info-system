import json
import sys
from pathlib import Path
import config
from chunker.semantic_chunker import SemanticChunkingPipeline
from chunker.chunk_models import save_chunks

def main():
    print("Chunking newly added FAQ file...")
    
    # Initialize pipeline to reuse loading logic
    pipeline = SemanticChunkingPipeline()
    pipeline.load_entities()
    
    # Process only the specific file
    faq_file = Path("data/markdown/pages/cse_frequently_asked_questions.md")
    
    # Run the processing for one file
    new_chunks = pipeline.process_file(faq_file)
    print(f"Generated {len(new_chunks)} chunks from {faq_file.name}")
    
    # Append to chunks.json
    chunks_file = config.CHUNKS_FILE
    existing_chunks = []
    if chunks_file.exists():
        with open(chunks_file, 'r', encoding='utf-8') as f:
            existing_chunks = json.load(f)
            
    # Remove any previous chunks from this specific file to avoid duplicates
    existing_chunks = [c for c in existing_chunks if c.get("source_file") != str(faq_file).replace("\\", "/")]
    
    # Convert new_chunks (which are Chunk objects) to dicts
    new_chunks_dicts = [c.to_dict() for c in new_chunks]
    
    existing_chunks.extend(new_chunks_dicts)
    
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump(existing_chunks, f, indent=2, ensure_ascii=False)
        
    print(f"Total chunks now in chunks.json: {len(existing_chunks)}")

if __name__ == "__main__":
    main()
