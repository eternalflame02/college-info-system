# 🔧 PRODUCTION-GRADE PROMPT FOR CODING AGENT

## ChromaDB RAG System - Embedding & Retrieval Pipeline

You are a **senior Python ML engineer** building a **production-grade RAG (Retrieval-Augmented Generation) system** for the MBCET CSE academic knowledge base.

Your task is to **embed 2,060 semantic chunks** using **Google EmbeddingGemma** and ingest them into **ChromaDB** with advanced retrieval capabilities optimized for **table-heavy academic content**.

---

## 🎯 OVERALL OBJECTIVE

Build a complete RAG ingestion and retrieval system that:

* Embeds 2,060 chunks using **`google/embeddinggemma-300m`** (768 dimensions)
* Stores embeddings in **ChromaDB** with rich metadata
* Implements **query routing** for optimal retrieval across different content types
* Handles **table-dominant data** (88% of chunks are tables)
* Provides **adaptive distance thresholding** for result quality
* Generates **comprehensive validation reports**
* Optimizes for **GTX 1650 Ti GPU** (4GB VRAM)

---

## 🧱 INPUTS

### 1. Semantic Chunks

**Location:** `data/chunks/chunks.json`

**Structure:**
```json
[
  {
    "chunk_id": "pdf_syllabus_sem3_courses_abc12345",
    "text": "| Semester | Course Code | Course Name | Credits |\n|---|---|---|---|\n| 3 | CS301 | DBMS | 4 |",
    "source_type": "html | pdf",
    "source_file": "data/markdown/pdfs/syllabus.md",
    "section_hierarchy": ["Academics", "B.Tech CSE", "Semester 3"],
    "content_type": "table | profile | section | regulation | list",
    "entity_refs": ["course_cs301", "faculty_dr_jisha_john"],
    "page_range": [5, 5] | null,
    "word_count": 180,
    "hash": "abc12345..."
  }
]
```

### 2. Chunk Statistics

**Location:** `data/chunks/chunk_report.json`

```json
{
  "total_chunks": 2060,
  "duplicates_skipped": 440,
  "chunks_by_type": {
    "table": 1829,
    "profile": 101,
    "regulation": 50,
    "section": 69,
    "list": 11
  },
  "chunks_by_source": {
    "html": 191,
    "pdf": 1869
  }
}
```

---

## 📤 OUTPUTS

### 1. ChromaDB Persistent Storage

**Location:** `./chroma_db/`

**Collection:** `mbcet_cse_knowledge` (single collection, all 2,060 chunks)

### 2. Embedding Cache (Backup)

**Location:** `data/embeddings/embedding_cache.npz`

```python
# NumPy compressed format
{
    "chunk_ids": [...],           # 2060 IDs
    "embeddings": [...],          # 2060 x 768 float32 array
    "model": "google/embeddinggemma-300m",
    "dimensions": 768
}
```

### 3. Ingestion Report

**Location:** `data/validation/chromadb_ingestion_report.json`

```json
{
  "ingestion_timestamp": "2025-03-12T14:30:00Z",
  "total_chunks_processed": 2060,
  "successfully_ingested": 2060,
  "failed_chunks": [],
  "embedding_stats": {
    "model": "google/embeddinggemma-300m",
    "dimensions": 768,
    "batch_size": 64,
    "total_time_seconds": 28.5,
    "chunks_per_second": 72.3,
    "gpu_used": true,
    "vram_peak_mb": 1650
  },
  "chromadb_stats": {
    "collection_name": "mbcet_cse_knowledge",
    "total_documents": 2060,
    "chunks_by_type": {
      "table": 1829,
      "profile": 101,
      "regulation": 50,
      "section": 69,
      "list": 11
    }
  }
}
```

### 4. Validation Report

**Location:** `data/validation/chromadb_validation_report.json`

```json
{
  "validation_timestamp": "2025-03-12T14:31:00Z",
  "collection_health": {
    "expected_count": 2060,
    "actual_count": 2060,
    "status": "PASS"
  },
  "content_type_validation": {
    "table": {"expected": 1829, "actual": 1829, "status": "PASS"},
    "profile": {"expected": 101, "actual": 101, "status": "PASS"},
    "regulation": {"expected": 50, "actual": 50, "status": "PASS"},
    "section": {"expected": 69, "actual": 69, "status": "PASS"},
    "list": {"expected": 11, "actual": 11, "status": "PASS"}
  },
  "sample_query_tests": [
    {
      "query": "Who is the head of CSE department?",
      "expected_content_type": "profile",
      "top_result_content_type": "profile",
      "top_result_distance": 0.23,
      "status": "PASS"
    },
    {
      "query": "What courses are in Semester 3?",
      "expected_content_type": "table",
      "top_result_content_type": "table",
      "top_result_distance": 0.18,
      "status": "PASS"
    }
  ],
  "performance_metrics": {
    "average_query_time_ms": 12.5,
    "p95_query_time_ms": 18.2,
    "p99_query_time_ms": 24.1
  }
}
```

### 5. Activity Logs

**Location:** `logs/chromadb_ingestion.log`

Structured JSON logs for all operations.

---

## 🧠 ARCHITECTURE REQUIREMENTS

### Module Structure

Create: `rag_ingestion.py`

**Must expose these functions:**

```python
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb

# ========== EMBEDDING FUNCTIONS ==========

def load_embedding_model(model_name: str = "google/embeddinggemma-300m", device: str = "auto") -> SentenceTransformer:
    """
    Load EmbeddingGemma model.
    
    Args:
        model_name: Model identifier
        device: 'cuda', 'cpu', or 'auto' (auto-detect GPU)
    
    Returns:
        SentenceTransformer model instance
    """
    pass

def generate_embeddings(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 64,
    show_progress: bool = True
) -> np.ndarray:
    """
    Generate embeddings for all chunks.
    
    Args:
        model: Loaded SentenceTransformer model
        texts: List of chunk texts
        batch_size: Batch size (64 optimal for GTX 1650 Ti)
        show_progress: Show progress bar
    
    Returns:
        NumPy array of shape (len(texts), 768)
    """
    pass

def cache_embeddings(
    chunk_ids: List[str],
    embeddings: np.ndarray,
    output_path: str = "data/embeddings/embedding_cache.npz"
) -> None:
    """
    Cache embeddings to disk for fast re-loading.
    
    Args:
        chunk_ids: List of chunk IDs
        embeddings: NumPy array of embeddings
        output_path: Where to save cache
    """
    pass

def load_cached_embeddings(cache_path: str) -> Tuple[List[str], np.ndarray]:
    """
    Load embeddings from cache if exists.
    
    Returns:
        (chunk_ids, embeddings) or (None, None) if cache doesn't exist
    """
    pass

# ========== CHROMADB FUNCTIONS ==========

def initialize_chromadb(persist_directory: str = "./chroma_db") -> chromadb.PersistentClient:
    """
    Initialize ChromaDB persistent client.
    
    Args:
        persist_directory: Where to store ChromaDB data
    
    Returns:
        ChromaDB client instance
    """
    pass

def create_collection(
    client: chromadb.PersistentClient,
    collection_name: str = "mbcet_cse_knowledge",
    recreate: bool = False
) -> chromadb.Collection:
    """
    Create or get existing collection.
    
    Args:
        client: ChromaDB client
        collection_name: Name of collection
        recreate: If True, delete existing collection and create new one
    
    Returns:
        ChromaDB collection instance
    """
    pass

def prepare_metadata(chunk: Dict) -> Dict:
    """
    Extract and prepare metadata for ChromaDB from chunk.
    
    Args:
        chunk: Chunk dictionary from chunks.json
    
    Returns:
        Metadata dictionary with extracted structured fields
    
    Example output:
    {
        "source_file": "data/markdown/pdfs/syllabus.md",
        "source_type": "pdf",
        "content_type": "table",
        "section_hierarchy": "Academics > B.Tech CSE > Semester 3",
        "entity_refs": "course_cs301,course_cs302",
        "word_count": 180,
        "semester": 3,  # Extracted if available
        "has_table": true,
        "table_course_count": 2  # Parsed from table if content_type == "table"
    }
    """
    pass

def ingest_chunks_to_chromadb(
    collection: chromadb.Collection,
    chunks: List[Dict],
    embeddings: np.ndarray,
    batch_size: int = 100
) -> Dict:
    """
    Ingest chunks with embeddings into ChromaDB.
    
    Args:
        collection: ChromaDB collection
        chunks: List of chunk dictionaries
        embeddings: Pre-generated embeddings
        batch_size: Batch size for ChromaDB ingestion
    
    Returns:
        Ingestion statistics dictionary
    """
    pass

# ========== VALIDATION FUNCTIONS ==========

def validate_chromadb_ingestion(
    collection: chromadb.Collection,
    expected_stats: Dict,
    test_queries: List[Dict]
) -> Dict:
    """
    Comprehensive validation of ChromaDB ingestion.
    
    Args:
        collection: ChromaDB collection to validate
        expected_stats: Expected counts from chunk_report.json
        test_queries: List of test query dictionaries
    
    Returns:
        Validation report dictionary
    """
    pass

# ========== QUERY ROUTING FUNCTIONS ==========

def classify_query_type(query: str) -> str:
    """
    Classify query into type for routing.
    
    Args:
        query: User query string
    
    Returns:
        Query type: "faculty" | "course" | "timetable" | "regulation" | "general"
    
    Logic:
    - If contains "who", "faculty", "professor", "HOD", "head" → "faculty"
    - If contains "course", "subject", "semester X" → "course"
    - If contains "timetable", "schedule", "timing" → "timetable"
    - If contains "regulation", "R2019", "R2023", "curriculum" → "regulation"
    - Else → "general"
    """
    pass

def query_chromadb(
    collection: chromadb.Collection,
    query_text: str,
    query_type: str = None,
    n_results: int = 5,
    content_type_filter: str = None,
    distance_threshold: float = None
) -> Dict:
    """
    Query ChromaDB with routing and adaptive filtering.
    
    Args:
        collection: ChromaDB collection
        query_text: User query
        query_type: Classified query type (auto-detect if None)
        n_results: Number of results to return
        content_type_filter: Filter by specific content_type
        distance_threshold: Maximum distance (None = adaptive)
    
    Returns:
        Query results with metadata
    
    Routing Logic:
    - "faculty" queries → filter content_type="profile", n_results=3
    - "course" queries → filter content_type="table", n_results=10
    - "regulation" queries → filter content_type="regulation", n_results=5
    - "general" queries → no filter, n_results=5
    """
    pass

def apply_adaptive_distance_threshold(results: Dict, query_type: str) -> Dict:
    """
    Apply adaptive distance threshold based on best result quality.
    
    Logic:
    - If best result distance < 0.3 (excellent) → threshold = 0.6
    - If best result distance 0.3-0.5 (good) → threshold = 0.7
    - If best result distance > 0.5 (poor) → threshold = 0.8, warn user
    
    Args:
        results: Raw ChromaDB query results
        query_type: Classified query type
    
    Returns:
        Filtered results with quality warnings
    """
    pass

# ========== MAIN PIPELINE ==========

def run_ingestion_pipeline(
    chunks_path: str = "data/chunks/chunks.json",
    chunk_report_path: str = "data/chunks/chunk_report.json",
    force_reembed: bool = False
) -> None:
    """
    Main ingestion pipeline.
    
    Steps:
    1. Load chunks from JSON
    2. Check for cached embeddings
    3. If not cached or force_reembed:
        a. Load EmbeddingGemma model
        b. Generate embeddings (batch_size=64)
        c. Cache embeddings
    4. Initialize ChromaDB
    5. Create collection
    6. Ingest chunks with metadata
    7. Validate ingestion
    8. Generate reports
    9. Print summary
    
    Args:
        chunks_path: Path to chunks.json
        chunk_report_path: Path to chunk_report.json
        force_reembed: Force re-embedding even if cache exists
    """
    pass
```

---

## 🔧 DETAILED IMPLEMENTATION SPECIFICATIONS

### 1. Embedding Generation

#### Model Loading

```python
import torch
from sentence_transformers import SentenceTransformer

def load_embedding_model(model_name="google/embeddinggemma-300m", device="auto"):
    """
    Load EmbeddingGemma with optimal settings for GTX 1650 Ti.
    """
    # Auto-detect device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading {model_name} on {device}...")
    
    model = SentenceTransformer(model_name, device=device)
    
    # Verify model loaded correctly
    print(f"✅ Model loaded: {model_name}")
    print(f"   Device: {device}")
    print(f"   Embedding dimensions: {model.get_sentence_embedding_dimension()}")
    
    return model
```

#### Embedding Generation with Progress Tracking

```python
import numpy as np
from tqdm import tqdm

def generate_embeddings(
    model,
    texts,
    batch_size=64,
    show_progress=True
):
    """
    Generate embeddings with optimal batching for GTX 1650 Ti.
    
    Batch size 64 is optimal:
    - Model size: ~1.2GB VRAM
    - Batch (64 chunks): ~400MB VRAM
    - Total: ~1.6GB (safe for 4GB GPU)
    """
    print(f"Generating embeddings for {len(texts)} chunks...")
    print(f"Batch size: {batch_size}")
    
    # Normalize text (optional but recommended)
    texts = [t.strip() for t in texts]
    
    # Generate embeddings
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True  # L2 normalization for cosine similarity
    )
    
    print(f"✅ Generated {len(embeddings)} embeddings")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Dtype: {embeddings.dtype}")
    
    return embeddings
```

#### Caching Strategy

```python
import os
import numpy as np

def cache_embeddings(chunk_ids, embeddings, output_path="data/embeddings/embedding_cache.npz"):
    """
    Cache embeddings to disk for instant re-loading.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    np.savez_compressed(
        output_path,
        chunk_ids=np.array(chunk_ids),
        embeddings=embeddings,
        model="google/embeddinggemma-300m",
        dimensions=768
    )
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Embeddings cached to {output_path}")
    print(f"   File size: {file_size_mb:.2f} MB")

def load_cached_embeddings(cache_path="data/embeddings/embedding_cache.npz"):
    """
    Load embeddings from cache.
    """
    if not os.path.exists(cache_path):
        print("⚠️  No embedding cache found")
        return None, None
    
    print(f"Loading embeddings from cache: {cache_path}")
    
    data = np.load(cache_path, allow_pickle=True)
    chunk_ids = data['chunk_ids'].tolist()
    embeddings = data['embeddings']
    
    print(f"✅ Loaded {len(chunk_ids)} cached embeddings")
    print(f"   Shape: {embeddings.shape}")
    
    return chunk_ids, embeddings
```

---

### 2. ChromaDB Ingestion

#### Collection Creation

```python
import chromadb

def initialize_chromadb(persist_directory="./chroma_db"):
    """
    Initialize persistent ChromaDB client.
    """
    print(f"Initializing ChromaDB at {persist_directory}...")
    
    client = chromadb.PersistentClient(path=persist_directory)
    
    print(f"✅ ChromaDB initialized")
    print(f"   Storage path: {persist_directory}")
    
    return client

def create_collection(client, collection_name="mbcet_cse_knowledge", recreate=False):
    """
    Create or get collection.
    """
    # Delete existing if recreate=True
    if recreate:
        try:
            client.delete_collection(name=collection_name)
            print(f"🗑️  Deleted existing collection: {collection_name}")
        except:
            pass
    
    # Create collection
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "description": "MBCET CSE Department Knowledge Base",
            "embedding_model": "google/embeddinggemma-300m",
            "embedding_dimensions": 768,
            "total_chunks": 2060
        }
    )
    
    print(f"✅ Collection ready: {collection_name}")
    print(f"   Current document count: {collection.count()}")
    
    return collection
```

#### Metadata Preparation with Table Structure Extraction

```python
import re

def prepare_metadata(chunk):
    """
    Extract structured metadata from chunk.
    
    For table chunks, extract additional structure info.
    """
    metadata = {
        "source_file": chunk["source_file"],
        "source_type": chunk["source_type"],
        "content_type": chunk["content_type"],
        "section_hierarchy": " > ".join(chunk["section_hierarchy"]),
        "entity_refs": ",".join(chunk["entity_refs"]) if chunk["entity_refs"] else "",
        "word_count": chunk["word_count"]
    }
    
    # Add page range if available
    if chunk.get("page_range"):
        metadata["page_start"] = chunk["page_range"][0]
        metadata["page_end"] = chunk["page_range"][1]
    
    # Extract semester if mentioned in section_hierarchy
    for section in chunk["section_hierarchy"]:
        semester_match = re.search(r'Semester\s+(\d+)', section, re.IGNORECASE)
        if semester_match:
            metadata["semester"] = int(semester_match.group(1))
            break
    
    # For table chunks, extract table-specific metadata
    if chunk["content_type"] == "table":
        metadata["has_table"] = True
        
        # Count table rows (rough estimate)
        row_count = chunk["text"].count('\n|') - 2  # Subtract header and separator
        metadata["table_row_count"] = max(row_count, 0)
        
        # Detect if table contains courses
        if any(entity.startswith("course_") for entity in chunk["entity_refs"]):
            metadata["table_contains_courses"] = True
            metadata["table_course_count"] = sum(1 for e in chunk["entity_refs"] if e.startswith("course_"))
    
    # For profile chunks, extract faculty info
    if chunk["content_type"] == "profile":
        faculty_entities = [e for e in chunk["entity_refs"] if e.startswith("faculty_")]
        if faculty_entities:
            metadata["faculty_id"] = faculty_entities[0]
    
    return metadata
```

#### Batch Ingestion

```python
from tqdm import tqdm

def ingest_chunks_to_chromadb(collection, chunks, embeddings, batch_size=100):
    """
    Ingest chunks in batches with progress tracking.
    """
    print(f"Ingesting {len(chunks)} chunks into ChromaDB...")
    
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    failed_chunks = []
    
    for i in tqdm(range(0, len(chunks), batch_size), desc="Ingesting batches"):
        batch_chunks = chunks[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]
        
        # Prepare batch data
        ids = [c["chunk_id"] for c in batch_chunks]
        documents = [c["text"] for c in batch_chunks]
        metadatas = [prepare_metadata(c) for c in batch_chunks]
        embeddings_list = batch_embeddings.tolist()
        
        try:
            # Add to ChromaDB
            collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings_list,
                metadatas=metadatas
            )
        except Exception as e:
            print(f"❌ Failed to ingest batch {i//batch_size + 1}: {e}")
            failed_chunks.extend(ids)
    
    # Generate stats
    stats = {
        "total_processed": len(chunks),
        "successfully_ingested": len(chunks) - len(failed_chunks),
        "failed_chunks": failed_chunks,
        "final_collection_count": collection.count()
    }
    
    print(f"✅ Ingestion complete")
    print(f"   Successfully ingested: {stats['successfully_ingested']}")
    print(f"   Failed: {len(failed_chunks)}")
    print(f"   Final collection size: {stats['final_collection_count']}")
    
    return stats
```

---

### 3. Query Routing System

#### Query Classification

```python
import re

def classify_query_type(query):
    """
    Classify query into type for optimized retrieval.
    """
    query_lower = query.lower()
    
    # Faculty queries
    faculty_keywords = ['who is', 'faculty', 'professor', 'hod', 'head of department', 
                       'dr.', 'dr ', 'staff', 'teacher', 'instructor']
    if any(kw in query_lower for kw in faculty_keywords):
        return "faculty"
    
    # Course/Syllabus queries
    course_keywords = ['course', 'subject', 'syllabus', 'credit', 'semester']
    semester_pattern = r'semester\s+\d+|sem\s+\d+|s\d+'
    if any(kw in query_lower for kw in course_keywords) or re.search(semester_pattern, query_lower):
        return "course"
    
    # Timetable queries
    timetable_keywords = ['timetable', 'schedule', 'timing', 'class timing', 'when is']
    if any(kw in query_lower for kw in timetable_keywords):
        return "timetable"
    
    # Regulation queries
    regulation_keywords = ['regulation', 'r2019', 'r2023', 'curriculum', 'scheme', 'grading']
    if any(kw in query_lower for kw in regulation_keywords):
        return "regulation"
    
    # Default
    return "general"
```

#### Routed Query Execution

```python
def query_chromadb(
    collection,
    query_text,
    query_type=None,
    n_results=5,
    content_type_filter=None,
    distance_threshold=None
):
    """
    Execute query with routing logic.
    """
    # Auto-classify if not provided
    if query_type is None:
        query_type = classify_query_type(query_text)
    
    print(f"Query type: {query_type}")
    
    # Routing logic
    if query_type == "faculty":
        # Faculty queries: focus on profiles
        where_filter = {"content_type": "profile"}
        n_results = 3
    
    elif query_type == "course":
        # Course queries: focus on tables
        where_filter = {"content_type": "table"}
        n_results = 10  # More results for tables
    
    elif query_type == "timetable":
        # Timetable queries
        where_filter = {"content_type": "table"}
        n_results = 5
    
    elif query_type == "regulation":
        # Regulation queries
        where_filter = {"content_type": "regulation"}
        n_results = 5
    
    else:
        # General queries: no filter
        where_filter = None
        n_results = 5
    
    # Override with explicit filter if provided
    if content_type_filter:
        where_filter = {"content_type": content_type_filter}
    
    # Execute query
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where_filter if where_filter else None
    )
    
    # Apply adaptive distance threshold
    results = apply_adaptive_distance_threshold(results, query_type)
    
    return results
```

#### Adaptive Distance Thresholding

```python
def apply_adaptive_distance_threshold(results, query_type):
    """
    Filter results based on adaptive distance threshold.
    """
    if not results['distances'] or len(results['distances'][0]) == 0:
        return results
    
    distances = results['distances'][0]
    best_distance = min(distances)
    
    # Determine threshold based on best result quality
    if best_distance < 0.3:
        # Excellent match
        threshold = 0.6
        quality = "excellent"
    elif best_distance < 0.5:
        # Good match
        threshold = 0.7
        quality = "good"
    else:
        # Poor match
        threshold = 0.8
        quality = "poor"
    
    # Filter results
    filtered_indices = [i for i, d in enumerate(distances) if d <= threshold]
    
    # Apply filtering
    filtered_results = {
        'ids': [[results['ids'][0][i] for i in filtered_indices]],
        'distances': [[results['distances'][0][i] for i in filtered_indices]],
        'documents': [[results['documents'][0][i] for i in filtered_indices]],
        'metadatas': [[results['metadatas'][0][i] for i in filtered_indices]],
        'quality': quality,
        'best_distance': best_distance,
        'threshold_used': threshold,
        'original_count': len(distances),
        'filtered_count': len(filtered_indices)
    }
    
    # Warning for poor matches
    if quality == "poor":
        print(f"⚠️  Warning: Best match distance is {best_distance:.3f} (poor quality)")
        print(f"   Consider rephrasing query or checking if information exists in knowledge base")
    
    return filtered_results
```

---

### 4. Validation System

#### Comprehensive Validation

```python
import json
import time

def validate_chromadb_ingestion(collection, expected_stats, test_queries):
    """
    Validate ChromaDB ingestion with comprehensive checks.
    """
    print("\n" + "="*60)
    print("CHROMADB VALIDATION")
    print("="*60)
    
    validation_report = {
        "validation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "collection_health": {},
        "content_type_validation": {},
        "sample_query_tests": [],
        "performance_metrics": {}
    }
    
    # 1. Collection count validation
    actual_count = collection.count()
    expected_count = expected_stats["total_chunks"]
    
    validation_report["collection_health"] = {
        "expected_count": expected_count,
        "actual_count": actual_count,
        "status": "PASS" if actual_count == expected_count else "FAIL"
    }
    
    print(f"\n1. Collection Health Check:")
    print(f"   Expected: {expected_count} chunks")
    print(f"   Actual: {actual_count} chunks")
    print(f"   Status: {validation_report['collection_health']['status']}")
    
    # 2. Content type validation
    print(f"\n2. Content Type Validation:")
    
    for content_type, expected_count in expected_stats["chunks_by_type"].items():
        result = collection.get(where={"content_type": content_type})
        actual_count = len(result['ids'])
        
        status = "PASS" if actual_count == expected_count else "FAIL"
        
        validation_report["content_type_validation"][content_type] = {
            "expected": expected_count,
            "actual": actual_count,
            "status": status
        }
        
        print(f"   {content_type}: {actual_count}/{expected_count} ({status})")
    
    # 3. Sample query tests
    print(f"\n3. Sample Query Tests:")
    
    query_times = []
    
    for test in test_queries:
        start_time = time.time()
        
        results = query_chromadb(
            collection,
            test["query"],
            query_type=test.get("query_type")
        )
        
        query_time_ms = (time.time() - start_time) * 1000
        query_times.append(query_time_ms)
        
        if results['filtered_count'] > 0:
            top_result = results['metadatas'][0][0]
            top_distance = results['distances'][0][0]
            
            test_result = {
                "query": test["query"],
                "expected_content_type": test.get("expected_content_type"),
                "top_result_content_type": top_result.get("content_type"),
                "top_result_distance": round(top_distance, 3),
                "results_count": results['filtered_count'],
                "query_time_ms": round(query_time_ms, 2),
                "status": "PASS" if top_result.get("content_type") == test.get("expected_content_type") else "WARN"
            }
        else:
            test_result = {
                "query": test["query"],
                "status": "FAIL",
                "error": "No results returned"
            }
        
        validation_report["sample_query_tests"].append(test_result)
        
        print(f"   Query: '{test['query']}'")
        print(f"   Top result: {test_result.get('top_result_content_type')} (distance: {test_result.get('top_result_distance')})")
        print(f"   Status: {test_result['status']}")
    
    # 4. Performance metrics
    if query_times:
        validation_report["performance_metrics"] = {
            "average_query_time_ms": round(sum(query_times) / len(query_times), 2),
            "min_query_time_ms": round(min(query_times), 2),
            "max_query_time_ms": round(max(query_times), 2)
        }
        
        print(f"\n4. Performance Metrics:")
        print(f"   Average query time: {validation_report['performance_metrics']['average_query_time_ms']:.2f}ms")
        print(f"   Min query time: {validation_report['performance_metrics']['min_query_time_ms']:.2f}ms")
        print(f"   Max query time: {validation_report['performance_metrics']['max_query_time_ms']:.2f}ms")
    
    print("\n" + "="*60)
    
    return validation_report
```

---

### 5. Main Pipeline

```python
import json
import time
import os

def run_ingestion_pipeline(
    chunks_path="data/chunks/chunks.json",
    chunk_report_path="data/chunks/chunk_report.json",
    force_reembed=False
):
    """
    Main ingestion pipeline orchestrator.
    """
    print("\n" + "="*60)
    print("CHROMADB RAG INGESTION PIPELINE")
    print("="*60)
    
    start_time = time.time()
    
    # ========== STEP 1: Load Data ==========
    print("\n[STEP 1] Loading chunks...")
    
    with open(chunks_path) as f:
        chunks = json.load(f)
    
    with open(chunk_report_path) as f:
        chunk_report = json.load(f)
    
    print(f"✅ Loaded {len(chunks)} chunks")
    print(f"   Distribution: {chunk_report['chunks_by_type']}")
    
    # ========== STEP 2: Check Embedding Cache ==========
    print("\n[STEP 2] Checking embedding cache...")
    
    cache_path = "data/embeddings/embedding_cache.npz"
    cached_ids, cached_embeddings = load_cached_embeddings(cache_path)
    
    if cached_ids and not force_reembed:
        # Verify cache matches current chunks
        current_ids = [c["chunk_id"] for c in chunks]
        
        if set(cached_ids) == set(current_ids):
            print("✅ Using cached embeddings (matched)")
            embeddings = cached_embeddings
        else:
            print("⚠️  Cache mismatch, regenerating embeddings")
            cached_embeddings = None
    else:
        cached_embeddings = None
    
    # ========== STEP 3: Generate Embeddings ==========
    if cached_embeddings is None:
        print("\n[STEP 3] Generating embeddings...")
        
        embed_start = time.time()
        
        # Load model
        model = load_embedding_model("google/embeddinggemma-300m", device="auto")
        
        # Extract texts
        texts = [c["text"] for c in chunks]
        
        # Generate embeddings
        embeddings = generate_embeddings(
            model,
            texts,
            batch_size=64,  # Optimal for GTX 1650 Ti
            show_progress=True
        )
        
        embed_time = time.time() - embed_start
        
        print(f"✅ Embedding generation complete in {embed_time:.2f}s")
        print(f"   Speed: {len(chunks)/embed_time:.1f} chunks/second")
        
        # Cache embeddings
        chunk_ids = [c["chunk_id"] for c in chunks]
        cache_embeddings(chunk_ids, embeddings, cache_path)
    else:
        print("\n[STEP 3] Skipped (using cache)")
        embed_time = 0
    
    # ========== STEP 4: Initialize ChromaDB ==========
    print("\n[STEP 4] Initializing ChromaDB...")
    
    client = initialize_chromadb("./chroma_db")
    collection = create_collection(client, "mbcet_cse_knowledge", recreate=force_reembed)
    
    # ========== STEP 5: Ingest Chunks ==========
    if collection.count() == 0 or force_reembed:
        print("\n[STEP 5] Ingesting chunks into ChromaDB...")
        
        ingest_stats = ingest_chunks_to_chromadb(
            collection,
            chunks,
            embeddings,
            batch_size=100
        )
    else:
        print("\n[STEP 5] Skipped (collection already populated)")
        ingest_stats = {
            "total_processed": len(chunks),
            "successfully_ingested": len(chunks),
            "failed_chunks": [],
            "final_collection_count": collection.count()
        }
    
    # ========== STEP 6: Validation ==========
    print("\n[STEP 6] Validating ChromaDB...")
    
    test_queries = [
        {"query": "Who is the head of CSE department?", "expected_content_type": "profile", "query_type": "faculty"},
        {"query": "What courses are in Semester 3?", "expected_content_type": "table", "query_type": "course"},
        {"query": "R2023 regulations", "expected_content_type": "regulation", "query_type": "regulation"},
        {"query": "Database management systems syllabus", "expected_content_type": "table", "query_type": "course"},
        {"query": "faculty specializing in machine learning", "expected_content_type": "profile", "query_type": "faculty"}
    ]
    
    validation_report = validate_chromadb_ingestion(
        collection,
        chunk_report,
        test_queries
    )
    
    # ========== STEP 7: Generate Reports ==========
    print("\n[STEP 7] Generating reports...")
    
    total_time = time.time() - start_time
    
    # Ingestion report
    ingestion_report = {
        "ingestion_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_chunks_processed": len(chunks),
        "successfully_ingested": ingest_stats["successfully_ingested"],
        "failed_chunks": ingest_stats["failed_chunks"],
        "embedding_stats": {
            "model": "google/embeddinggemma-300m",
            "dimensions": 768,
            "batch_size": 64,
            "total_time_seconds": round(embed_time, 2),
            "chunks_per_second": round(len(chunks)/embed_time, 2) if embed_time > 0 else 0,
            "gpu_used": torch.cuda.is_available()
        },
        "chromadb_stats": {
            "collection_name": "mbcet_cse_knowledge",
            "total_documents": collection.count(),
            "chunks_by_type": chunk_report["chunks_by_type"]
        },
        "total_pipeline_time_seconds": round(total_time, 2)
    }
    
    # Save reports
    os.makedirs("data/validation", exist_ok=True)
    
    with open("data/validation/chromadb_ingestion_report.json", "w") as f:
        json.dump(ingestion_report, f, indent=2)
    
    with open("data/validation/chromadb_validation_report.json", "w") as f:
        json.dump(validation_report, f, indent=2)
    
    print(f"✅ Reports saved to data/validation/")
    
    # ========== STEP 8: Summary ==========
    print("\n" + "="*60)
    print("INGESTION PIPELINE COMPLETE")
    print("="*60)
    print(f"Total chunks: {len(chunks)}")
    print(f"Successfully ingested: {ingest_stats['successfully_ingested']}")
    print(f"Failed: {len(ingest_stats['failed_chunks'])}")
    print(f"Embedding time: {embed_time:.2f}s")
    print(f"Total pipeline time: {total_time:.2f}s")
    print(f"\nChromaDB collection: mbcet_cse_knowledge")
    print(f"Storage location: ./chroma_db/")
    print(f"\nReports:")
    print(f"  - data/validation/chromadb_ingestion_report.json")
    print(f"  - data/validation/chromadb_validation_report.json")
    print("="*60)
```

---

## 🚨 ERROR HANDLING & EDGE CASES

### 1. GPU Out of Memory

```python
def generate_embeddings_with_fallback(model, texts, batch_size=64):
    """
    Try GPU embedding with fallback to CPU on OOM.
    """
    try:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings
    
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("⚠️  GPU OOM detected, retrying with smaller batch size...")
            torch.cuda.empty_cache()
            
            # Retry with half batch size
            return generate_embeddings_with_fallback(model, texts, batch_size=batch_size//2)
        else:
            raise e
```

### 2. Malformed Chunk Handling

```python
def prepare_metadata_safe(chunk):
    """
    Safe metadata preparation with error handling.
    """
    try:
        return prepare_metadata(chunk)
    except Exception as e:
        print(f"⚠️  Error preparing metadata for {chunk.get('chunk_id')}: {e}")
        
        # Return minimal metadata
        return {
            "source_file": chunk.get("source_file", "unknown"),
            "content_type": chunk.get("content_type", "unknown"),
            "error": str(e)
        }
```

### 3. ChromaDB Connection Issues

```python
def initialize_chromadb_safe(persist_directory="./chroma_db", max_retries=3):
    """
    Initialize ChromaDB with retry logic.
    """
    for attempt in range(max_retries):
        try:
            client = chromadb.PersistentClient(path=persist_directory)
            return client
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️  ChromaDB initialization failed (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(2)
            else:
                raise Exception(f"Failed to initialize ChromaDB after {max_retries} attempts: {e}")
```

---

## 📋 INTEGRATION REQUIREMENTS

### CLI Integration

**Must be callable from:**

```bash
# Run full ingestion pipeline
python main.py --stage embed

# Force re-embedding
python main.py --stage embed --force

# Query test
python main.py --stage query --text "Who is the HOD?"
```

### Expected Console Output

```
============================================================
CHROMADB RAG INGESTION PIPELINE
============================================================

[STEP 1] Loading chunks...
✅ Loaded 2060 chunks
   Distribution: {'table': 1829, 'profile': 101, 'regulation': 50, 'section': 69, 'list': 11}

[STEP 2] Checking embedding cache...
⚠️  No embedding cache found

[STEP 3] Generating embeddings...
Loading google/embeddinggemma-300m on cuda...
✅ Model loaded: google/embeddinggemma-300m
   Device: cuda
   Embedding dimensions: 768
Generating embeddings for 2060 chunks...
Batch size: 64
100%|██████████| 2060/2060 [00:28<00:00, 72.3 chunks/s]
✅ Generated 2060 embeddings
   Shape: (2060, 768)
   Dtype: float32
✅ Embedding generation complete in 28.48s
   Speed: 72.3 chunks/second
✅ Embeddings cached to data/embeddings/embedding_cache.npz
   File size: 6.42 MB

[STEP 4] Initializing ChromaDB...
Initializing ChromaDB at ./chroma_db...
✅ ChromaDB initialized
   Storage path: ./chroma_db
✅ Collection ready: mbcet_cse_knowledge
   Current document count: 0

[STEP 5] Ingesting chunks into ChromaDB...
Ingesting 2060 chunks into ChromaDB...
Ingesting batches: 100%|██████████| 21/21 [00:03<00:00,  6.2 batches/s]
✅ Ingestion complete
   Successfully ingested: 2060
   Failed: 0
   Final collection size: 2060

[STEP 6] Validating ChromaDB...

============================================================
CHROMADB VALIDATION
============================================================

1. Collection Health Check:
   Expected: 2060 chunks
   Actual: 2060 chunks
   Status: PASS

2. Content Type Validation:
   table: 1829/1829 (PASS)
   profile: 101/101 (PASS)
   regulation: 50/50 (PASS)
   section: 69/69 (PASS)
   list: 11/11 (PASS)

3. Sample Query Tests:
   Query: 'Who is the head of CSE department?'
   Top result: profile (distance: 0.234)
   Status: PASS
   
   Query: 'What courses are in Semester 3?'
   Top result: table (distance: 0.187)
   Status: PASS
   
   [...]

4. Performance Metrics:
   Average query time: 12.45ms
   Min query time: 8.23ms
   Max query time: 18.91ms

============================================================

[STEP 7] Generating reports...
✅ Reports saved to data/validation/

============================================================
INGESTION PIPELINE COMPLETE
============================================================
Total chunks: 2060
Successfully ingested: 2060
Failed: 0
Embedding time: 28.48s
Total pipeline time: 35.62s

ChromaDB collection: mbcet_cse_knowledge
Storage location: ./chroma_db/

Reports:
  - data/validation/chromadb_ingestion_report.json
  - data/validation/chromadb_validation_report.json
============================================================
```

---

## ✅ SUCCESS CRITERIA

The implementation is correct if:

- ✅ All 2,060 chunks embedded successfully
- ✅ Embedding time < 60 seconds on GPU
- ✅ ChromaDB collection created with correct count
- ✅ All content types validated (table, profile, regulation, section, list)
- ✅ Sample queries return relevant results
- ✅ Average query time < 20ms
- ✅ Validation report shows all PASS
- ✅ Embeddings cached for re-use
- ✅ No crashes on malformed data
- ✅ Reports generated successfully

---

## 📦 DEPENDENCIES

```bash
pip install sentence-transformers torch chromadb numpy tqdm
```

**Verify GPU availability:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

---

## 🧪 TESTING CHECKLIST

Before considering complete:

- [ ] Embeddings generate in <60s on GPU
- [ ] Cache saves and loads correctly
- [ ] ChromaDB collection persists across restarts
- [ ] All 2060 chunks ingested
- [ ] Faculty queries return profile chunks
- [ ] Course queries return table chunks
- [ ] Regulation queries return regulation chunks
- [ ] Distance thresholding filters low-quality results
- [ ] Validation reports generate correctly
- [ ] No errors in logs
- [ ] Works on both GPU and CPU (fallback)

---

**GOOD LUCK! This is production-grade code. Execute methodically and validate each step.** 🚀