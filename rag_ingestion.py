
"""
RAG Ingestion & Retrieval Pipeline for MBCET CSE Knowledge Base.

Embeds semantic chunks using EmbeddingGemma, ingests into ChromaDB,
and provides query routing with adaptive distance thresholding.

Usage:
    python main.py --stage embed           # Run full ingestion pipeline
    python main.py --stage embed --force   # Force re-embedding
    python main.py --stage query --text "Who is the HOD?"
"""

import json
import hashlib
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Fix Windows console encoding for emoji/unicode characters
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
from tqdm import tqdm

import config
from chunker.knowledge_graph import generate_knowledge_graph_documents

# Configure module logger
logger = logging.getLogger(__name__)


# ========================================================================
# EMBEDDING FUNCTIONS
# ========================================================================


def load_embedding_model(
    model_name: str = None,
    device: str = "auto"
):
    """
    Load EmbeddingGemma model with optimal settings for GTX 1650 Ti.

    Args:
        model_name: HuggingFace model identifier. Defaults to config.
        device: 'cuda', 'cpu', or 'auto' (auto-detect GPU).

    Returns:
        SentenceTransformer model instance.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    if model_name is None:
        model_name = config.EMBEDDING_MODEL

    # Auto-detect device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {model_name} on {device}...")

    # Set HF token for gated model access
    hf_token = config.HF_TOKEN
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    model = SentenceTransformer(
        model_name,
        device=device,
        token=hf_token if hf_token else None
    )

    print(f"[OK] Model loaded: {model_name}")
    print(f"   Device: {device}")
    print(f"   Embedding dimensions: {model.get_sentence_embedding_dimension()}")

    # Truncate to 512 tokens for performance on 4GB GPU
    # Default is 2048 which causes extremely slow encoding with long chunks
    model.max_seq_length = 512
    print(f"   Max sequence length: {model.max_seq_length}")

    if device == "cuda":
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {vram_mb:.0f} MB")

    return model


def generate_embeddings(
    model,
    texts: List[str],
    batch_size: int = None,
    show_progress: bool = True
) -> np.ndarray:
    """
    Generate embeddings for all chunks with optimal batching.

    Batch size 64 is optimal for GTX 1650 Ti:
    - Model size: ~1.2GB VRAM
    - Batch (64 chunks): ~400MB VRAM
    - Total: ~1.6GB (safe for 4GB GPU)

    Args:
        model: Loaded SentenceTransformer model.
        texts: List of chunk texts.
        batch_size: Batch size. Defaults to config.
        show_progress: Show progress bar.

    Returns:
        NumPy array of shape (len(texts), 768).
    """
    if batch_size is None:
        batch_size = config.EMBEDDING_BATCH_SIZE

    print(f"Generating embeddings for {len(texts)} chunks...")
    print(f"Batch size: {batch_size}")

    # Normalize text
    texts = [t.strip() for t in texts]

    # Generate embeddings
    # Note: prompt_name="Retrieval-document" omitted intentionally 
    # it prepends a prefix that dramatically increases token length
    # for already-long table chunks, causing ~30x slowdown on 4GB GPU.
    # Queries still use Retrieval-query prompt for asymmetric search.
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalization for cosine similarity
    )

    print(f"[OK] Generated {len(embeddings)} embeddings")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Dtype: {embeddings.dtype}")

    return embeddings


def generate_embeddings_with_fallback(
    model,
    texts: List[str],
    batch_size: int = None
) -> np.ndarray:
    """
    Try GPU embedding with automatic fallback on OOM.

    Recursively halves batch size on CUDA out-of-memory errors.
    Falls back to CPU as last resort.
    """
    import torch

    if batch_size is None:
        batch_size = config.EMBEDDING_BATCH_SIZE

    if batch_size < 1:
        batch_size = 1

    try:
        return generate_embeddings(model, texts, batch_size=batch_size)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()

            if batch_size > 1:
                new_batch = max(batch_size // 2, 1)
                print(f"[WARN]  GPU OOM detected, retrying with batch_size={new_batch}...")
                return generate_embeddings_with_fallback(
                    model, texts, batch_size=new_batch
                )
            else:
                print("[WARN]  GPU OOM even with batch_size=1, falling back to CPU...")
                model = model.to("cpu")
                return generate_embeddings(model, texts, batch_size=8)
        else:
            raise e


def cache_embeddings(
    chunk_ids: List[str],
    embeddings: np.ndarray,
    output_path: str = None
) -> None:
    """
    Cache embeddings to disk for fast re-loading.

    Args:
        chunk_ids: List of chunk IDs.
        embeddings: NumPy array of embeddings.
        output_path: Where to save cache. Defaults to config.
    """
    if output_path is None:
        output_path = str(config.EMBEDDING_CACHE_FILE)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.savez_compressed(
        output_path,
        chunk_ids=np.array(chunk_ids),
        embeddings=embeddings,
        model=config.EMBEDDING_MODEL,
        dimensions=config.EMBEDDING_DIMENSIONS
    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[OK] Embeddings cached to {output_path}")
    print(f"   File size: {file_size_mb:.2f} MB")


def load_cached_embeddings(
    cache_path: str = None
) -> Tuple[Optional[List[str]], Optional[np.ndarray]]:
    """
    Load embeddings from cache if exists.

    Returns:
        (chunk_ids, embeddings) or (None, None) if cache doesn't exist.
    """
    if cache_path is None:
        cache_path = str(config.EMBEDDING_CACHE_FILE)

    if not os.path.exists(cache_path):
        print("[WARN]  No embedding cache found")
        return None, None

    print(f"Loading embeddings from cache: {cache_path}")

    data = np.load(cache_path, allow_pickle=True)
    chunk_ids = data["chunk_ids"].tolist()
    embeddings = data["embeddings"]

    print(f"[OK] Loaded {len(chunk_ids)} cached embeddings")
    print(f"   Shape: {embeddings.shape}")

    return chunk_ids, embeddings


# ========================================================================
# CHROMADB FUNCTIONS
# ========================================================================


def initialize_chromadb(persist_directory: str = None):
    """
    Initialize persistent ChromaDB client.

    Args:
        persist_directory: Where to store ChromaDB data. Defaults to config.

    Returns:
        ChromaDB PersistentClient instance.
    """
    import chromadb

    if persist_directory is None:
        persist_directory = str(config.CHROMADB_DIR)

    print(f"Initializing ChromaDB at {persist_directory}...")

    client = chromadb.PersistentClient(path=persist_directory)

    print(f"[OK] ChromaDB initialized")
    print(f"   Storage path: {persist_directory}")

    return client


def initialize_chromadb_safe(persist_directory: str = None, max_retries: int = 3):
    """
    Initialize ChromaDB with retry logic for connection issues.
    """
    for attempt in range(max_retries):
        try:
            return initialize_chromadb(persist_directory)
        except Exception as e:
            if attempt < max_retries - 1:
                print(
                    f"[WARN]  ChromaDB initialization failed "
                    f"(attempt {attempt + 1}/{max_retries}): {e}"
                )
                time.sleep(2)
            else:
                raise Exception(
                    f"Failed to initialize ChromaDB after {max_retries} attempts: {e}"
                )


def create_collection(client, collection_name: str = None, recreate: bool = False):
    """
    Create or get existing ChromaDB collection.

    Args:
        client: ChromaDB client.
        collection_name: Name of collection. Defaults to config.
        recreate: If True, delete existing and create fresh.

    Returns:
        ChromaDB collection instance.
    """
    if collection_name is None:
        collection_name = config.CHROMADB_COLLECTION

    # Delete existing if recreate=True
    if recreate:
        try:
            client.delete_collection(name=collection_name)
            print(f"  Deleted existing collection: {collection_name}")
        except Exception:
            pass

    # Create collection
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "description": "MBCET CSE Department Knowledge Base",
            "embedding_model": config.EMBEDDING_MODEL,
            "embedding_dimensions": config.EMBEDDING_DIMENSIONS,
            "total_chunks": 2060,
        },
    )

    print(f"[OK] Collection ready: {collection_name}")
    print(f"   Current document count: {collection.count()}")

    return collection


def prepare_metadata(chunk: Dict) -> Dict:
    """
    Extract and prepare structured metadata from a chunk for ChromaDB.

    For table chunks, extracts additional structure info (row count,
    course count). For profile chunks, extracts faculty ID.

    Args:
        chunk: Chunk dictionary from chunks.json.

    Returns:
        Flat metadata dictionary (ChromaDB compatible).
    """
    metadata = {
        "source_file": chunk["source_file"],
        "source_type": chunk["source_type"],
        "content_type": chunk["content_type"],
        "section_hierarchy": " > ".join(chunk["section_hierarchy"]),
        "entity_refs": ",".join(chunk["entity_refs"]) if chunk["entity_refs"] else "",
        "word_count": chunk["word_count"],
    }

    # Include extra relational metadata when available
    for key, value in chunk.get("metadata", {}).items():
        if value is None:
            continue
        if isinstance(value, list):
            metadata[key] = ",".join(str(v) for v in value)
        elif isinstance(value, (str, int, float, bool)):
            metadata[key] = value
        else:
            metadata[key] = str(value)

    # Add page range if available
    if chunk.get("page_range"):
        metadata["page_start"] = chunk["page_range"][0]
        metadata["page_end"] = chunk["page_range"][1]

    # Extract semester if mentioned in section_hierarchy
    for section in chunk["section_hierarchy"]:
        semester_match = re.search(r"Semester\s+(\d+)", section, re.IGNORECASE)
        if semester_match:
            metadata["semester"] = int(semester_match.group(1))
            break

    # For table chunks, extract table-specific metadata
    if chunk["content_type"] == "table":
        metadata["has_table"] = True

        # Count table rows (rough estimate)
        row_count = chunk["text"].count("\n|") - 2  # Subtract header and separator
        metadata["table_row_count"] = max(row_count, 0)

        # Detect if table contains courses
        if any(entity.startswith("course_") for entity in chunk["entity_refs"]):
            metadata["table_contains_courses"] = True
            metadata["table_course_count"] = sum(
                1 for e in chunk["entity_refs"] if e.startswith("course_")
            )

    # For profile chunks, extract faculty info
    if chunk["content_type"] == "profile":
        faculty_entities = [
            e for e in chunk["entity_refs"] if e.startswith("faculty_")
        ]
        if faculty_entities:
            metadata["faculty_id"] = faculty_entities[0]

    return metadata


def prepare_metadata_safe(chunk: Dict) -> Dict:
    """
    Safe metadata preparation with error handling.
    Returns minimal metadata on failure.
    """
    try:
        return prepare_metadata(chunk)
    except Exception as e:
        logger.warning(
            f"Error preparing metadata for {chunk.get('chunk_id')}: {e}"
        )
        return {
            "source_file": chunk.get("source_file", "unknown"),
            "source_type": chunk.get("source_type", "unknown"),
            "content_type": chunk.get("content_type", "unknown"),
            "word_count": chunk.get("word_count", 0),
            "error": str(e),
        }


def ingest_chunks_to_chromadb(
    collection,
    chunks: List[Dict],
    embeddings: np.ndarray,
    batch_size: int = 100,
) -> Dict:
    """
    Ingest chunks with pre-generated embeddings into ChromaDB in batches.

    Args:
        collection: ChromaDB collection.
        chunks: List of chunk dictionaries.
        embeddings: Pre-generated embeddings array.
        batch_size: Batch size for ChromaDB ingestion.

    Returns:
        Ingestion statistics dictionary.
    """
    print(f"Ingesting {len(chunks)} chunks into ChromaDB...")

    failed_chunks = []

    for i in tqdm(range(0, len(chunks), batch_size), desc="Ingesting batches"):
        batch_chunks = chunks[i : i + batch_size]
        batch_embeddings = embeddings[i : i + batch_size]

        # Prepare batch data
        ids = [c["chunk_id"] for c in batch_chunks]
        documents = [c["text"] for c in batch_chunks]
        metadatas = [prepare_metadata_safe(c) for c in batch_chunks]
        embeddings_list = batch_embeddings.tolist()

        try:
            collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings_list,
                metadatas=metadatas,
            )
        except Exception as e:
            print(f"[ERR] Failed to ingest batch {i // batch_size + 1}: {e}")
            failed_chunks.extend(ids)

    # Generate stats
    stats = {
        "total_processed": len(chunks),
        "successfully_ingested": len(chunks) - len(failed_chunks),
        "failed_chunks": failed_chunks,
        "final_collection_count": collection.count(),
    }

    print(f"[OK] Ingestion complete")
    print(f"   Successfully ingested: {stats['successfully_ingested']}")
    print(f"   Failed: {len(failed_chunks)}")
    print(f"   Final collection size: {stats['final_collection_count']}")

    return stats


# ========================================================================
# QUERY ROUTING FUNCTIONS
# ========================================================================


def classify_query_type(query: str) -> str:
    """
    Classify query into type for optimized retrieval routing.

    Args:
        query: User query string.

    Returns:
        Query type: "teaching" | "faculty" | "course" | "timetable" | "regulation" | "general"
    """
    query_lower = query.lower()

    # Teaching-assignment relational queries
    teaching_keywords = [
        "who teaches",
        "who taught",
        "teaches",
        "taught by",
        "instructor for",
        "handles",
        "handled by",
    ]
    if any(kw in query_lower for kw in teaching_keywords):
        return "teaching"

    # Faculty queries
    faculty_keywords = [
        "who is", "faculty", "professor", "hod", "head of department",
        "dr.", "dr ", "staff", "teacher", "instructor",
    ]
    if any(kw in query_lower for kw in faculty_keywords):
        return "faculty"

    # Course/Syllabus queries
    course_keywords = ["course", "subject", "syllabus", "credit", "semester"]
    semester_pattern = r"semester\s+\d+|sem\s+\d+|s\d+"
    if any(kw in query_lower for kw in course_keywords) or re.search(
        semester_pattern, query_lower
    ):
        return "course"

    # Timetable queries
    timetable_keywords = [
        "timetable", "schedule", "timing", "class timing", "when is",
    ]
    if any(kw in query_lower for kw in timetable_keywords):
        return "timetable"

    # Regulation queries
    regulation_keywords = [
        "regulation", "r2019", "r2023", "curriculum", "scheme", "grading",
        "attendance", "cgpa", "sgpa", "credit requirement", "exam rule",
        "supplementary", "probation", "promotion",
    ]
    if any(kw in query_lower for kw in regulation_keywords):
        return "regulation"

    # Default
    return "general"


def apply_adaptive_distance_threshold(results: Dict, query_type: str) -> Dict:
    """
    Filter results based on adaptive distance threshold.

    Logic:
    - best distance < 0.3 (excellent)  threshold = 0.6
    - best distance 0.30.5 (good)  threshold = 0.7
    - best distance > 0.5 (poor)  threshold = 0.8, warn user

    Args:
        results: Raw ChromaDB query results.
        query_type: Classified query type.

    Returns:
        Filtered results with quality metadata.
    """
    if not results["distances"] or len(results["distances"][0]) == 0:
        return {
            **results,
            "quality": "none",
            "best_distance": None,
            "threshold_used": None,
            "original_count": 0,
            "filtered_count": 0,
        }

    distances = results["distances"][0]
    best_distance = min(distances)

    # Determine threshold based on best result quality
    if best_distance < 0.3:
        threshold = 0.6
        quality = "excellent"
    elif best_distance < 0.5:
        threshold = 0.7
        quality = "good"
    else:
        threshold = 0.8
        quality = "poor"

    # Teaching queries are routed to knowledge_graph chunks, which are compact
    # synthetic statements and can have slightly higher distances in practice.
    if query_type == "teaching":
        threshold = max(threshold, 1.2)

    # Faculty profile text is often sparse/noisy; allow a wider threshold
    # to avoid dropping valid profile hits and triggering noisy mixed fallback.
    if query_type == "faculty":
        threshold = max(threshold, 1.35)

    # Filter results
    filtered_indices = [i for i, d in enumerate(distances) if d <= threshold]

    # Apply filtering
    filtered_results = {
        "ids": [[results["ids"][0][i] for i in filtered_indices]],
        "distances": [[results["distances"][0][i] for i in filtered_indices]],
        "documents": [[results["documents"][0][i] for i in filtered_indices]],
        "metadatas": [[results["metadatas"][0][i] for i in filtered_indices]],
        "quality": quality,
        "best_distance": best_distance,
        "threshold_used": threshold,
        "original_count": len(distances),
        "filtered_count": len(filtered_indices),
    }

    # Warning for poor matches
    if quality == "poor":
        print(f"[WARN]  Warning: Best match distance is {best_distance:.3f} (poor quality)")
        print(
            "   Consider rephrasing query or checking if information "
            "exists in knowledge base"
        )

    return filtered_results


def _extract_semester_query_signal(query_text: str) -> Optional[int]:
    match = re.search(r"\b(?:semester|sem|s)\s*([1-8])\b", query_text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _extract_faculty_query_signal(query_text: str) -> Optional[str]:
    try:
        from chunker.entity_registry import EntityRegistry

        registry = EntityRegistry()
        registry.load_all()

        normalized_text = query_text.strip()
        exact = registry.find_exact_match(normalized_text)
        if exact and exact.startswith("faculty_"):
            return exact

        # Try title-based faculty mention extraction.
        candidates = re.findall(
            r"(?:dr|prof|mr|ms|mrs)\.?\s+[a-z]+(?:\s+[a-z]+){0,3}",
            query_text,
            flags=re.IGNORECASE,
        )
        for candidate in candidates:
            exact = registry.find_exact_match(candidate)
            if exact and exact.startswith("faculty_"):
                return exact

            fuzzy = registry.find_fuzzy_match(candidate, entity_type="faculty")
            if fuzzy and fuzzy.startswith("faculty_"):
                return fuzzy
    except Exception:
        return None

    return None


def _rerank_with_query_signals(results: Dict, query_text: str, query_type: str) -> Dict:
    """
    Re-rank routed results using lightweight metadata/query signals.

    This improves precision for faculty and semester-specific queries without
    relying on ChromaDB operator-specific metadata filters.
    """
    if not results.get("distances") or not results["distances"][0]:
        return results

    ids = results["ids"][0]
    distances = results["distances"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    semester_signal = _extract_semester_query_signal(query_text)
    faculty_signal = _extract_faculty_query_signal(query_text) if query_type == "faculty" else None

    scored_indices = []
    for idx, distance in enumerate(distances):
        meta = metas[idx] or {}
        score = float(distance)

        if semester_signal is not None:
            sem_meta = meta.get("semester")
            try:
                if sem_meta is not None and int(sem_meta) == semester_signal:
                    score -= 0.08
            except (ValueError, TypeError):
                pass

        if faculty_signal:
            if meta.get("faculty_id") == faculty_signal:
                score -= 0.12
            else:
                refs = str(meta.get("entity_refs", ""))
                if faculty_signal in refs:
                    score -= 0.08

        scored_indices.append((score, idx))

    ranked = [idx for _, idx in sorted(scored_indices, key=lambda x: x[0])]
    return {
        "ids": [[ids[i] for i in ranked]],
        "distances": [[distances[i] for i in ranked]],
        "documents": [[docs[i] for i in ranked]],
        "metadatas": [[metas[i] for i in ranked]],
    }


def query_chromadb(
    collection,
    query_text: str,
    query_type: str = None,
    n_results: int = 5,
    content_type_filter: str = None,
    distance_threshold: float = None,
    query_embedding: Optional[List[List[float]]] = None,
) -> Dict:
    """
    Query ChromaDB with routing and adaptive filtering.

    Routing Logic:
    - "teaching"  filter content_type="knowledge_graph", n_results=5
    - "faculty"  filter content_type="profile", n_results=3
    - "course"  filter content_type="table", n_results=10
    - "timetable"  filter content_type="table", n_results=5
    - "regulation"  filter content_type="regulation", n_results=5
    - "general"  no filter, n_results=5

    Args:
        collection: ChromaDB collection.
        query_text: User query string.
        query_type: Classified query type (auto-detect if None).
        n_results: Number of results to return.
        content_type_filter: Explicit filter override.
        distance_threshold: Maximum distance (None = adaptive).

    Returns:
        Query results with metadata and quality info.
    """
    # Auto-classify if not provided
    if query_type is None:
        query_type = classify_query_type(query_text)

    print(f"Query type: {query_type}")

    # Routing logic
    where_filter = None

    if query_type == "faculty":
        where_filter = {"content_type": "profile"}
        n_results = 3
    elif query_type == "teaching":
        where_filter = {"content_type": "knowledge_graph"}
        n_results = 5
    elif query_type == "course":
        where_filter = {"content_type": "table"}
        n_results = 10
    elif query_type == "timetable":
        where_filter = {"content_type": "table"}
        n_results = 5
    elif query_type == "regulation":
        where_filter = {"content_type": "regulation"}
        n_results = 5
    else:
        where_filter = None
        n_results = 5

    # Override with explicit filter if provided
    if content_type_filter:
        where_filter = {"content_type": content_type_filter}

    # Generate query embedding if not pre-computed.
    if query_embedding is None:
        model = load_embedding_model(device="auto")
        query_embedding = model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()

        # Free model
        del model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Execute query with pre-computed embedding
    query_kwargs = {
        "query_embeddings": query_embedding,
        "n_results": n_results,
    }
    if where_filter:
        query_kwargs["where"] = where_filter

    results = collection.query(**query_kwargs)

    # Improve ranking precision with query-aware metadata signals before thresholding.
    results = _rerank_with_query_signals(results, query_text, query_type)

    # Apply adaptive distance threshold
    results = apply_adaptive_distance_threshold(results, query_type)

    return results


def _content_type_distribution(results: Dict) -> Dict[str, int]:
    distribution: Dict[str, int] = {}
    metadatas = results.get("metadatas", [[]])
    if not metadatas or not metadatas[0]:
        return distribution

    for meta in metadatas[0]:
        ctype = (meta or {}).get("content_type", "unknown")
        distribution[ctype] = distribution.get(ctype, 0) + 1

    return distribution


def _rerank_mixed_results(results: Dict) -> Dict:
    """
    Re-rank fallback results with a lightweight diversity penalty.

    Keeps retrieval deterministic while discouraging repeated single-type outputs.
    """
    if not results.get("distances") or not results["distances"][0]:
        return results

    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    seen_type_counts: Dict[str, int] = {}
    mixed_scores = []
    diversity_lambda = 0.08

    for idx, distance in enumerate(distances):
        ctype = (metadatas[idx] or {}).get("content_type", "unknown")
        prior = seen_type_counts.get(ctype, 0)
        penalty = diversity_lambda * (prior / max(idx + 1, 1))
        mixed_scores.append(float(distance + penalty))
        seen_type_counts[ctype] = prior + 1

    ranked = sorted(range(len(mixed_scores)), key=lambda i: mixed_scores[i])
    reranked = {
        "ids": [[results["ids"][0][i] for i in ranked]],
        "distances": [[results["distances"][0][i] for i in ranked]],
        "documents": [[results["documents"][0][i] for i in ranked]],
        "metadatas": [[results["metadatas"][0][i] for i in ranked]],
        "quality": results.get("quality"),
        "best_distance": results.get("best_distance"),
        "threshold_used": results.get("threshold_used"),
        "original_count": results.get("original_count", 0),
        "filtered_count": results.get("filtered_count", 0),
        "mixing_scores": [mixed_scores[i] for i in ranked],
    }
    return reranked


def query_chromadb_with_fallback(
    collection,
    query_text: str,
    query_type: str = None,
    n_results: int = 5,
    enable_fallback: bool = True,
    rerank_mixed: bool = True,
) -> Dict:
    """
    Run strict routed retrieval first, then fallback to mixed-content retrieval when needed.

    Fallback triggers only for general route or when strict route returns none/poor quality.
    """
    if query_type is None:
        query_type = classify_query_type(query_text)

    # Compute query embedding once and reuse for both primary and fallback retrieval.
    model = load_embedding_model(device="auto")
    query_embedding = model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()
    del model

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    primary = query_chromadb(
        collection,
        query_text=query_text,
        query_type=query_type,
        n_results=n_results,
        query_embedding=query_embedding,
    )

    primary_count = primary.get("filtered_count", 0)
    primary_quality = primary.get("quality")

    # Keep strict routing for non-general queries unless route is empty.
    # For teaching, allow fallback when route is poor and sparse to avoid false negatives.
    fallback_needed = False
    if enable_fallback:
        if query_type == "general":
            fallback_needed = True
        elif primary_count == 0:
            fallback_needed = True
        elif query_type == "teaching" and primary_quality == "poor" and primary_count < 2:
            fallback_needed = True
        elif query_type == "regulation" and primary_quality == "poor":
            fallback_needed = True

    if not fallback_needed:
        return {
            **primary,
            "fallback_triggered": False,
            "fallback_count": 0,
            "content_type_distribution": _content_type_distribution(primary),
        }

    fallback_raw = collection.query(
        query_embeddings=query_embedding,
        n_results=max(n_results * 2, 8),
    )
    fallback = apply_adaptive_distance_threshold(fallback_raw, "general")

    # Re-apply query-signal reranking on fallback results using original query type.
    # This improves partial-name faculty queries (e.g., "Who is Dr Tessy").
    fallback = _rerank_with_query_signals(fallback, query_text, query_type)

    # If adaptive threshold filters everything, keep top broad candidates
    # to avoid zero-result responses for open-ended general queries.
    if fallback.get("filtered_count", 0) == 0 and fallback_raw.get("distances"):
        keep = min(n_results, len(fallback_raw["distances"][0]))
        fallback = {
            "ids": [fallback_raw["ids"][0][:keep]],
            "distances": [fallback_raw["distances"][0][:keep]],
            "documents": [fallback_raw["documents"][0][:keep]],
            "metadatas": [fallback_raw["metadatas"][0][:keep]],
            "quality": "poor",
            "best_distance": min(fallback_raw["distances"][0]),
            "threshold_used": None,
            "original_count": len(fallback_raw["distances"][0]),
            "filtered_count": keep,
        }

    if rerank_mixed and fallback.get("filtered_count", 0) > 1:
        fallback = _rerank_mixed_results(fallback)

    return {
        "ids": fallback.get("ids", [[]]),
        "distances": fallback.get("distances", [[]]),
        "documents": fallback.get("documents", [[]]),
        "metadatas": fallback.get("metadatas", [[]]),
        "quality": fallback.get("quality", "none"),
        "best_distance": fallback.get("best_distance"),
        "threshold_used": fallback.get("threshold_used"),
        "original_count": primary.get("filtered_count", 0),
        "filtered_count": fallback.get("filtered_count", 0),
        "fallback_triggered": True,
        "fallback_count": fallback.get("filtered_count", 0),
        "content_type_distribution": _content_type_distribution(fallback),
    }


# ========================================================================
# VALIDATION FUNCTIONS
# ========================================================================


def validate_chromadb_ingestion(
    collection,
    expected_stats: Dict,
    test_queries: List[Dict],
) -> Dict:
    """
    Comprehensive validation of ChromaDB ingestion.

    Checks:
    1. Collection count matches expected
    2. Content type distribution matches
    3. Sample queries return relevant results
    4. Performance metrics

    Args:
        collection: ChromaDB collection to validate.
        expected_stats: Expected counts from chunk_report.json.
        test_queries: List of test query dictionaries.

    Returns:
        Validation report dictionary.
    """
    print("\n" + "=" * 60)
    print("CHROMADB VALIDATION")
    print("=" * 60)

    validation_report = {
        "validation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "collection_health": {},
        "content_type_validation": {},
        "sample_query_tests": [],
        "performance_metrics": {},
    }

    # 1. Collection count validation
    actual_count = collection.count()
    expected_count = expected_stats["total_chunks"]

    validation_report["collection_health"] = {
        "expected_count": expected_count,
        "actual_count": actual_count,
        "status": "PASS" if actual_count == expected_count else "FAIL",
    }

    print(f"\n1. Collection Health Check:")
    print(f"   Expected: {expected_count} chunks")
    print(f"   Actual: {actual_count} chunks")
    print(f"   Status: {validation_report['collection_health']['status']}")

    # 2. Content type validation
    print(f"\n2. Content Type Validation:")

    for content_type, expected_ct_count in expected_stats["chunks_by_type"].items():
        result = collection.get(where={"content_type": content_type})
        actual_ct_count = len(result["ids"])

        status = "PASS" if actual_ct_count == expected_ct_count else "FAIL"

        validation_report["content_type_validation"][content_type] = {
            "expected": expected_ct_count,
            "actual": actual_ct_count,
            "status": status,
        }

        print(f"   {content_type}: {actual_ct_count}/{expected_ct_count} ({status})")

    # 3. Sample query tests
    print(f"\n3. Sample Query Tests:")

    query_times = []

    for test in test_queries:
        start_time = time.time()

        results = query_chromadb(
            collection,
            test["query"],
            query_type=test.get("query_type"),
        )

        query_time_ms = (time.time() - start_time) * 1000
        query_times.append(query_time_ms)

        if results["filtered_count"] > 0:
            top_result = results["metadatas"][0][0]
            top_distance = results["distances"][0][0]

            test_result = {
                "query": test["query"],
                "expected_content_type": test.get("expected_content_type"),
                "top_result_content_type": top_result.get("content_type"),
                "top_result_distance": round(top_distance, 3),
                "results_count": results["filtered_count"],
                "query_time_ms": round(query_time_ms, 2),
                "status": (
                    "PASS"
                    if top_result.get("content_type")
                    == test.get("expected_content_type")
                    else "WARN"
                ),
            }
        else:
            test_result = {
                "query": test["query"],
                "status": "FAIL",
                "error": "No results returned",
            }

        validation_report["sample_query_tests"].append(test_result)

        print(f"   Query: '{test['query']}'")
        print(
            f"   Top result: {test_result.get('top_result_content_type')} "
            f"(distance: {test_result.get('top_result_distance')})"
        )
        print(f"   Status: {test_result['status']}")

    # 4. Performance metrics
    if query_times:
        validation_report["performance_metrics"] = {
            "average_query_time_ms": round(
                sum(query_times) / len(query_times), 2
            ),
            "min_query_time_ms": round(min(query_times), 2),
            "max_query_time_ms": round(max(query_times), 2),
        }

        print(f"\n4. Performance Metrics:")
        avg = validation_report["performance_metrics"]["average_query_time_ms"]
        mn = validation_report["performance_metrics"]["min_query_time_ms"]
        mx = validation_report["performance_metrics"]["max_query_time_ms"]
        print(f"   Average query time: {avg:.2f}ms")
        print(f"   Min query time: {mn:.2f}ms")
        print(f"   Max query time: {mx:.2f}ms")

    print("\n" + "=" * 60)

    return validation_report


# ========================================================================
# MAIN PIPELINE
# ========================================================================


def append_knowledge_graph_chunks(
    chunks: List[Dict],
    chunk_report: Dict,
    data_dir: Path,
) -> Tuple[List[Dict], Dict, int]:
    """
    Append synthetic knowledge-graph chunks derived from canonical graph docs.

    Returns:
        (updated_chunks, updated_chunk_report, added_count)
    """
    kg_documents = generate_knowledge_graph_documents(data_dir)
    kg_chunks: List[Dict] = []

    for doc in kg_documents:
        metadata = doc.get("metadata", {})
        faculty_id = metadata.get("faculty_id")
        course_ids = metadata.get("course_ids", [])
        entity_refs = []
        if isinstance(faculty_id, str) and faculty_id:
            entity_refs.append(faculty_id)
        if isinstance(course_ids, list):
            entity_refs.extend([cid for cid in course_ids if isinstance(cid, str)])

        text = doc.get("text", "").strip()
        if not text:
            continue

        chunk_id = doc.get("id") or (
            "kg_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        )

        kg_chunks.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "source_type": "synthetic",
                "source_file": metadata.get(
                    "source_file", "data/entities/teaching_assignments.json"
                ),
                "section_hierarchy": ["Knowledge Graph", "Teaching Assignments"],
                "content_type": "knowledge_graph",
                "entity_refs": sorted(set(entity_refs)),
                "page_range": None,
                "word_count": len(text.split()),
                "hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "metadata": metadata,
            }
        )

    existing_ids = {c["chunk_id"] for c in chunks}
    unique_kg_chunks = [c for c in kg_chunks if c["chunk_id"] not in existing_ids]
    chunks.extend(unique_kg_chunks)

    chunk_report["total_chunks"] = chunk_report.get("total_chunks", 0) + len(unique_kg_chunks)
    chunk_report.setdefault("chunks_by_type", {})
    chunk_report["chunks_by_type"]["knowledge_graph"] = (
        chunk_report["chunks_by_type"].get("knowledge_graph", 0) + len(unique_kg_chunks)
    )

    return chunks, chunk_report, len(unique_kg_chunks)


def run_ingestion_pipeline(
    chunks_path: str = None,
    chunk_report_path: str = None,
    force_reembed: bool = False,
) -> None:
    """
    Main ingestion pipeline orchestrator.

    Steps:
    1. Load chunks from JSON
    2. Check for cached embeddings
    3. Generate embeddings (if needed)
    4. Initialize ChromaDB
    5. Create collection
    6. Ingest chunks with metadata
    7. Validate ingestion
    8. Generate reports + summary

    Args:
        chunks_path: Path to chunks.json. Defaults to config.
        chunk_report_path: Path to chunk_report.json. Defaults to config.
        force_reembed: Force re-embedding even if cache exists.
    """
    import torch

    if chunks_path is None:
        chunks_path = str(config.CHUNKS_FILE)
    if chunk_report_path is None:
        chunk_report_path = str(config.CHUNK_REPORT_FILE)

    print("\n" + "=" * 60)
    print("CHROMADB RAG INGESTION PIPELINE")
    print("=" * 60)

    start_time = time.time()

    # ========== STEP 1: Load Data ==========
    print("\n[STEP 1] Loading chunks...")

    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    with open(chunk_report_path, encoding="utf-8") as f:
        chunk_report = json.load(f)

    chunks, chunk_report, kg_added = append_knowledge_graph_chunks(
        chunks, chunk_report, config.DATA_DIR
    )
    if kg_added:
        print(f"[OK] Added {kg_added} synthetic knowledge-graph chunks")

    print(f"[OK] Loaded {len(chunks)} chunks")
    print(f"   Distribution: {chunk_report['chunks_by_type']}")

    # ========== STEP 2: Check Embedding Cache ==========
    print("\n[STEP 2] Checking embedding cache...")

    cache_path = str(config.EMBEDDING_CACHE_FILE)
    cached_ids, cached_embeddings = load_cached_embeddings(cache_path)

    embeddings = None

    if cached_ids is not None and not force_reembed:
        # Verify cache matches current chunks
        current_ids = [c["chunk_id"] for c in chunks]

        if set(cached_ids) == set(current_ids):
            print("[OK] Using cached embeddings (matched)")
            # Reorder embeddings to match current chunk order
            id_to_idx = {cid: idx for idx, cid in enumerate(cached_ids)}
            reorder = [id_to_idx[cid] for cid in current_ids]
            embeddings = cached_embeddings[reorder]
        else:
            print("[WARN]  Cache mismatch, regenerating embeddings")

    # ========== STEP 3: Generate Embeddings ==========
    if embeddings is None:
        print("\n[STEP 3] Generating embeddings...")

        embed_start = time.time()

        # Load model
        model = load_embedding_model(config.EMBEDDING_MODEL, device="auto")

        # Extract texts
        texts = [c["text"] for c in chunks]

        # Generate embeddings with OOM fallback
        embeddings = generate_embeddings_with_fallback(
            model,
            texts,
            batch_size=config.EMBEDDING_BATCH_SIZE,
        )

        embed_time = time.time() - embed_start

        print(f"[OK] Embedding generation complete in {embed_time:.2f}s")
        print(f"   Speed: {len(chunks) / embed_time:.1f} chunks/second")

        # Cache embeddings
        chunk_ids = [c["chunk_id"] for c in chunks]
        cache_embeddings(chunk_ids, embeddings, cache_path)

        # Free GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("\n[STEP 3] Skipped (using cache)")
        embed_time = 0

    # ========== STEP 4: Initialize ChromaDB ==========
    print("\n[STEP 4] Initializing ChromaDB...")

    client = initialize_chromadb_safe(str(config.CHROMADB_DIR))
    collection = create_collection(
        client, config.CHROMADB_COLLECTION, recreate=force_reembed
    )

    # ========== STEP 5: Ingest Chunks ==========
    if collection.count() == 0 or force_reembed:
        print("\n[STEP 5] Ingesting chunks into ChromaDB...")

        ingest_stats = ingest_chunks_to_chromadb(
            collection, chunks, embeddings, batch_size=100
        )
    else:
        print("\n[STEP 5] Skipped (collection already populated)")
        ingest_stats = {
            "total_processed": len(chunks),
            "successfully_ingested": len(chunks),
            "failed_chunks": [],
            "final_collection_count": collection.count(),
        }

    # ========== STEP 6: Validation ==========
    print("\n[STEP 6] Validating ChromaDB...")

    test_queries = [
        {
            "query": "Who is the head of CSE department?",
            "expected_content_type": "profile",
            "query_type": "faculty",
        },
        {
            "query": "What courses are in Semester 3?",
            "expected_content_type": "table",
            "query_type": "course",
        },
        {
            "query": "R2023 regulations",
            "expected_content_type": "regulation",
            "query_type": "regulation",
        },
        {
            "query": "Database management systems syllabus",
            "expected_content_type": "table",
            "query_type": "course",
        },
        {
            "query": "faculty specializing in machine learning",
            "expected_content_type": "profile",
            "query_type": "faculty",
        },
        {
            "query": "Who teaches Artificial Intelligence?",
            "expected_content_type": "knowledge_graph",
            "query_type": "teaching",
        },
    ]

    validation_report = validate_chromadb_ingestion(
        collection, chunk_report, test_queries
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
            "model": config.EMBEDDING_MODEL,
            "dimensions": config.EMBEDDING_DIMENSIONS,
            "batch_size": config.EMBEDDING_BATCH_SIZE,
            "total_time_seconds": round(embed_time, 2),
            "chunks_per_second": (
                round(len(chunks) / embed_time, 2) if embed_time > 0 else 0
            ),
            "gpu_used": torch.cuda.is_available(),
        },
        "chromadb_stats": {
            "collection_name": config.CHROMADB_COLLECTION,
            "total_documents": collection.count(),
            "chunks_by_type": chunk_report["chunks_by_type"],
        },
        "total_pipeline_time_seconds": round(total_time, 2),
    }

    # Save reports
    os.makedirs(str(config.VALIDATION_DIR), exist_ok=True)

    with open(str(config.INGESTION_REPORT_FILE), "w", encoding="utf-8") as f:
        json.dump(ingestion_report, f, indent=2)

    with open(str(config.VALIDATION_REPORT_FILE), "w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2)

    print(f"[OK] Reports saved to {config.VALIDATION_DIR}")

    # ========== STEP 8: Summary ==========
    print("\n" + "=" * 60)
    print("INGESTION PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total chunks: {len(chunks)}")
    print(f"Successfully ingested: {ingest_stats['successfully_ingested']}")
    print(f"Failed: {len(ingest_stats['failed_chunks'])}")
    print(f"Embedding time: {embed_time:.2f}s")
    print(f"Total pipeline time: {total_time:.2f}s")
    print(f"\nChromaDB collection: {config.CHROMADB_COLLECTION}")
    print(f"Storage location: {config.CHROMADB_DIR}")
    print(f"\nReports:")
    print(f"  - {config.INGESTION_REPORT_FILE}")
    print(f"  - {config.VALIDATION_REPORT_FILE}")
    print("=" * 60)


# ========================================================================
# STANDALONE QUERY FUNCTION
# ========================================================================


def run_query(query_text: str) -> None:
    """
    Run a standalone query against the ChromaDB collection.

    Args:
        query_text: The user query to execute.
    """
    print("\n" + "=" * 60)
    print("CHROMADB QUERY")
    print("=" * 60)

    # Initialize ChromaDB
    client = initialize_chromadb_safe(str(config.CHROMADB_DIR))

    try:
        collection = client.get_collection(name=config.CHROMADB_COLLECTION)
    except Exception:
        print("[ERR] Collection not found. Run ingestion first:")
        print("   python main.py --stage embed")
        return

    print(f"Collection: {config.CHROMADB_COLLECTION}")
    print(f"Documents: {collection.count()}")

    # Classify and query
    query_type = classify_query_type(query_text)
    print(f"\nQuery: '{query_text}'")
    print(f"Classified as: {query_type}")

    results = query_chromadb_with_fallback(
        collection,
        query_text,
        query_type=query_type,
        enable_fallback=True,
        rerank_mixed=True,
    )

    # Display results
    print(f"\n--- Results ({results['filtered_count']} matches) ---")
    print(f"Quality: {results['quality']}")
    print(f"Best distance: {results['best_distance']}")
    print(f"Threshold used: {results['threshold_used']}")
    print(f"Fallback triggered: {results.get('fallback_triggered', False)}")
    if results.get("fallback_triggered"):
        print(f"Fallback matches: {results.get('fallback_count', 0)}")
        print(f"Type mix: {results.get('content_type_distribution', {})}")

    if results["filtered_count"] > 0:
        for i in range(results["filtered_count"]):
            meta = results["metadatas"][0][i]
            dist = results["distances"][0][i]
            doc = results["documents"][0][i]

            print(f"\n--- Result {i + 1} (distance: {dist:.3f}) ---")
            print(f"Type: {meta.get('content_type')}")
            print(f"Source: {meta.get('source_file')}")
            print(f"Hierarchy: {meta.get('section_hierarchy')}")
            # Show first 300 chars of content
            preview = doc[:300] + "..." if len(doc) > 300 else doc
            print(f"Content:\n{preview}")
    else:
        print("No relevant results found.")

    print("\n" + "=" * 60)
