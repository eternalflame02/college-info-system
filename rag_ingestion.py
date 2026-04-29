
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
from typing import Any, Dict, List, Optional, Tuple

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

# Process-local cache for entity registry to avoid repeated reloads
# during a single query lifecycle.
_ENTITY_REGISTRY_CACHE: Dict[str, Any] = {
    "registry": None,
    "registry_cls": None,
}

_QUALITY_BANDS_BASE: Dict[str, float] = {
    "excellent": 0.75,
    "good": 1.00,
    "fair": 1.20,
}

_QUALITY_TYPE_ADJUST: Dict[str, float] = {
    "teaching": 0.00,
    "advisory": 0.05,
    "faculty": 0.10,
    "regulation": 0.15,
}

_QUALITY_MARGIN_BY_LABEL: Dict[str, float] = {
    "excellent": 0.18,
    "good": 0.30,
    "fair": 0.32,
    "poor": 0.33,
}

_THRESHOLD_CAP_BY_TYPE: Dict[str, float] = {
    "general": 1.25,
    "advisory": 1.40,
    "course": 1.35,
    "timetable": 1.35,
    "teaching": 1.35,
    "faculty": 1.45,
    "regulation": 1.55,
}

_SYLLABUS_SOURCE_HINTS = (
    "syllabus",
    "curriculum",
    "autonomy",
    "s5s6",
    "s7s8",
)


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
        has_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
        device = "cuda" if has_cuda else "cpu"

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

    if device == "cuda" and torch.cuda.device_count() > 0:
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

    # Prefer model-provided document encoding (EmbeddingGemma optimized).
    # Fall back to prompt_name-based encode for compatibility.
    if config.EMBEDDING_USE_QUERY_DOC_METHODS and hasattr(model, "encode_document"):
        embeddings = model.encode_document(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    else:
        embeddings = model.encode(
            texts,
            prompt_name=config.EMBEDDING_DOCUMENT_PROMPT_NAME,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
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
            "hnsw:space": "cosine",
            "total_chunks": 2060,
        },
    )

    print(f"[OK] Collection ready: {collection_name}")
    print(f"   Current document count: {collection.count()}")

    return collection


def get_rag_collections(client, recreate: bool = False, create_missing: bool = False) -> Dict[str, Any]:
    """
    Build collection handles for retrieval/ingestion.

    Keys:
    - table
    - non_table
    - legacy
    """
    collections: Dict[str, Any] = {}

    def _get_or_create(name: str):
        if recreate:
            try:
                client.delete_collection(name=name)
                print(f"  Deleted existing collection: {name}")
            except Exception:
                pass

        if create_missing:
            return client.get_or_create_collection(
                name=name,
                metadata={
                    "description": "MBCET CSE Department Knowledge Base",
                    "embedding_model": config.EMBEDDING_MODEL,
                    "embedding_dimensions": config.EMBEDDING_DIMENSIONS,
                    "hnsw:space": "cosine",
                },
            )
        return client.get_collection(name=name)

    if config.CHROMADB_MULTI_COLLECTION_ENABLED:
        try:
            collections["table"] = _get_or_create(config.CHROMADB_TABLE_COLLECTION)
        except Exception:
            pass
        try:
            collections["non_table"] = _get_or_create(config.CHROMADB_NON_TABLE_COLLECTION)
        except Exception:
            pass

    if config.CHROMADB_ENABLE_LEGACY_FALLBACK or not collections:
        try:
            target_name = config.CHROMADB_COLLECTION
            if recreate and create_missing:
                collections["legacy"] = create_collection(client, target_name, recreate=True)
            elif create_missing:
                collections["legacy"] = create_collection(client, target_name, recreate=False)
            else:
                collections["legacy"] = client.get_collection(name=target_name)
        except Exception:
            pass

    return collections


def _split_chunks_for_collections(chunks: List[Dict], embeddings: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """Split chunk/embedding streams by collection family."""
    table_indices = [i for i, chunk in enumerate(chunks) if chunk.get("content_type") == "table"]
    non_table_indices = [i for i, chunk in enumerate(chunks) if chunk.get("content_type") != "table"]

    def _select(indices: List[int]) -> Tuple[List[Dict], np.ndarray]:
        if not indices:
            return [], np.zeros((0, embeddings.shape[1]), dtype=embeddings.dtype)
        selected_chunks = [chunks[i] for i in indices]
        selected_embeddings = embeddings[indices]
        return selected_chunks, selected_embeddings

    table_chunks, table_embeddings = _select(table_indices)
    non_table_chunks, non_table_embeddings = _select(non_table_indices)

    return {
        "table": {
            "chunks": table_chunks,
            "embeddings": table_embeddings,
        },
        "non_table": {
            "chunks": non_table_chunks,
            "embeddings": non_table_embeddings,
        },
        "legacy": {
            "chunks": chunks,
            "embeddings": embeddings,
        },
    }


def _resolve_collection_route(collection_or_map: Any, query_type: str) -> Tuple[Any, List[Any], str]:
    """Resolve primary + fallback collections for a query type."""
    if not isinstance(collection_or_map, dict):
        return collection_or_map, [], "legacy"

    collection_map = {k: v for k, v in collection_or_map.items() if v is not None}
    if not collection_map:
        raise ValueError("No ChromaDB collections available for retrieval")

    preferred_primary_key = "non_table"
    if query_type in {"course", "timetable", "advisory"}:
        preferred_primary_key = "table"
    elif query_type in {"faculty", "regulation", "general", "teaching"}:
        preferred_primary_key = "non_table"

    if preferred_primary_key not in collection_map:
        if "legacy" in collection_map:
            preferred_primary_key = "legacy"
        else:
            preferred_primary_key = next(iter(collection_map.keys()))

    primary = collection_map[preferred_primary_key]
    fallback = [
        coll
        for key, coll in collection_map.items()
        if key != preferred_primary_key
    ]

    return primary, fallback, preferred_primary_key


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

    # Extract faculty info from entity refs for all chunk types.
    # Faculty pages are often represented as markdown tables, not "profile" chunks.
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
            collection.upsert(
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
        Query type: "teaching" | "faculty" | "advisory" | "course" | "timetable" | "regulation" | "general"
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

    # Student advisory/recommendation queries
    advisory_keywords = [
        "elective", "minor", "which one should", "which should i pick",
        "recommend", "best for", "interested in", "planning to move",
        "theory", "application", "practical", "hands-on", "common subjects",
        "common course", "career", "track",
    ]
    if any(kw in query_lower for kw in advisory_keywords):
        return "advisory"

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
    - Determine quality band from corpus-calibrated distance buckets.
    - Build threshold as best_distance + quality-specific margin.
    - Apply query-type caps to keep broad fallbacks bounded.

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

    type_adjust = _QUALITY_TYPE_ADJUST.get(query_type, 0.0)
    excellent_cut = _QUALITY_BANDS_BASE["excellent"] + type_adjust
    good_cut = _QUALITY_BANDS_BASE["good"] + type_adjust
    fair_cut = _QUALITY_BANDS_BASE["fair"] + type_adjust

    if best_distance < excellent_cut:
        quality = "excellent"
    elif best_distance < good_cut:
        quality = "good"
    elif best_distance < fair_cut:
        quality = "fair"
    else:
        quality = "poor"

    margin = _QUALITY_MARGIN_BY_LABEL[quality]
    cap = _THRESHOLD_CAP_BY_TYPE.get(query_type, _THRESHOLD_CAP_BY_TYPE["general"])
    threshold = min(best_distance + margin, cap)

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


def _is_syllabus_like_query(query_text: str) -> bool:
    query_lower = query_text.lower()
    syllabus_markers = [
        "syllabus",
        "module",
        "course outcome",
        "course outcomes",
        "co ",
        "curriculum",
    ]
    return any(marker in query_lower for marker in syllabus_markers)


def _extract_course_code_query_signal(query_text: str) -> List[str]:
    """Extract likely course codes from query text for metadata matching."""
    matches = re.findall(
        r"\b(?:\d{2}[A-Z]{2,4}[A-Z]?\d{2}[A-Z]?|[A-Z]{2,4}\d[A-Z]\d{2}[A-Z]?)\b",
        query_text.upper(),
    )
    return sorted(set(matches))


def _get_entity_registry_cached():
    """Return a loaded EntityRegistry instance with lightweight process caching."""
    from chunker.entity_registry import EntityRegistry

    cached_registry = _ENTITY_REGISTRY_CACHE.get("registry")
    cached_cls = _ENTITY_REGISTRY_CACHE.get("registry_cls")

    # Reuse only when the underlying class identity is unchanged.
    # This keeps tests with monkeypatched registries deterministic.
    if cached_registry is not None and cached_cls is EntityRegistry:
        return cached_registry

    registry = EntityRegistry()
    registry.load_all()
    _ENTITY_REGISTRY_CACHE["registry"] = registry
    _ENTITY_REGISTRY_CACHE["registry_cls"] = EntityRegistry
    return registry


def _extract_faculty_query_signal(query_text: str) -> Optional[str]:
    try:
        registry = _get_entity_registry_cached()

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

            # Partial-name backoff (for queries like "Who is Dr Tessy?"):
            # match candidate tokens against known normalized aliases.
            candidate_norm = re.sub(r"\s+", " ", candidate.strip().lower())
            candidate_norm = re.sub(r"\b(dr|prof|mr|ms|mrs)\.?\s+", "", candidate_norm)
            tokens = [tok for tok in candidate_norm.split() if tok]
            if not tokens:
                continue

            alias_lookup = getattr(registry, "lookup", {}) or {}
            matches = []
            for alias_norm, entity_id in alias_lookup.items():
                if not str(entity_id).startswith("faculty_"):
                    continue
                alias_text = str(alias_norm)
                if all(tok in alias_text for tok in tokens):
                    matches.append(str(entity_id))
            if matches:
                return sorted(set(matches))[0]
    except Exception:
        return None

    return None


def _rerank_with_query_signals(
    results: Dict,
    query_text: str,
    query_type: str,
    faculty_signal: Optional[str] = None,
) -> Dict:
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
    syllabus_query = _is_syllabus_like_query(query_text)
    query_codes = _extract_course_code_query_signal(query_text)
    if query_type == "faculty" and faculty_signal is None:
        faculty_signal = _extract_faculty_query_signal(query_text)

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

            source_file = str(meta.get("source_file", "")).lower()
            section_hierarchy = str(meta.get("section_hierarchy", "")).lower()
            if any(token in source_file for token in ["workshop", "seminar", "activities"]):
                score += 0.15
            if any(token in section_hierarchy for token in ["workshop", "seminar"]):
                score += 0.12
            if any(token in source_file for token in ["faculty_", "/faculty/", "faculty-"]):
                score -= 0.12

        if query_type == "regulation":
            if str(meta.get("content_type", "")).lower() == "regulation":
                score -= 0.12

        if query_type == "course" and syllabus_query:
            source_file = str(meta.get("source_file", "")).lower()
            source_type = str(meta.get("source_type", "")).lower()
            section_hierarchy = str(meta.get("section_hierarchy", "")).lower()
            entity_refs = str(meta.get("entity_refs", "")).lower()
            timetable_codes = str(meta.get("timetable_course_codes", "")).upper()

            if source_type == "pdf":
                score -= 0.08

            if any(token in source_file for token in _SYLLABUS_SOURCE_HINTS):
                score -= 0.14

            if any(token in section_hierarchy for token in ["module", "course outcomes", "course outcome"]):
                score -= 0.05

            if any(token in source_file for token in ["activities", "workshop", "seminar", "blog"]):
                score += 0.22

            if query_codes:
                if any(code in timetable_codes for code in query_codes):
                    score -= 0.10
                elif any(f"course_{code.lower()}" in entity_refs for code in query_codes):
                    score -= 0.08
                else:
                    score += 0.03

        scored_indices.append((score, idx))

    ranked = [idx for _, idx in sorted(scored_indices, key=lambda x: x[0])]
    reranked = {
        "ids": [[ids[i] for i in ranked]],
        "distances": [[distances[i] for i in ranked]],
        "documents": [[docs[i] for i in ranked]],
        "metadatas": [[metas[i] for i in ranked]],
    }
    for passthrough_key in [
        "quality",
        "best_distance",
        "threshold_used",
        "original_count",
        "filtered_count",
    ]:
        if passthrough_key in results:
            reranked[passthrough_key] = results[passthrough_key]
    return reranked


def _tag_results_collection(results: Dict, collection_key: str) -> Dict:
    """Attach retrieval collection key to each metadata record."""
    tagged = {
        "ids": [list(results.get("ids", [[]])[0])],
        "distances": [list(results.get("distances", [[]])[0])],
        "documents": [list(results.get("documents", [[]])[0])],
        "metadatas": [[]],
    }
    metas = results.get("metadatas", [[]])
    meta_items = metas[0] if metas and metas[0] else []
    for meta in meta_items:
        meta_copy = dict(meta or {})
        meta_copy["retrieval_collection"] = collection_key
        tagged["metadatas"][0].append(meta_copy)
    for passthrough_key in [
        "quality",
        "best_distance",
        "threshold_used",
        "original_count",
        "filtered_count",
    ]:
        if passthrough_key in results:
            tagged[passthrough_key] = results[passthrough_key]
    return tagged


def _merge_raw_results(results_list: List[Dict], max_items: int) -> Dict:
    """Merge raw Chroma-style results from multiple collections by best distance."""
    merged: Dict[str, Dict[str, Any]] = {}

    for res in results_list:
        ids = res.get("ids", [[]])[0] if res.get("ids") else []
        distances = res.get("distances", [[]])[0] if res.get("distances") else []
        docs = res.get("documents", [[]])[0] if res.get("documents") else []
        metas = res.get("metadatas", [[]])[0] if res.get("metadatas") else []

        for idx, chunk_id in enumerate(ids):
            distance = float(distances[idx])
            if chunk_id in merged and merged[chunk_id]["distance"] <= distance:
                continue
            merged[chunk_id] = {
                "distance": distance,
                "document": docs[idx],
                "metadata": metas[idx] if idx < len(metas) else {},
            }

    ranked = sorted(merged.items(), key=lambda item: item[1]["distance"])[:max_items]
    return {
        "ids": [[chunk_id for chunk_id, _ in ranked]],
        "distances": [[payload["distance"] for _, payload in ranked]],
        "documents": [[payload["document"] for _, payload in ranked]],
        "metadatas": [[payload["metadata"] for _, payload in ranked]],
    }


def _filter_faculty_linked_candidates(
    results: Dict,
    query_text: str,
    faculty_signal: Optional[str] = None,
) -> Dict:
    """Prefer faculty-linked fallback candidates when a faculty name signal exists."""
    if faculty_signal is None:
        faculty_signal = _extract_faculty_query_signal(query_text)
    if not faculty_signal:
        return results

    ids = results.get("ids", [[]])
    if not ids or not ids[0]:
        return results

    keep_indices: List[int] = []
    metas = results.get("metadatas", [[]])[0]
    docs = results.get("documents", [[]])[0]
    for idx, meta in enumerate(metas):
        meta_obj = meta or {}
        refs = [
            token.strip()
            for token in str(meta_obj.get("entity_refs", "")).split(",")
            if token.strip().startswith("faculty_")
        ]
        is_signal_match = meta_obj.get("faculty_id") == faculty_signal or faculty_signal in refs
        if is_signal_match:
            source_file = str(meta_obj.get("source_file", "")).lower()
            is_dedicated_faculty_page = any(
                token in source_file for token in ["faculty_", "/faculty/", "faculty-"]
            )
            if any(token in source_file for token in ["workshop", "seminar", "activities"]) and not is_dedicated_faculty_page:
                continue
            # Reject aggregate chunks (workshops/seminars) that mention many faculty.
            if len(set(refs)) > 1 and not is_dedicated_faculty_page:
                continue
            keep_indices.append(idx)

    if not keep_indices:
        for idx, meta in enumerate(metas):
            meta_obj = meta or {}
            source_file = str(meta_obj.get("source_file", "")).lower()
            if "frequently" in source_file:
                keep_indices.append(idx)
                continue
            if not any(token in source_file for token in ["faculty_", "/faculty/", "faculty-"]):
                continue
            doc_text = str(docs[idx]).lower() if idx < len(docs) else ""
            if re.search(r"\b(dr|prof|mr|ms|mrs)\.?\s+[a-z]+", doc_text):
                keep_indices.append(idx)

    # Always preserve FAQ chunks that were matched semantically
    for idx, meta in enumerate(metas):
        meta_obj = meta or {}
        if "frequently" in str(meta_obj.get("source_file", "")).lower() and idx not in keep_indices:
            keep_indices.append(idx)

    if not keep_indices:
        empty = {
            "ids": [[]],
            "distances": [[]],
            "documents": [[]],
            "metadatas": [[]],
        }
        for passthrough_key in [
            "quality",
            "best_distance",
            "threshold_used",
            "original_count",
        ]:
            if passthrough_key in results:
                empty[passthrough_key] = results[passthrough_key]
        empty["filtered_count"] = 0
        return empty

    # Prefer dedicated faculty profile pages when available.
    profile_page_indices = []
    for idx in keep_indices:
        meta_obj = metas[idx] or {}
        source_file = str(meta_obj.get("source_file", "")).lower()
        if any(token in source_file for token in ["faculty_", "/faculty/", "faculty-"]):
            profile_page_indices.append(idx)
    if profile_page_indices:
        keep_indices = profile_page_indices

    filtered = {
        "ids": [[results["ids"][0][i] for i in keep_indices]],
        "distances": [[results["distances"][0][i] for i in keep_indices]],
        "documents": [[results["documents"][0][i] for i in keep_indices]],
        "metadatas": [[results["metadatas"][0][i] for i in keep_indices]],
    }
    for passthrough_key in [
        "quality",
        "best_distance",
        "threshold_used",
        "original_count",
    ]:
        if passthrough_key in results:
            filtered[passthrough_key] = results[passthrough_key]
    filtered["filtered_count"] = len(keep_indices)
    return filtered


def _has_faculty_signal_match(results: Dict, faculty_signal: Optional[str]) -> bool:
    """Return True when at least one result appears linked to the requested faculty."""
    if not faculty_signal:
        return False

    ids = results.get("ids", [[]])
    if not ids or not ids[0]:
        return False

    metas = results.get("metadatas", [[]])[0]
    for meta in metas:
        meta_obj = meta or {}
        if meta_obj.get("faculty_id") == faculty_signal:
            return True

        refs = [
            token.strip()
            for token in str(meta_obj.get("entity_refs", "")).split(",")
            if token.strip().startswith("faculty_")
        ]
        if faculty_signal in refs:
            return True

    return False


def _filter_regulation_candidates(results: Dict) -> Dict:
    """Keep regulation-typed candidates when present for regulation queries."""
    ids = results.get("ids", [[]])
    if not ids or not ids[0]:
        return results

    metas = results.get("metadatas", [[]])[0]
    keep_indices = []
    for idx, meta in enumerate(metas):
        meta_obj = meta or {}
        ctype = str(meta_obj.get("content_type", "")).lower()
        source_file = str(meta_obj.get("source_file", "")).lower()
        if ctype == "regulation" or "frequently" in source_file:
            keep_indices.append(idx)
    if not keep_indices:
        return results

    filtered = {
        "ids": [[results["ids"][0][i] for i in keep_indices]],
        "distances": [[results["distances"][0][i] for i in keep_indices]],
        "documents": [[results["documents"][0][i] for i in keep_indices]],
        "metadatas": [[results["metadatas"][0][i] for i in keep_indices]],
    }
    for passthrough_key in [
        "quality",
        "best_distance",
        "threshold_used",
        "original_count",
    ]:
        if passthrough_key in results:
            filtered[passthrough_key] = results[passthrough_key]
    filtered["filtered_count"] = len(keep_indices)
    return filtered


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
    - "advisory"  broad course-oriented retrieval, n_results=12
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

    if isinstance(collection, dict):
        primary, _fallback, primary_key = _resolve_collection_route(collection, query_type)
        tagged = query_chromadb(
            primary,
            query_text=query_text,
            query_type=query_type,
            n_results=n_results,
            content_type_filter=content_type_filter,
            distance_threshold=distance_threshold,
            query_embedding=query_embedding,
        )
        tagged["retrieval_primary_collection"] = primary_key
        return tagged

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
    elif query_type == "advisory":
        where_filter = None
        n_results = 12
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
        if config.EMBEDDING_USE_QUERY_DOC_METHODS and hasattr(model, "encode_query"):
            query_embedding = model.encode_query(
                [query_text],
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).tolist()
        else:
            query_embedding = model.encode(
                [query_text],
                prompt_name=config.EMBEDDING_QUERY_PROMPT_NAME,
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
    faculty_signal = _extract_faculty_query_signal(query_text) if query_type == "faculty" else None
    results = _rerank_with_query_signals(
        results,
        query_text,
        query_type,
        faculty_signal=faculty_signal,
    )
    reranked_results = results

    # Apply adaptive distance threshold
    results = apply_adaptive_distance_threshold(results, query_type)

    # Syllabus/detail queries can be overly pruned by strict course thresholds.
    # Keep a small top-ranked tail when we have candidates but thresholding drops all.
    if (
        query_type == "course"
        and _is_syllabus_like_query(query_text)
        and results.get("filtered_count", 0) == 0
        and reranked_results.get("distances")
        and reranked_results["distances"][0]
    ):
        keep = min(max(n_results, 3), len(reranked_results["distances"][0]))
        results = {
            "ids": [reranked_results["ids"][0][:keep]],
            "distances": [reranked_results["distances"][0][:keep]],
            "documents": [reranked_results["documents"][0][:keep]],
            "metadatas": [reranked_results["metadatas"][0][:keep]],
            "quality": "poor",
            "best_distance": min(reranked_results["distances"][0]),
            "threshold_used": None,
            "original_count": len(reranked_results["distances"][0]),
            "filtered_count": keep,
        }

    # Apply strict signal filters after thresholding when query intent is explicit.
    if query_type == "faculty":
        results = _filter_faculty_linked_candidates(
            results,
            query_text,
            faculty_signal=faculty_signal,
        )
    elif query_type == "regulation":
        results = _filter_regulation_candidates(results)

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
    if config.EMBEDDING_USE_QUERY_DOC_METHODS and hasattr(model, "encode_query"):
        query_embedding = model.encode_query(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()
    else:
        query_embedding = model.encode(
            [query_text],
            prompt_name=config.EMBEDDING_QUERY_PROMPT_NAME,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()
    del model

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    primary_collection, fallback_collections, primary_collection_key = _resolve_collection_route(collection, query_type)

    primary = query_chromadb(
        primary_collection,
        query_text=query_text,
        query_type=query_type,
        n_results=n_results,
        query_embedding=query_embedding,
    )
    primary = _tag_results_collection(primary, primary_collection_key)

    primary_count = primary.get("filtered_count", 0)
    primary_quality = primary.get("quality")
    faculty_signal = _extract_faculty_query_signal(query_text) if query_type == "faculty" else None

    # Keep strict routing for non-general queries unless route is empty.
    # For teaching, allow fallback when route is poor and sparse to avoid false negatives.
    fallback_needed = False
    if enable_fallback:
        if query_type == "general":
            fallback_needed = True
        elif primary_count == 0:
            fallback_needed = True
        elif query_type == "course" and _is_syllabus_like_query(query_text) and primary_count < 2:
            fallback_needed = True
        elif query_type == "advisory" and primary_count < 3:
            fallback_needed = True
        elif query_type == "faculty":
            if faculty_signal and not _has_faculty_signal_match(primary, faculty_signal):
                fallback_needed = True
            elif not faculty_signal:
                fallback_needed = True
        elif query_type == "teaching" and primary_quality in {"poor", "fair"} and primary_count < 2:
            fallback_needed = True
        elif query_type == "regulation" and primary_count == 0:
            fallback_needed = True

    if not fallback_needed:
        return {
            **primary,
            "fallback_triggered": False,
            "fallback_count": 0,
            "primary_collection": primary_collection_key,
            "content_type_distribution": _content_type_distribution(primary),
        }

    fallback_raw_candidates = [
        _tag_results_collection(
            primary_collection.query(
                query_embeddings=query_embedding,
                n_results=max(n_results * 2, 8),
            ),
            primary_collection_key,
        )
    ]

    for index, fallback_collection in enumerate(fallback_collections):
        fallback_raw_candidates.append(
            _tag_results_collection(
                fallback_collection.query(
                    query_embeddings=query_embedding,
                    n_results=max(n_results * 2, 8),
                ),
                f"fallback_{index + 1}",
            )
        )

    fallback_raw = _merge_raw_results(
        fallback_raw_candidates,
        max_items=max(n_results * 3, 12),
    )
    fallback_threshold_type = query_type if query_type else "general"
    fallback = apply_adaptive_distance_threshold(fallback_raw, fallback_threshold_type)

    # Re-apply query-signal reranking on fallback results using original query type.
    # This improves partial-name faculty queries (e.g., "Who is Dr Tessy").
    fallback = _rerank_with_query_signals(
        fallback,
        query_text,
        query_type,
        faculty_signal=faculty_signal,
    )

    if query_type == "faculty":
        fallback = _filter_faculty_linked_candidates(
            fallback,
            query_text,
            faculty_signal=faculty_signal,
        )
    elif query_type == "regulation":
        fallback = _filter_regulation_candidates(fallback)

    # If adaptive threshold filters everything, keep top broad candidates
    # to avoid zero-result responses for open-ended general queries.
    if (
        query_type != "faculty"
        and fallback.get("filtered_count", 0) == 0
        and fallback_raw.get("distances")
    ):
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
        "primary_collection": primary_collection_key,
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
            id_to_idx = {cid: idx for idx, cid in enumerate(cached_ids)}
            reorder = [id_to_idx[cid] for cid in current_ids]
            embeddings = cached_embeddings[reorder]
        else:
            missing_ids_set = set(current_ids) - set(cached_ids)
            print(f"[WARN] Cache mismatch. Missing {len(missing_ids_set)} embeddings. Partially regenerating.")
            
            # Identify missing chunks and their exact IDs in a consistent order
            missing_chunks = [c for c in chunks if c["chunk_id"] in missing_ids_set]
            missing_texts = [c["text"] for c in missing_chunks]
            missing_ids = [c["chunk_id"] for c in missing_chunks]
            
            if missing_texts:
                model = load_embedding_model(config.EMBEDDING_MODEL, device="auto")
                new_embeddings = generate_embeddings(
                    model,
                    missing_texts,
                    batch_size=config.EMBEDDING_BATCH_SIZE,
                )
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Update cache
                cached_ids = list(cached_ids) + missing_ids
                if cached_embeddings.shape[0] > 0:
                    cached_embeddings = np.vstack([cached_embeddings, new_embeddings])
                else:
                    cached_embeddings = new_embeddings
                
                # Save cache since we got new ones
                cache_embeddings(cached_ids, cached_embeddings, cache_path)
            
            # Reorder all
            id_to_idx = {cid: idx for idx, cid in enumerate(cached_ids)}
            reorder = [id_to_idx[cid] for cid in current_ids]
            embeddings = cached_embeddings[reorder]

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
    collection_map = get_rag_collections(
        client,
        recreate=force_reembed,
        create_missing=True,
    )

    if not collection_map:
        raise RuntimeError("No ChromaDB collections available. Check collection configuration.")

    print("[OK] Collection handles ready:")
    for key, coll in collection_map.items():
        try:
            count = coll.count()
        except Exception:
            count = "unknown"
        print(f"   {key}: {count}")

    primary_validation_collection = (
        collection_map.get("legacy")
        or collection_map.get("non_table")
        or next(iter(collection_map.values()))
    )

    # ========== STEP 5: Ingest Chunks ==========
    if force_reembed or any(coll.count() < len(chunks) for coll in collection_map.values()):
        print("\n[STEP 5] Ingesting chunks into ChromaDB...")

        split_payload = _split_chunks_for_collections(chunks, embeddings)
        ingest_stats_by_collection: Dict[str, Dict] = {}

        for key, coll in collection_map.items():
            payload = split_payload.get(key, split_payload["legacy"])
            payload_chunks = payload["chunks"]
            payload_embeddings = payload["embeddings"]

            if not payload_chunks:
                ingest_stats_by_collection[key] = {
                    "total_processed": 0,
                    "successfully_ingested": 0,
                    "failed_chunks": [],
                    "final_collection_count": coll.count(),
                }
                continue


            ingest_stats_by_collection[key] = ingest_chunks_to_chromadb(
                coll,
                payload_chunks,
                payload_embeddings,
                batch_size=100,
            )

        aggregate_failed = sorted(
            {
                chunk_id
                for stats in ingest_stats_by_collection.values()
                for chunk_id in stats.get("failed_chunks", [])
            }
        )
        ingest_stats = {
            "total_processed": len(chunks),
            "successfully_ingested": len(chunks) - len(aggregate_failed),
            "failed_chunks": aggregate_failed,
            "final_collection_count": primary_validation_collection.count(),
            "collections": ingest_stats_by_collection,
        }
    else:
        print("\n[STEP 5] Skipped (collection already populated)")
        ingest_stats = {
            "total_processed": len(chunks),
            "successfully_ingested": len(chunks),
            "failed_chunks": [],
            "final_collection_count": primary_validation_collection.count(),
            "collections": {
                key: {
                    "total_processed": len(chunks),
                    "successfully_ingested": len(chunks),
                    "failed_chunks": [],
                    "final_collection_count": coll.count(),
                }
                for key, coll in collection_map.items()
            },
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
        primary_validation_collection,
        chunk_report,
        test_queries,
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
            "total_documents": primary_validation_collection.count(),
            "chunks_by_type": chunk_report["chunks_by_type"],
            "collection_counts": {
                key: coll.count() for key, coll in collection_map.items()
            },
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

    collection_map = get_rag_collections(
        client,
        recreate=False,
        create_missing=False,
    )
    if not collection_map:
        print("[ERR] Collection not found. Run ingestion first:")
        print("   python main.py --stage embed")
        return

    print(f"Collection: {config.CHROMADB_COLLECTION}")
    for key, coll in collection_map.items():
        print(f"Documents ({key}): {coll.count()}")

    # Classify and query
    query_type = classify_query_type(query_text)
    print(f"\nQuery: '{query_text}'")
    print(f"Classified as: {query_type}")

    results = query_chromadb_with_fallback(
        collection_map,
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
