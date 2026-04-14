"""
MBCET CSE Chatbot Backend.

Provides answer_question() which combines:
1. Knowledge Graph lookup (for relational "who teaches X" queries)
2. ChromaDB vector retrieval (for general queries)
3. Groq LLM (Llama 3.3 70B) for answer synthesis

The embedding model and ChromaDB collection are cached as module-level
singletons to avoid the ~11-second cold-start per query.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import config

import sys
# Fix Windows console encoding for emoji/unicode characters
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

logger = logging.getLogger(__name__)

# ========================================================================
# SINGLETON RESOURCE MANAGER
# ========================================================================

_embedding_model = None
_chromadb_collection = None
_knowledge_graph = None


def _get_embedding_model():
    """Load embedding model once and cache it."""
    global _embedding_model
    if _embedding_model is None:
        from rag_ingestion import load_embedding_model
        _embedding_model = load_embedding_model(device="auto")
    return _embedding_model


def _get_chromadb_collection():
    """Get or create the ChromaDB collection (cached)."""
    global _chromadb_collection
    if _chromadb_collection is None:
        import chromadb
        client = chromadb.PersistentClient(path=str(config.CHROMADB_DIR))
        _chromadb_collection = client.get_collection(
            name=config.CHROMADB_COLLECTION,
        )
    return _chromadb_collection


def _get_knowledge_graph():
    """Load the canonical knowledge graph JSON (cached)."""
    global _knowledge_graph
    if _knowledge_graph is None:
        from chunker.knowledge_graph import load_knowledge_graph
        _knowledge_graph = load_knowledge_graph()
    return _knowledge_graph


# ========================================================================
# DATA CLASSES
# ========================================================================


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with metadata."""
    chunk_id: str
    text: str
    content_type: str
    source_file: str
    section_hierarchy: str
    distance: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class ChatResponse:
    """Full chatbot response."""
    answer: str
    query_type: str
    source_chunks: List[RetrievedChunk] = field(default_factory=list)
    kg_answer: Optional[str] = None
    quality: str = "unknown"
    response_time_ms: float = 0.0


# ========================================================================
# QUERY CLASSIFICATION
# ========================================================================


def classify_query(query: str) -> str:
    """
    Classify query into type for routing.

    Returns:
        "teaching" | "faculty" | "course" | "timetable" | "regulation" | "general"
    """
    from rag_ingestion import classify_query_type
    return classify_query_type(query)


# ========================================================================
# RETRIEVAL
# ========================================================================


def _retrieve_from_kg(query: str) -> Optional[str]:
    """Try to answer from the knowledge graph directly."""
    graph = _get_knowledge_graph()
    if graph is None:
        return None

    from chunker.knowledge_graph import query_knowledge_graph
    answer = query_knowledge_graph(graph, query)

    # The KG query function returns a fallback message for unsupported queries
    if answer and "Supported queries:" not in answer and "No matching" not in answer and "No instructor" not in answer and "No courses" not in answer:
        return answer
    return None


def _retrieve_from_chromadb(
    query: str,
    query_type: str,
    n_results: int = 5,
) -> List[RetrievedChunk]:
    """
    Retrieve relevant chunks using routed retrieval with controlled fallback.
    """
    collection = _get_chromadb_collection()
    from rag_ingestion import query_chromadb_with_fallback

    results = query_chromadb_with_fallback(
        collection,
        query_text=query,
        query_type=query_type,
        n_results=n_results,
        enable_fallback=True,
        rerank_mixed=True,
    )

    # Parse results
    chunks = []
    if results["ids"] and len(results["ids"][0]) > 0:
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            chunks.append(RetrievedChunk(
                chunk_id=results["ids"][0][i],
                text=results["documents"][0][i],
                content_type=meta.get("content_type", "unknown"),
                source_file=meta.get("source_file", "unknown"),
                section_hierarchy=meta.get("section_hierarchy", ""),
                distance=results["distances"][0][i],
                metadata=meta,
            ))

    return chunks


def _confidence_label(distance: float) -> str:
    if distance < 0.3:
        return "excellent"
    if distance < 0.6:
        return "good"
    if distance < 1.0:
        return "fair"
    return "poor"


def _smart_truncate(text: str, max_chars: int = 1400, min_chars: int = 900) -> str:
    if len(text) <= max_chars:
        return text

    window = text[:max_chars]
    boundary = max(window.rfind(". "), window.rfind("\n"))
    if boundary >= min_chars:
        clipped = text[: boundary + 1].rstrip()
    else:
        clipped = window.rstrip()

    remaining = len(text) - len(clipped)
    return f"{clipped}\n[truncated {remaining} chars]"


# ========================================================================
# LLM ANSWER SYNTHESIS
# ========================================================================


def _get_groq_client():
    """Initialize Groq client."""
    try:
        from groq import Groq
    except ImportError:
        return None

    api_key = config.GROQ_API_KEY
    if not api_key:
        return None

    return Groq(api_key=api_key)


def _synthesize_answer_with_llm(
    query: str,
    query_type: str,
    chunks: List[RetrievedChunk],
    kg_answer: Optional[str] = None,
) -> str:
    """
    Use Groq (Llama 3.3 70B) to synthesize a natural-language answer from retrieved chunks.
    Falls back to formatted chunk display if Groq is unavailable.
    """
    client = _get_groq_client()

    if client is None:
        # Fallback: no LLM available, format chunks directly
        return _format_chunks_as_answer(query, query_type, chunks, kg_answer)

    # Build context from chunks
    context_parts = []

    if kg_answer:
        context_parts.append(f"[Knowledge Graph Result]\n{kg_answer}")

    sorted_chunks = sorted(chunks[:5], key=lambda c: c.distance)
    for i, chunk in enumerate(sorted_chunks, 1):
        source = Path(chunk.source_file).stem if chunk.source_file else "unknown"
        confidence = _confidence_label(chunk.distance)
        context_parts.append(
            f"[Source {i} | confidence:{confidence} | distance:{chunk.distance:.3f} | {source} ({chunk.content_type})]\n"
            f"{_smart_truncate(chunk.text)}"
        )

    context = "\n\n---\n\n".join(context_parts)

    system_prompt = """You are a helpful assistant for the MBCET CSE department.
Answer ONLY from the provided context and keep responses factual.

Rules:
- Prioritize higher-confidence sources and reconcile multiple sources explicitly.
- If information is missing or conflicting, say so clearly.
- For faculty queries: include designation, qualifications, and email if present.
- For course/timetable queries: include code/name/credits/schedule details only when present.
- Use concise bullet points for lists.
- Never fabricate facts not present in context."""

    user_message = f"""Context:
{context}

Question: {query}"""

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            model=config.GROQ_MODEL,
            temperature=0.3,
            max_tokens=1024,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Groq API error: {e}")
        return _format_chunks_as_answer(query, query_type, chunks, kg_answer)


def _format_chunks_as_answer(
    query: str,
    query_type: str,
    chunks: List[RetrievedChunk],
    kg_answer: Optional[str] = None,
) -> str:
    """Format retrieved chunks as a readable answer (no-LLM fallback)."""
    parts = []

    if kg_answer:
        parts.append(f"**Knowledge Graph:** {kg_answer}")

    if chunks:
        parts.append("**Relevant information found:**\n")
        for i, chunk in enumerate(chunks[:3], 1):
            source = Path(chunk.source_file).stem if chunk.source_file else "unknown"
            # Truncate long chunks
            text = chunk.text[:500]
            if len(chunk.text) > 500:
                text += "..."
            parts.append(f"**Source {i}** ({chunk.content_type}  {source}):\n{text}\n")

    if not parts:
        parts.append(
            "I couldn't find relevant information for your query in the knowledge base. "
            "Try rephrasing your question or asking about MBCET CSE department faculty, "
            "courses, syllabi, or regulations."
        )

    return "\n".join(parts)


# ========================================================================
# MAIN ENTRY POINT
# ========================================================================


def answer_question(query: str) -> ChatResponse:
    """
    Main entry point: answer a user question about MBCET CSE department.

    Pipeline:
    1. Classify query type
    2. Check Knowledge Graph (for relational queries)
    3. Retrieve from ChromaDB
    4. Synthesize answer via LLM (or formatted fallback)

    Args:
        query: User's natural language question.

    Returns:
        ChatResponse with answer, sources, and metadata.
    """
    start_time = time.time()

    # 1. Classify
    query_type = classify_query(query)
    logger.info(f"Query: '{query}'  Type: {query_type}")

    # 2. Knowledge Graph lookup (for teaching/relational queries)
    kg_answer = None
    if query_type == "teaching":
        kg_answer = _retrieve_from_kg(query)

    # 3. ChromaDB retrieval
    chunks = _retrieve_from_chromadb(query, query_type, n_results=5)

    # Determine quality
    if chunks:
        best_distance = min(c.distance for c in chunks)
        if best_distance < 0.3:
            quality = "excellent"
        elif best_distance < 0.6:
            quality = "good"
        elif best_distance < 1.0:
            quality = "fair"
        else:
            quality = "poor"
    else:
        quality = "none"

    # 4. Synthesize answer
    answer = _synthesize_answer_with_llm(query, query_type, chunks, kg_answer)

    response_time = (time.time() - start_time) * 1000

    return ChatResponse(
        answer=answer,
        query_type=query_type,
        source_chunks=chunks,
        kg_answer=kg_answer,
        quality=quality,
        response_time_ms=round(response_time, 2),
    )


def warmup():
    """Pre-load all resources to avoid cold-start latency."""
    print("[*] Warming up chatbot resources...")
    _get_embedding_model()
    _get_chromadb_collection()
    _get_knowledge_graph()
    print("[OK] Chatbot ready!")
