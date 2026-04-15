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
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

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
        from rag_ingestion import initialize_chromadb_safe, get_rag_collections

        client = initialize_chromadb_safe(str(config.CHROMADB_DIR))
        _chromadb_collection = get_rag_collections(
            client,
            recreate=False,
            create_missing=False,
        )

        if not _chromadb_collection:
            raise RuntimeError(
                "No ChromaDB collection found. Run embedding stage first: python main.py --stage embed"
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


@dataclass
class RetrievalPlan:
    """Planner output controlling retrieval strategy."""
    query_type: str
    source_mode: str = "hybrid"  # kg_only | vector_only | hybrid | no_retrieval
    rewritten_query: str = ""
    rationale: str = ""


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


def _default_retrieval_plan(query: str) -> RetrievalPlan:
    """Fast deterministic fallback plan when LLM planning is unavailable."""
    q_norm = query.strip().lower()
    greeting_markers = {
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
        "thanks", "thank you", "how are you", "yo", "hola",
    }
    if q_norm in greeting_markers or any(q_norm.startswith(m + " ") for m in greeting_markers):
        return RetrievalPlan(
            query_type="general",
            source_mode="no_retrieval",
            rewritten_query=query,
            rationale="greeting_smalltalk_default",
        )

    query_type = classify_query(query)
    if query_type == "teaching":
        mode = "hybrid"
    elif query_type == "faculty":
        mode = "vector_only"
    elif query_type == "regulation":
        mode = "hybrid"
    else:
        mode = "vector_only"
    return RetrievalPlan(
        query_type=query_type,
        source_mode=mode,
        rewritten_query=query,
        rationale="default_router",
    )


def _extract_json_block(raw_text: str) -> Optional[Dict]:
    """Extract planner JSON from model output, including fenced blocks."""
    if not raw_text:
        return None

    text = raw_text.strip()

    # Direct JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Fenced JSON block
    fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    # Fallback to first object-like block
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        try:
            parsed = json.loads(text[brace_start : brace_end + 1])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return None


def _normalize_planner_output(plan: Dict, original_query: str) -> RetrievalPlan:
    """Normalize raw planner dictionary into validated RetrievalPlan."""
    valid_query_types = {"teaching", "faculty", "course", "timetable", "regulation", "general"}
    valid_modes = {"kg_only", "vector_only", "hybrid", "no_retrieval"}

    query_type = str(plan.get("query_type", "")).strip().lower()
    source_mode = str(plan.get("source_mode", "")).strip().lower()
    rewritten_query = str(plan.get("rewritten_query", "")).strip()
    rationale = str(plan.get("rationale", "")).strip()

    if query_type not in valid_query_types:
        query_type = classify_query(original_query)

    if source_mode not in valid_modes:
        source_mode = "hybrid" if query_type in {"teaching", "regulation"} else "vector_only"

    if not rewritten_query:
        rewritten_query = original_query

    return RetrievalPlan(
        query_type=query_type,
        source_mode=source_mode,
        rewritten_query=rewritten_query,
        rationale=rationale or "planner_normalized",
    )


def _should_use_llm_planner(query: str, default_query_type: str) -> bool:
    """Use planner selectively to balance latency and quality."""
    if default_query_type in {"general", "regulation", "teaching"}:
        return True

    # Ambiguous intent markers where source choice is uncertain.
    ambiguous_markers = ["or", "and", "about", "details", "explain", "tell me"]
    q_lower = query.lower()
    if any(marker in q_lower for marker in ambiguous_markers):
        return True

    return False


def _plan_retrieval_with_llm(query: str) -> RetrievalPlan:
    """Plan retrieval strategy (kg_only/vector_only/hybrid/no_retrieval) using LLM with strict fallback."""
    default_plan = _default_retrieval_plan(query)

    if not _should_use_llm_planner(query, default_plan.query_type):
        return default_plan

    client = _get_groq_client()
    if client is None:
        return default_plan

    planner_system_prompt = """You are a retrieval planner for an academic QA system.
Given a user query, output ONLY a JSON object with keys:
- query_type: one of [teaching, faculty, course, timetable, regulation, general]
- source_mode: one of [kg_only, vector_only, hybrid, no_retrieval]
- rewritten_query: short rewritten retrieval query
- rationale: brief reason

Rules:
- Queries like 'Who is Dr X', 'Who is Prof Y', or 'Tell me about faculty Z' are faculty/profile queries.
- Use kg_only only for strict relation lookup queries like 'who teaches X'.
- Use vector_only for descriptive profile/course/timetable lookups.
- Use hybrid when relation + context/policy may both be needed, or query is broad.
- Use no_retrieval for greetings, gratitude, or pure conversational small talk.
- Do not include markdown or any text outside JSON."""

    planner_user_prompt = f"Query: {query}"

    try:
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": planner_system_prompt},
                {"role": "user", "content": planner_user_prompt},
            ],
            model=config.GROQ_MODEL,
            temperature=0.0,
            max_tokens=220,
        )
        content = (completion.choices[0].message.content or "").strip()
        parsed = _extract_json_block(content)
        if not parsed:
            return default_plan
        return _normalize_planner_output(parsed, original_query=query)
    except Exception as exc:
        logger.warning(f"Planner call failed: {exc}")
        return default_plan


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

    # Surface collection-level retrieval behavior for runtime debugging.
    collection_hits: Dict[str, int] = {}
    metadatas = results.get("metadatas", [[]])
    if metadatas and metadatas[0]:
        for meta in metadatas[0]:
            key = (meta or {}).get("retrieval_collection", "unknown")
            collection_hits[key] = collection_hits.get(key, 0) + 1
    logger.info(
        "Vector retrieval summary: primary_collection=%s fallback=%s type_mix=%s collection_hits=%s",
        results.get("primary_collection", "legacy"),
        results.get("fallback_triggered", False),
        results.get("content_type_distribution", {}),
        collection_hits,
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


def _execute_retrieval_plan(query: str, plan: RetrievalPlan) -> Tuple[Optional[str], List[RetrievedChunk]]:
    """Execute planner-selected retrieval mode and return (kg_answer, chunks)."""
    retrieval_query = plan.rewritten_query or query

    if plan.source_mode == "no_retrieval":
        return None, []

    if plan.source_mode == "kg_only":
        kg_answer = _retrieve_from_kg(query)
        if kg_answer:
            return kg_answer, []
        # Safety net: if KG misses, recover through routed vector retrieval.
        return None, _retrieve_from_chromadb(retrieval_query, plan.query_type, n_results=5)

    if plan.source_mode == "vector_only":
        return None, _retrieve_from_chromadb(retrieval_query, plan.query_type, n_results=5)

    # hybrid
    kg_answer = None
    if plan.query_type in {"teaching", "regulation", "general", "course", "faculty", "timetable"}:
        kg_answer = _retrieve_from_kg(query)
    chunks = _retrieve_from_chromadb(retrieval_query, plan.query_type, n_results=5)
    return kg_answer, chunks


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
        content = (chat_completion.choices[0].message.content or "").strip()
        if not content or content.lower() in {"none", "null", "n/a"}:
            logger.info("LLM synthesis returned empty/null-like content; using formatted fallback")
            return _format_chunks_as_answer(query, query_type, chunks, kg_answer)
        return content
    except Exception as e:
        logger.warning(f"Groq API error: {e}")
        return _format_chunks_as_answer(query, query_type, chunks, kg_answer)


def _answer_without_retrieval(query: str) -> str:
    """Answer direct conversational queries (greetings/small-talk) without retrieval."""
    client = _get_groq_client()
    if client is None:
        q = query.strip().lower()
        if any(g in q for g in ("hi", "hello", "hey", "good morning", "good evening")):
            return "Hello! I can help with MBCET CSE info such as faculty, courses, regulations, and timetables."
        if "thank" in q:
            return "You're welcome. If you want, ask about any MBCET CSE faculty, course, or regulation detail."
        return "Hi! Ask me about MBCET CSE faculty, courses, timetables, or regulations."

    system_prompt = """You are a polite assistant for MBCET CSE users.
The user asked a conversational query that does not require retrieval.
Reply briefly and naturally (1-3 sentences), and optionally suggest what academic queries you can help with.
Do not invent factual department details in this mode."""

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            model=config.GROQ_MODEL,
            temperature=0.3,
            max_tokens=120,
        )
        return (chat_completion.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning(f"Direct no-retrieval response failed: {exc}")
        return "Hi! I can help with MBCET CSE faculty, courses, timetables, and regulations."


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

    # 1. Plan retrieval (LLM planner with deterministic fallback)
    plan = _plan_retrieval_with_llm(query)
    query_type = plan.query_type
    logger.info(
        "Query='%s' planner_mode=%s query_type=%s rewritten='%s' rationale='%s'",
        query,
        plan.source_mode,
        plan.query_type,
        plan.rewritten_query,
        plan.rationale,
    )

    # 2. Execute retrieval (or bypass retrieval for conversational queries)
    if plan.source_mode == "no_retrieval":
        kg_answer, chunks = None, []
        quality = "not_applicable"
        answer = _answer_without_retrieval(query)
        response_time = (time.time() - start_time) * 1000
        return ChatResponse(
            answer=answer,
            query_type=query_type,
            source_chunks=chunks,
            kg_answer=kg_answer,
            quality=quality,
            response_time_ms=round(response_time, 2),
        )

    kg_answer, chunks = _execute_retrieval_plan(query, plan)
    logger.info(
        "Retrieval result: mode=%s kg_hit=%s chunk_count=%d",
        plan.source_mode,
        bool(kg_answer),
        len(chunks),
    )

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

    if chunks:
        top = sorted(chunks, key=lambda c: c.distance)[:3]
        top_summary = [f"{c.content_type}:{c.distance:.3f}:{Path(c.source_file).stem}" for c in top]
        logger.info("Top chunks: %s", " | ".join(top_summary))

    # 3. Synthesize answer
    answer = _synthesize_answer_with_llm(query, query_type, chunks, kg_answer)

    response_time = (time.time() - start_time) * 1000
    logger.info("Answer generated: query_type=%s quality=%s response_time_ms=%.2f", query_type, quality, response_time)

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
