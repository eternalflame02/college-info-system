import argparse
import json
import os
import random
import statistics
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import config
from rag_ingestion import load_embedding_model, generate_embeddings_with_fallback
from wide_quality_eval import build_query_set, _tokenize


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _gpu_mem_used_mb() -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        value = result.stdout.strip().splitlines()[0].strip()
        return int(value)
    except Exception:
        return None


def _post_embeddings(url: str, model_id: str, texts: List[str], api_token: str | None) -> List[List[float]]:
    payload = {
        "model": model_id,
        "input": texts,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
    }
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    from urllib.request import Request, urlopen

    req = Request(url, data=data, headers=headers, method="POST")
    with urlopen(req, timeout=300) as resp:
        body = resp.read().decode("utf-8")
    parsed = json.loads(body)
    return [item["embedding"] for item in parsed.get("data", [])]


def embed_with_lmstudio(
    texts: List[str],
    model_id: str,
    endpoint: str,
    batch_size: int,
    api_token: str | None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    embeddings: List[List[float]] = []
    batch_latencies: List[float] = []
    max_vram = 0

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        started = time.perf_counter()
        batch_emb = _post_embeddings(endpoint, model_id, batch, api_token)
        elapsed = time.perf_counter() - started
        batch_latencies.append(elapsed)
        embeddings.extend(batch_emb)

        vram = _gpu_mem_used_mb()
        if vram is not None:
            max_vram = max(max_vram, vram)

    mat = np.asarray(embeddings, dtype=np.float32)
    mat = _l2_normalize(mat)

    stats = {
        "batch_count": len(batch_latencies),
        "batch_latency_mean_s": float(statistics.mean(batch_latencies) if batch_latencies else 0.0),
        "batch_latency_max_s": float(max(batch_latencies) if batch_latencies else 0.0),
        "embedding_count": int(mat.shape[0]),
        "embedding_dim": int(mat.shape[1]) if mat.size else 0,
        "max_vram_mb": int(max_vram),
    }
    return mat, stats


def embed_with_sentence_transformer(
    texts: List[str],
    batch_size: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    model = load_embedding_model(model_name="google/embeddinggemma-300m", device="cuda")
    started = time.perf_counter()
    embeddings = generate_embeddings_with_fallback(model, texts, batch_size=batch_size)
    elapsed = time.perf_counter() - started
    embeddings = _l2_normalize(np.asarray(embeddings, dtype=np.float32))

    vram = _gpu_mem_used_mb()
    stats = {
        "elapsed_s": float(elapsed),
        "embedding_count": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.size else 0,
        "max_vram_mb": int(vram or 0),
    }
    return embeddings, stats


def embed_queries_sentence_transformer(texts: List[str], batch_size: int) -> Tuple[np.ndarray, float]:
    model = load_embedding_model(model_name="google/embeddinggemma-300m", device="cuda")
    started = time.perf_counter()

    if hasattr(model, "encode_query"):
        emb = model.encode_query(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    else:
        emb = model.encode(
            texts,
            prompt_name=config.EMBEDDING_QUERY_PROMPT_NAME,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    elapsed = time.perf_counter() - started
    emb = _l2_normalize(np.asarray(emb, dtype=np.float32))
    return emb, float(elapsed)


def _first_hit_rank(
    ranked_indices: np.ndarray,
    expected_entities: set,
    doc_entities: List[set],
    doc_tokens: List[set],
    query_tokens: set,
) -> int | None:
    for rank, idx in enumerate(ranked_indices, start=1):
        if expected_entities:
            if expected_entities & doc_entities[idx]:
                return rank
        else:
            if len(query_tokens & doc_tokens[idx]) >= 2:
                return rank
    return None


def evaluate_retrieval(
    query_embs: np.ndarray,
    doc_embs: np.ndarray,
    queries: List,
    doc_entities: List[set],
    doc_tokens: List[set],
    top_k: int,
) -> Dict[str, float]:
    hit_vals = []
    hit_at_1_vals = []
    mrr_vals = []
    ranks = []

    for q_idx, query in enumerate(queries):
        sims = doc_embs @ query_embs[q_idx]
        if top_k >= len(sims):
            ranked = np.argsort(-sims)
        else:
            top_idx = np.argpartition(-sims, top_k)[:top_k]
            ranked = top_idx[np.argsort(-sims[top_idx])]

        expected_entities = set(query.expected_entity_ids or [])
        query_tokens = set(_tokenize(query.text))
        hit_rank = _first_hit_rank(ranked, expected_entities, doc_entities, doc_tokens, query_tokens)

        if hit_rank is None:
            hit_vals.append(0.0)
            hit_at_1_vals.append(0.0)
            mrr_vals.append(0.0)
        else:
            hit_vals.append(1.0)
            hit_at_1_vals.append(1.0 if hit_rank == 1 else 0.0)
            mrr_vals.append(1.0 / hit_rank)
            ranks.append(hit_rank)

    median_rank = float(statistics.median(ranks) if ranks else 0.0)

    return {
        "hit_at_1": float(sum(hit_at_1_vals) / max(len(hit_at_1_vals), 1)),
        "hit_at_k": float(sum(hit_vals) / max(len(hit_vals), 1)),
        "mrr": float(sum(mrr_vals) / max(len(mrr_vals), 1)),
        "median_rank": median_rank,
        "query_count": int(len(queries)),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark LM Studio embedding models")
    parser.add_argument("--lmstudio-url", type=str, default="http://localhost:1234")
    parser.add_argument("--q4-model", type=str, default="text-embedding-embeddinggemma-300m-qat@q4_0")
    parser.add_argument("--q8-model", type=str, default="text-embedding-embeddinggemma-300m-qat@q8_0")
    parser.add_argument("--queries", type=int, default=75)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--sample-docs", type=int, default=500)
    parser.add_argument("--batch-docs", type=int, default=64)
    parser.add_argument("--batch-queries", type=int, default=64)
    parser.add_argument("--batch-lmstudio", type=int, default=32)
    args = parser.parse_args()

    endpoint = args.lmstudio_url.rstrip("/") + "/v1/embeddings"
    api_token = os.getenv("LM_API_TOKEN")

    chunks = _load_json(config.CHUNKS_FILE)
    faculty = _load_json(config.FACULTY_FILE)
    courses = _load_json(config.COURSES_FILE)
    programs = _load_json(config.PROGRAMS_FILE)
    teaching_assignments = _load_json(config.TEACHING_ASSIGNMENTS_FILE)

    queries = build_query_set(
        faculty=faculty,
        courses=courses,
        programs=programs,
        teaching_assignments=teaching_assignments,
        target_queries=args.queries,
    )

    if args.sample_docs and args.sample_docs < len(chunks):
        rng = random.Random(42)
        sampled = rng.sample(chunks, args.sample_docs)
    else:
        sampled = chunks

    doc_texts = [c.get("text", "") for c in sampled]
    doc_entities = [set(c.get("entity_refs") or []) for c in sampled]
    doc_tokens = [set(_tokenize(c.get("text", ""))) for c in sampled]
    query_texts = [q.text for q in queries]

    results = {
        "run_config": {
            "lmstudio_url": args.lmstudio_url,
            "q4_model": args.q4_model,
            "q8_model": args.q8_model,
            "query_count": len(queries),
            "top_k": args.top_k,
            "sample_docs": len(doc_texts),
            "batch_docs": args.batch_docs,
            "batch_queries": args.batch_queries,
            "batch_lmstudio": args.batch_lmstudio,
        }
    }

    print("[1/3] Baseline: google/embeddinggemma-300m (SentenceTransformer)")
    base_docs, base_doc_stats = embed_with_sentence_transformer(doc_texts, args.batch_docs)
    base_queries, base_query_time = embed_queries_sentence_transformer(query_texts, args.batch_queries)
    base_metrics = evaluate_retrieval(
        base_queries,
        base_docs,
        queries,
        doc_entities,
        doc_tokens,
        args.top_k,
    )
    results["baseline_sentence_transformer"] = {
        "doc_embedding": base_doc_stats,
        "query_embedding": {"elapsed_s": base_query_time},
        "retrieval_metrics": base_metrics,
    }

    print("[2/3] LM Studio Q4")
    q4_docs, q4_doc_stats = embed_with_lmstudio(
        doc_texts, args.q4_model, endpoint, args.batch_lmstudio, api_token
    )
    q4_queries, q4_query_stats = embed_with_lmstudio(
        query_texts, args.q4_model, endpoint, args.batch_lmstudio, api_token
    )
    q4_metrics = evaluate_retrieval(
        q4_queries,
        q4_docs,
        queries,
        doc_entities,
        doc_tokens,
        args.top_k,
    )
    results["lmstudio_q4"] = {
        "doc_embedding": q4_doc_stats,
        "query_embedding": q4_query_stats,
        "retrieval_metrics": q4_metrics,
    }

    print("[3/3] LM Studio Q8")
    q8_docs, q8_doc_stats = embed_with_lmstudio(
        doc_texts, args.q8_model, endpoint, args.batch_lmstudio, api_token
    )
    q8_queries, q8_query_stats = embed_with_lmstudio(
        query_texts, args.q8_model, endpoint, args.batch_lmstudio, api_token
    )
    q8_metrics = evaluate_retrieval(
        q8_queries,
        q8_docs,
        queries,
        doc_entities,
        doc_tokens,
        args.top_k,
    )
    results["lmstudio_q8"] = {
        "doc_embedding": q8_doc_stats,
        "query_embedding": q8_query_stats,
        "retrieval_metrics": q8_metrics,
    }

    output_path = config.VALIDATION_DIR / "embedding_bench_lmstudio.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Results written to {output_path}")


if __name__ == "__main__":
    main()
