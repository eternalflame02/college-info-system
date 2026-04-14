"""
Wide quality evaluation for chunk quality, embeddings, retrieval, and knowledge graph.

Outputs are written to data/validation:
- chunk_quality_metrics.json
- embedding_quality_metrics.json
- retrieval_metrics.json
- kg_quality_metrics.json
- wide_eval_summary.md
- improvement_backlog.md
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import config
from knowledge_graph.builder import build_knowledge_graph, validate_graph
from rag_ingestion import load_embedding_model, query_chromadb_with_fallback


@dataclass
class EvalQuery:
    text: str
    query_type: str
    expected_content_type: str | None = None
    expected_entity_ids: List[str] | None = None


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(payload)


def _safe_percentiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


def _tokenize(text: str) -> List[str]:
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if len(t) > 2]


def _parse_entity_refs(raw: str | list | None) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str):
        return [x.strip() for x in raw.split(",") if x.strip()]
    return []


def evaluate_chunk_quality(chunks: List[dict]) -> dict:
    total = len(chunks)
    word_counts = [int(c.get("word_count", 0) or 0) for c in chunks]
    content_type_counts = Counter(c.get("content_type", "unknown") for c in chunks)
    source_type_counts = Counter(c.get("source_type", "unknown") for c in chunks)

    missing_section = sum(1 for c in chunks if not c.get("section_hierarchy"))
    missing_content_type = sum(1 for c in chunks if not c.get("content_type"))
    missing_source = sum(1 for c in chunks if not c.get("source_file"))

    hash_counts = Counter(c.get("hash", "") for c in chunks if c.get("hash"))
    exact_duplicate_chunks = sum(v - 1 for v in hash_counts.values() if v > 1)

    # Near-duplicate proxy: normalized first-200-char token signature collisions.
    sig_counts: Counter[str] = Counter()
    for c in chunks:
        text = (c.get("text") or "")[:200]
        toks = sorted(set(_tokenize(text)))[:16]
        sig = "|".join(toks)
        if sig:
            sig_counts[sig] += 1
    near_duplicate_proxy = sum(v - 1 for v in sig_counts.values() if v > 1)

    entity_ref_counts = [len(c.get("entity_refs", []) or []) for c in chunks]
    with_entities = sum(1 for n in entity_ref_counts if n > 0)

    # Type-specific quality checks.
    table_chunks = [c for c in chunks if c.get("content_type") == "table"]
    profile_chunks = [c for c in chunks if c.get("content_type") == "profile"]
    regulation_chunks = [c for c in chunks if c.get("content_type") == "regulation"]

    table_has_pipe_ratio = (
        sum(1 for c in table_chunks if "|" in (c.get("text") or "")) / len(table_chunks)
        if table_chunks
        else 0.0
    )
    profile_has_faculty_ref_ratio = (
        sum(
            1
            for c in profile_chunks
            if any(ref.startswith("faculty_") for ref in (c.get("entity_refs") or []))
        )
        / len(profile_chunks)
        if profile_chunks
        else 0.0
    )
    regulation_keyword_ratio = (
        sum(
            1
            for c in regulation_chunks
            if any(
                k in (c.get("text") or "").lower()
                for k in ("regulation", "curriculum", "grading", "scheme")
            )
        )
        / len(regulation_chunks)
        if regulation_chunks
        else 0.0
    )

    return {
        "total_chunks": total,
        "word_count": {
            "min": int(min(word_counts) if word_counts else 0),
            "max": int(max(word_counts) if word_counts else 0),
            "mean": float(statistics.mean(word_counts) if word_counts else 0.0),
            "median": float(statistics.median(word_counts) if word_counts else 0.0),
            "percentiles": _safe_percentiles(word_counts),
        },
        "distribution": {
            "content_type_counts": dict(sorted(content_type_counts.items())),
            "source_type_counts": dict(sorted(source_type_counts.items())),
        },
        "metadata_completeness": {
            "missing_section_hierarchy": missing_section,
            "missing_content_type": missing_content_type,
            "missing_source_file": missing_source,
            "completeness_ratio": float(
                1.0 - (missing_section + missing_content_type + missing_source) / max(total * 3, 1)
            ),
        },
        "redundancy": {
            "exact_duplicate_chunk_count": int(exact_duplicate_chunks),
            "near_duplicate_proxy_count": int(near_duplicate_proxy),
            "exact_duplicate_ratio": float(exact_duplicate_chunks / max(total, 1)),
            "near_duplicate_proxy_ratio": float(near_duplicate_proxy / max(total, 1)),
        },
        "entity_coverage": {
            "chunks_with_entity_refs": int(with_entities),
            "ratio_with_entity_refs": float(with_entities / max(total, 1)),
            "entity_refs_per_chunk_percentiles": _safe_percentiles(entity_ref_counts),
        },
        "type_quality": {
            "table_has_pipe_ratio": float(table_has_pipe_ratio),
            "profile_has_faculty_ref_ratio": float(profile_has_faculty_ref_ratio),
            "regulation_keyword_ratio": float(regulation_keyword_ratio),
            "table_chunk_count": len(table_chunks),
            "profile_chunk_count": len(profile_chunks),
            "regulation_chunk_count": len(regulation_chunks),
        },
    }


def evaluate_embedding_quality(chunks: List[dict], embedding_cache_path: Path) -> dict:
    data = np.load(embedding_cache_path, allow_pickle=True)
    emb = np.asarray(data["embeddings"], dtype=np.float32)
    chunk_ids = data["chunk_ids"].tolist()

    id_to_type = {c.get("chunk_id"): c.get("content_type", "unknown") for c in chunks}
    labels = [id_to_type.get(cid, "unknown") for cid in chunk_ids]

    norms = np.linalg.norm(emb, axis=1)

    # Cosine neighborhood consistency (input embeddings are normalized in pipeline).
    sim = emb @ emb.T
    np.fill_diagonal(sim, -1.0)
    k = 5
    nn_idx = np.argpartition(-sim, kth=k, axis=1)[:, :k]

    same_type_hits = 0
    total_neighbors = emb.shape[0] * k
    outlier_threshold = 0.35
    outlier_indices = []
    mean_neighbor_similarity = []

    for i in range(emb.shape[0]):
        neighbors = nn_idx[i]
        my_type = labels[i]
        neigh_types = [labels[j] for j in neighbors]
        same_type_hits += sum(1 for t in neigh_types if t == my_type)
        neigh_sim = float(np.mean([sim[i, j] for j in neighbors]))
        mean_neighbor_similarity.append(neigh_sim)
        if neigh_sim < outlier_threshold:
            outlier_indices.append(i)

    # Type centroid separability.
    type_vectors: Dict[str, List[np.ndarray]] = defaultdict(list)
    for idx, t in enumerate(labels):
        type_vectors[t].append(emb[idx])

    centroids = {
        t: np.mean(np.vstack(vs), axis=0)
        for t, vs in type_vectors.items()
        if len(vs) >= 3
    }
    centroid_types = sorted(centroids.keys())
    centroid_sim = {}
    for i, t1 in enumerate(centroid_types):
        for t2 in centroid_types[i + 1 :]:
            c1 = centroids[t1]
            c2 = centroids[t2]
            denom = float(np.linalg.norm(c1) * np.linalg.norm(c2))
            if denom == 0.0:
                s = 0.0
            else:
                s = float(np.dot(c1, c2) / denom)
            centroid_sim[f"{t1}__{t2}"] = s

    outlier_chunk_ids = [chunk_ids[i] for i in outlier_indices[:50]]

    return {
        "embedding_count": int(emb.shape[0]),
        "embedding_dimensions": int(emb.shape[1]),
        "norm_stats": {
            "mean": float(np.mean(norms)),
            "std": float(np.std(norms)),
            "min": float(np.min(norms)),
            "max": float(np.max(norms)),
        },
        "neighborhood": {
            "k": k,
            "same_content_type_neighbor_ratio": float(same_type_hits / max(total_neighbors, 1)),
            "mean_neighbor_similarity": float(np.mean(mean_neighbor_similarity)),
            "neighbor_similarity_percentiles": _safe_percentiles(mean_neighbor_similarity),
        },
        "centroid_similarity": centroid_sim,
        "outliers": {
            "threshold_mean_neighbor_similarity": outlier_threshold,
            "count": len(outlier_indices),
            "sample_chunk_ids": outlier_chunk_ids,
        },
    }


def _expected_content_type_for_query_type(query_type: str) -> str | None:
    mapping = {
        "teaching": "knowledge_graph",
        "faculty": "profile",
        "course": "table",
        "timetable": "table",
        "regulation": "regulation",
        "general": None,
    }
    return mapping.get(query_type)


def build_query_set(
    faculty: List[dict],
    courses: List[dict],
    programs: List[dict],
    teaching_assignments: Dict[str, List[str]],
    target_queries: int,
) -> List[EvalQuery]:
    rng = random.Random(42)
    fac_by_id = {f["id"]: f for f in faculty if "id" in f}
    course_by_id = {c["id"]: c for c in courses if "id" in c}

    query_types = ["teaching", "faculty", "course", "timetable", "regulation", "general"]
    base = target_queries // len(query_types)
    rem = target_queries % len(query_types)
    quotas = {qt: base for qt in query_types}
    for qt in query_types[:rem]:
        quotas[qt] += 1

    teaching_pool: List[EvalQuery] = []
    ta_items = list(teaching_assignments.items())
    rng.shuffle(ta_items)
    for fid, course_ids in ta_items:
        fac = fac_by_id.get(fid)
        if not fac:
            continue
        fac_name = fac.get("name", fid)
        for cid in (course_ids or [])[:2]:
            course = course_by_id.get(cid)
            if not course:
                continue
            cname = course.get("name", cid)
            ccode = course.get("code", "")
            teaching_pool.append(
                EvalQuery(
                    text=f"Who teaches {cname} ({ccode})?",
                    query_type="teaching",
                    expected_content_type=_expected_content_type_for_query_type("teaching"),
                    expected_entity_ids=[fid, cid],
                )
            )
            teaching_pool.append(
                EvalQuery(
                    text=f"Which faculty handles {ccode} {cname} with {fac_name}?",
                    query_type="teaching",
                    expected_content_type=_expected_content_type_for_query_type("teaching"),
                    expected_entity_ids=[fid, cid],
                )
            )

    faculty_pool: List[EvalQuery] = []
    for fac in faculty:
        fid = fac.get("id")
        name = fac.get("name", fid)
        faculty_pool.append(
            EvalQuery(
                text=f"What is the profile of {name}?",
                query_type="faculty",
                expected_content_type=_expected_content_type_for_query_type("faculty"),
                expected_entity_ids=[fid] if fid else [],
            )
        )
        faculty_pool.append(
            EvalQuery(
                text=f"Give details about faculty member {name}",
                query_type="faculty",
                expected_content_type=_expected_content_type_for_query_type("faculty"),
                expected_entity_ids=[fid] if fid else [],
            )
        )

    course_pool: List[EvalQuery] = []
    timetable_pool: List[EvalQuery] = []
    for course in courses:
        cid = course.get("id")
        cname = course.get("name", cid)
        ccode = course.get("code", "")
        course_pool.append(
            EvalQuery(
                text=f"Show details for course {ccode} {cname}",
                query_type="course",
                expected_content_type=_expected_content_type_for_query_type("course"),
                expected_entity_ids=[cid] if cid else [],
            )
        )
        course_pool.append(
            EvalQuery(
                text=f"How many credits does {ccode} {cname} have?",
                query_type="course",
                expected_content_type=_expected_content_type_for_query_type("course"),
                expected_entity_ids=[cid] if cid else [],
            )
        )
        timetable_pool.append(
            EvalQuery(
                text=f"What is the timetable or slot for {ccode} {cname}?",
                query_type="timetable",
                expected_content_type=_expected_content_type_for_query_type("timetable"),
                expected_entity_ids=[cid] if cid else [],
            )
        )
        timetable_pool.append(
            EvalQuery(
                text=f"When is {ccode} {cname} scheduled?",
                query_type="timetable",
                expected_content_type=_expected_content_type_for_query_type("timetable"),
                expected_entity_ids=[cid] if cid else [],
            )
        )

    regulation_templates = [
        "What are the grading regulations for B.Tech CSE?",
        "Explain curriculum and scheme under R2023 regulation",
        "What does the regulation say about credits?",
        "Describe attendance and evaluation rules in CSE regulation",
        "What is the regulation for electives and open electives?",
        "Summarize academic regulations for promotion between semesters",
        "What are the exam rules in the current curriculum regulation?",
        "Regulation details for credit requirements in B.Tech AI",
        "How does the grading policy work according to regulation?",
        "Give regulation rules on course registration",
        "What are supplementary exam rules in regulation?",
        "What are audit course rules in regulation?",
    ]
    regulation_pool: List[EvalQuery] = []
    for text in regulation_templates:
        regulation_pool.append(
            EvalQuery(
                text=text,
                query_type="regulation",
                expected_content_type=_expected_content_type_for_query_type("regulation"),
                expected_entity_ids=[programs[0]["id"]] if programs else [],
            )
        )

    general_templates = [
        "List academic resources available in CSE department",
        "What activities are available in computer science engineering department?",
        "Summarize department highlights and facilities",
        "What are placement related opportunities for CSE students?",
        "Tell me about labs and research focus areas in CSE",
        "What student chapters exist in the CSE department?",
        "Give an overview of department advisory board information",
        "Where can I find semester wise syllabus links?",
        "Summarize recent department announcements",
        "What are important documents for CSE students?",
    ]
    general_pool: List[EvalQuery] = []
    for text in general_templates:
        general_pool.append(
            EvalQuery(
                text=text,
                query_type="general",
                expected_content_type=None,
                expected_entity_ids=[],
            )
        )

    pools = {
        "teaching": teaching_pool,
        "faculty": faculty_pool,
        "course": course_pool,
        "timetable": timetable_pool,
        "regulation": regulation_pool,
        "general": general_pool,
    }

    selected: List[EvalQuery] = []
    seen = set()

    for qtype in query_types:
        pool = pools[qtype]
        rng.shuffle(pool)
        for q in pool:
            if quotas[qtype] <= 0:
                break
            if q.text in seen:
                continue
            selected.append(q)
            seen.add(q.text)
            quotas[qtype] -= 1

    # Fill any remaining quota with synthetic teaching/course variants.
    while len(selected) < target_queries:
        fac = rng.choice(faculty)
        course = rng.choice(courses)
        fid = fac.get("id")
        cid = course.get("id")
        q = EvalQuery(
            text=f"Who is the instructor for {course.get('code', '')} {course.get('name', '')}?",
            query_type="teaching",
            expected_content_type=_expected_content_type_for_query_type("teaching"),
            expected_entity_ids=[fid, cid] if fid and cid else [],
        )
        if q.text in seen:
            continue
        selected.append(q)
        seen.add(q.text)

    return selected[:target_queries]


def _dcg(relevances: List[float]) -> float:
    return float(sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances)))


def _relevance_grade(
    query: EvalQuery,
    result_metadata: dict,
    result_doc: str,
) -> int:
    grade = 0
    expected_type = query.expected_content_type
    got_type = result_metadata.get("content_type")
    if expected_type and got_type == expected_type:
        grade += 1

    expected_entities = set(query.expected_entity_ids or [])
    result_entities = set(_parse_entity_refs(result_metadata.get("entity_refs")))
    if expected_entities and (expected_entities & result_entities):
        grade += 2

    # Semantic overlap proxy using query tokens in result document.
    q_tokens = set(_tokenize(query.text))
    d_tokens = set(_tokenize(result_doc or ""))
    overlap = len(q_tokens & d_tokens)
    if overlap >= 2:
        grade += 1

    return grade


def evaluate_retrieval_quality(
    queries: List[EvalQuery],
    collection,
    top_k: int,
    device: str,
) -> dict:
    per_query_logs = []
    mrr_vals = []
    ndcg_vals = []
    hit_vals = []
    precision_vals = []
    recall_proxy_vals = []
    latency_vals = []

    per_type_counts = Counter()
    per_type_hits = Counter()
    per_type_failures = Counter()
    route_mismatch = Counter()
    fallback_trigger_count = 0

    for query in queries:
        per_type_counts[query.query_type] += 1
        started = time.perf_counter()

        # Use the same routed+fallback retrieval path as runtime to avoid evaluator/runtime mismatch.
        result = query_chromadb_with_fallback(
            collection,
            query_text=query.text,
            query_type=query.query_type,
            n_results=top_k,
            enable_fallback=True,
            rerank_mixed=True,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        latency_vals.append(latency_ms)

        if result.get("fallback_triggered", False):
            fallback_trigger_count += 1

        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        grades = []
        relevant_positions = []
        got_types = []
        relevance_cutoff = 1 if query.query_type == "general" else 2

        for i, (doc, meta) in enumerate(zip(docs, metas)):
            meta = meta or {}
            g = _relevance_grade(query, meta, doc or "")
            grades.append(g)
            got_type = meta.get("content_type", "unknown")
            got_types.append(got_type)

            if g >= relevance_cutoff:
                relevant_positions.append(i + 1)

            if query.expected_content_type and got_type != query.expected_content_type:
                route_mismatch[f"{query.query_type}->{got_type}"] += 1

        hit = 1.0 if relevant_positions else 0.0
        hit_vals.append(hit)
        if hit > 0:
            per_type_hits[query.query_type] += 1
        else:
            per_type_failures[query.query_type] += 1

        if relevant_positions:
            mrr_vals.append(1.0 / min(relevant_positions))
        else:
            mrr_vals.append(0.0)

        # Binary precision@k proxy with query-type-aware relevance cutoff.
        rel_count = sum(1 for g in grades if g >= relevance_cutoff)
        precision_vals.append(rel_count / max(len(grades), 1))

        # Recall proxy: has at least one expected entity hit in top-k.
        expected_entities = set(query.expected_entity_ids or [])
        if expected_entities:
            has_entity_hit = False
            for meta in metas:
                got_entities = set(_parse_entity_refs((meta or {}).get("entity_refs")))
                if expected_entities & got_entities:
                    has_entity_hit = True
                    break
            recall_proxy_vals.append(1.0 if has_entity_hit else 0.0)
        else:
            recall_proxy_vals.append(hit)

        # NDCG using graded relevance.
        dcg = _dcg(grades)
        ideal = sorted(grades, reverse=True)
        idcg = _dcg(ideal)
        ndcg = (dcg / idcg) if idcg > 0 else 0.0
        ndcg_vals.append(ndcg)

        per_query_logs.append(
            {
                "query": query.text,
                "query_type": query.query_type,
                "expected_content_type": query.expected_content_type,
                "expected_entity_ids": query.expected_entity_ids or [],
                "top_k": len(docs),
                "fallback_triggered": bool(result.get("fallback_triggered", False)),
                "top_distances": [float(x) for x in dists],
                "top_content_types": got_types,
                "relevance_grades": grades,
                "hit": bool(hit),
                "mrr": float(mrr_vals[-1]),
                "ndcg": float(ndcg),
                "latency_ms": float(latency_ms),
            }
        )

    per_type_metrics = {}
    for t, n in sorted(per_type_counts.items()):
        h = per_type_hits[t]
        f = per_type_failures[t]
        per_type_metrics[t] = {
            "query_count": int(n),
            "hit_rate": float(h / max(n, 1)),
            "failure_rate": float(f / max(n, 1)),
        }

    return {
        "query_count": len(queries),
        "top_k": top_k,
        "aggregate_metrics": {
            "hit_at_k": float(sum(hit_vals) / max(len(hit_vals), 1)),
            "mrr": float(sum(mrr_vals) / max(len(mrr_vals), 1)),
            "ndcg_at_k": float(sum(ndcg_vals) / max(len(ndcg_vals), 1)),
            "precision_at_k_proxy": float(sum(precision_vals) / max(len(precision_vals), 1)),
            "recall_at_k_proxy": float(sum(recall_proxy_vals) / max(len(recall_proxy_vals), 1)),
            "fallback_trigger_ratio": float(fallback_trigger_count / max(len(queries), 1)),
            "latency_ms": {
                "mean": float(sum(latency_vals) / max(len(latency_vals), 1)),
                "min": float(min(latency_vals) if latency_vals else 0.0),
                "max": float(max(latency_vals) if latency_vals else 0.0),
                "percentiles": _safe_percentiles(latency_vals),
            },
        },
        "per_query_type": per_type_metrics,
        "routing_mismatch_counter": dict(sorted(route_mismatch.items())),
        "per_query_logs": per_query_logs,
    }


def evaluate_kg_quality(
    faculty: List[dict],
    courses: List[dict],
    programs: List[dict],
    chunks: List[dict],
) -> dict:
    graph, report = build_knowledge_graph(faculty, courses, programs, chunks)
    errors = validate_graph(graph)

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    node_ids = {n.get("id") for n in nodes}
    incident = set()
    evidence_lengths = []

    for e in edges:
        incident.add(e.get("source"))
        incident.add(e.get("target"))
        evidence_lengths.append(len(e.get("evidence", []) or []))

    orphan_nodes = [n for n in nodes if n.get("id") not in incident]

    # Connectivity over undirected adjacency.
    adjacency = defaultdict(set)
    for e in edges:
        s = e.get("source")
        t = e.get("target")
        if s in node_ids and t in node_ids:
            adjacency[s].add(t)
            adjacency[t].add(s)

    seen = set()
    components = []
    for nid in node_ids:
        if nid in seen:
            continue
        stack = [nid]
        comp = []
        seen.add(nid)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in adjacency[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(comp)

    largest_component = max((len(c) for c in components), default=0)

    # Coverage proxies.
    course_ids = {c.get("id") for c in courses if c.get("id")}
    faculty_ids = {f.get("id") for f in faculty if f.get("id")}

    teaches_edges = [e for e in edges if e.get("type") == "teaches"]
    part_of_edges = [e for e in edges if e.get("type") == "part_of"]
    prereq_edges = [e for e in edges if e.get("type") == "has_prerequisite"]

    courses_with_part_of = {e.get("source") for e in part_of_edges}
    faculty_with_teaches = {e.get("source") for e in teaches_edges}

    return {
        "build_report": report,
        "validator_errors": errors,
        "graph_structure": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "orphan_node_count": len(orphan_nodes),
            "orphan_node_ratio": float(len(orphan_nodes) / max(len(nodes), 1)),
            "connected_component_count": len(components),
            "largest_component_ratio": float(largest_component / max(len(nodes), 1)),
        },
        "evidence_quality": {
            "mean_evidence_per_edge": float(statistics.mean(evidence_lengths) if evidence_lengths else 0.0),
            "min_evidence_per_edge": int(min(evidence_lengths) if evidence_lengths else 0),
            "max_evidence_per_edge": int(max(evidence_lengths) if evidence_lengths else 0),
        },
        "coverage": {
            "courses_with_program_link_ratio": float(len(courses_with_part_of) / max(len(course_ids), 1)),
            "faculty_with_teaching_link_ratio": float(len(faculty_with_teaches) / max(len(faculty_ids), 1)),
            "prerequisite_edge_count": len(prereq_edges),
            "teaches_edge_count": len(teaches_edges),
            "part_of_edge_count": len(part_of_edges),
        },
    }


def build_improvement_backlog(
    chunk_metrics: dict,
    embedding_metrics: dict,
    retrieval_metrics: dict,
    kg_metrics: dict,
) -> List[dict]:
    backlog = []

    if retrieval_metrics["aggregate_metrics"]["hit_at_k"] < 0.8:
        backlog.append(
            {
                "priority": "P1",
                "area": "retrieval",
                "issue": "Hit@k below target",
                "change": "Retune route-specific thresholds and add fallback broad retrieval when filtered results are empty.",
                "success_metric": "Hit@k >= 0.85 and MRR >= 0.65 on 75-query benchmark",
            }
        )

    if retrieval_metrics["aggregate_metrics"]["latency_ms"]["mean"] > 1500:
        backlog.append(
            {
                "priority": "P1",
                "area": "retrieval/perf",
                "issue": "High average retrieval latency",
                "change": "Persist loaded embedding model during multi-query evaluation and avoid per-query model initialization in runtime query path.",
                "success_metric": "Mean query latency < 900ms on same hardware",
            }
        )

    if chunk_metrics["redundancy"]["near_duplicate_proxy_ratio"] > 0.2:
        backlog.append(
            {
                "priority": "P2",
                "area": "chunking",
                "issue": "High near-duplicate proxy ratio",
                "change": "Adjust table chunk split boundaries and deduplicate repetitive rows/headers before chunk emission.",
                "success_metric": "Near-duplicate proxy ratio reduced by >= 30%",
            }
        )

    if embedding_metrics["neighborhood"]["same_content_type_neighbor_ratio"] < 0.55:
        backlog.append(
            {
                "priority": "P2",
                "area": "embeddings",
                "issue": "Weak type-local embedding neighborhoods",
                "change": "Improve chunk semantic coherence and rebalance long table chunks; consider chunk-type-aware query expansion.",
                "success_metric": "Same-type neighbor ratio >= 0.65",
            }
        )

    if kg_metrics["coverage"]["prerequisite_edge_count"] == 0:
        backlog.append(
            {
                "priority": "P1",
                "area": "knowledge_graph",
                "issue": "No prerequisite edges extracted",
                "change": "Constrain prerequisite extraction to nearest course heading context and resolve ambiguous source course with local window rules.",
                "success_metric": "Non-zero prerequisite edges with manual precision >= 0.85 on sampled edges",
            }
        )

    if kg_metrics["graph_structure"]["orphan_node_ratio"] > 0.15:
        backlog.append(
            {
                "priority": "P2",
                "area": "knowledge_graph",
                "issue": "High orphan node ratio",
                "change": "Add additional relation rules for semester and course-to-course linkage using section hierarchy cues.",
                "success_metric": "Orphan node ratio <= 0.10",
            }
        )

    if not backlog:
        backlog.append(
            {
                "priority": "P3",
                "area": "general",
                "issue": "No critical issues found in wide run",
                "change": "Continue with periodic benchmark regression runs and increase manually labeled query set coverage.",
                "success_metric": "Metric stability within +/- 5% across runs",
            }
        )

    return backlog


def build_markdown_summary(
    started_at: str,
    chunk_metrics: dict,
    embedding_metrics: dict,
    retrieval_metrics: dict,
    kg_metrics: dict,
    backlog: List[dict],
) -> str:
    lines = []
    lines.append("# Wide Quality Evaluation Summary")
    lines.append("")
    lines.append(f"Generated at (UTC): {started_at}")
    lines.append("")

    lines.append("## Chunk Quality")
    lines.append(f"- Total chunks: {chunk_metrics['total_chunks']}")
    lines.append(f"- Word count mean / median: {chunk_metrics['word_count']['mean']:.2f} / {chunk_metrics['word_count']['median']:.2f}")
    lines.append(f"- Metadata completeness ratio: {chunk_metrics['metadata_completeness']['completeness_ratio']:.3f}")
    lines.append(f"- Exact duplicate ratio: {chunk_metrics['redundancy']['exact_duplicate_ratio']:.3f}")
    lines.append(f"- Near-duplicate proxy ratio: {chunk_metrics['redundancy']['near_duplicate_proxy_ratio']:.3f}")
    lines.append("")

    lines.append("## Embedding Quality")
    lines.append(f"- Embeddings: {embedding_metrics['embedding_count']} x {embedding_metrics['embedding_dimensions']}")
    lines.append(f"- Norm mean/std: {embedding_metrics['norm_stats']['mean']:.4f} / {embedding_metrics['norm_stats']['std']:.4f}")
    lines.append(f"- Same-type neighbor ratio (k={embedding_metrics['neighborhood']['k']}): {embedding_metrics['neighborhood']['same_content_type_neighbor_ratio']:.3f}")
    lines.append(f"- Outlier count: {embedding_metrics['outliers']['count']}")
    lines.append("")

    lines.append("## Retrieval Quality")
    ag = retrieval_metrics["aggregate_metrics"]
    lines.append(f"- Query count: {retrieval_metrics['query_count']}")
    lines.append(f"- Hit@{retrieval_metrics['top_k']}: {ag['hit_at_k']:.3f}")
    lines.append(f"- MRR: {ag['mrr']:.3f}")
    lines.append(f"- NDCG@{retrieval_metrics['top_k']}: {ag['ndcg_at_k']:.3f}")
    lines.append(f"- Precision@{retrieval_metrics['top_k']} proxy: {ag['precision_at_k_proxy']:.3f}")
    lines.append(f"- Recall@{retrieval_metrics['top_k']} proxy: {ag['recall_at_k_proxy']:.3f}")
    lines.append(f"- Latency mean/min/max (ms): {ag['latency_ms']['mean']:.2f} / {ag['latency_ms']['min']:.2f} / {ag['latency_ms']['max']:.2f}")
    lines.append("")

    lines.append("## Knowledge Graph Quality")
    lines.append(f"- Nodes / Edges: {kg_metrics['graph_structure']['node_count']} / {kg_metrics['graph_structure']['edge_count']}")
    lines.append(f"- Orphan node ratio: {kg_metrics['graph_structure']['orphan_node_ratio']:.3f}")
    lines.append(f"- Connected components: {kg_metrics['graph_structure']['connected_component_count']}")
    lines.append(f"- Largest component ratio: {kg_metrics['graph_structure']['largest_component_ratio']:.3f}")
    lines.append(f"- Prerequisite edges: {kg_metrics['coverage']['prerequisite_edge_count']}")
    lines.append("")

    lines.append("## Improvement Backlog")
    for item in backlog:
        lines.append(f"- [{item['priority']}] {item['area']}: {item['issue']}")
        lines.append(f"  Change: {item['change']}")
        lines.append(f"  Success metric: {item['success_metric']}")

    lines.append("")
    return "\n".join(lines)


def open_collection():
    import chromadb

    candidates = [config.CHROMADB_DIR, config.DATA_DIR / "chroma_db"]
    last_error = None

    for base in candidates:
        try:
            client = chromadb.PersistentClient(path=str(base))
            coll = client.get_collection(config.CHROMADB_COLLECTION)
            return coll, str(base)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Failed to open ChromaDB collection from candidates {candidates}: {last_error}")


def main():
    parser = argparse.ArgumentParser(description="Run wide quality evaluation")
    parser.add_argument("--queries", type=int, default=75, help="Number of retrieval queries")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k retrieval depth")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Embedding device")
    args = parser.parse_args()

    started_at = datetime.now(timezone.utc).isoformat()
    print("=" * 72)
    print("WIDE QUALITY EVALUATION")
    print("=" * 72)
    print(f"Start time (UTC): {started_at}")

    chunks = _load_json(config.CHUNKS_FILE)
    faculty = _load_json(config.FACULTY_FILE)
    courses = _load_json(config.COURSES_FILE)
    programs = _load_json(config.PROGRAMS_FILE)
    teaching_assignments = _load_json(config.TEACHING_ASSIGNMENTS_FILE)

    print("\n[1/4] Evaluating chunk quality...")
    chunk_metrics = evaluate_chunk_quality(chunks)

    print("[2/4] Evaluating embedding quality...")
    chunk_cache_path = config.EMBEDDING_CACHE_FILE
    embedding_metrics = evaluate_embedding_quality(chunks, chunk_cache_path)

    print("[3/4] Evaluating retrieval quality...")
    queries = build_query_set(
        faculty=faculty,
        courses=courses,
        programs=programs,
        teaching_assignments=teaching_assignments,
        target_queries=args.queries,
    )

    collection, collection_path = open_collection()
    print(f"Using ChromaDB path: {collection_path}")
    retrieval_metrics = evaluate_retrieval_quality(
        queries=queries,
        collection=collection,
        top_k=args.top_k,
        device=args.device,
    )

    print("[4/4] Evaluating knowledge graph quality...")
    kg_metrics = evaluate_kg_quality(
        faculty=faculty,
        courses=courses,
        programs=programs,
        chunks=chunks,
    )

    backlog = build_improvement_backlog(
        chunk_metrics=chunk_metrics,
        embedding_metrics=embedding_metrics,
        retrieval_metrics=retrieval_metrics,
        kg_metrics=kg_metrics,
    )

    validation_dir = config.VALIDATION_DIR
    _write_json(validation_dir / "chunk_quality_metrics.json", chunk_metrics)
    _write_json(validation_dir / "embedding_quality_metrics.json", embedding_metrics)
    _write_json(validation_dir / "retrieval_metrics.json", retrieval_metrics)
    _write_json(validation_dir / "kg_quality_metrics.json", kg_metrics)
    _write_json(validation_dir / "improvement_backlog.json", {"items": backlog})

    summary_md = build_markdown_summary(
        started_at=started_at,
        chunk_metrics=chunk_metrics,
        embedding_metrics=embedding_metrics,
        retrieval_metrics=retrieval_metrics,
        kg_metrics=kg_metrics,
        backlog=backlog,
    )
    _write_text(validation_dir / "wide_eval_summary.md", summary_md)

    improvement_md = ["# Improvement Backlog", ""]
    for item in backlog:
        improvement_md.append(f"## {item['priority']} - {item['area']}")
        improvement_md.append(f"- Issue: {item['issue']}")
        improvement_md.append(f"- Recommended change: {item['change']}")
        improvement_md.append(f"- Success metric: {item['success_metric']}")
        improvement_md.append("")
    _write_text(validation_dir / "improvement_backlog.md", "\n".join(improvement_md))

    print("\n" + "=" * 72)
    print("EVALUATION COMPLETE")
    print("=" * 72)
    print(f"Chunk total: {chunk_metrics['total_chunks']}")
    print(f"Retrieval Hit@{args.top_k}: {retrieval_metrics['aggregate_metrics']['hit_at_k']:.3f}")
    print(f"Retrieval MRR: {retrieval_metrics['aggregate_metrics']['mrr']:.3f}")
    print(f"KG prerequisite edges: {kg_metrics['coverage']['prerequisite_edge_count']}")
    print(f"Reports written to: {validation_dir}")


if __name__ == "__main__":
    main()
