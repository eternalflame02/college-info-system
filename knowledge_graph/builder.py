"""
Phase-1 knowledge graph builder.

Scope:
- JSON output only
- Deterministic, high-confidence edges only
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import config


def _load_json(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return []


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def _contains_alias(line: str, aliases: List[str]) -> bool:
    ln = _normalize(line)
    for alias in aliases:
        a = _normalize(alias)
        if a and a in ln:
            return True
    return False


def _edge_id(edge_type: str, source: str, target: str) -> str:
    base = f"{edge_type}|{source}|{target}"
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:12]
    return f"edge_{edge_type}_{source}_{target}_{digest}"


def _course_code_tokens(courses: List[dict]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for course in courses:
        code = (course.get("code") or "").strip().upper()
        cid = course.get("id", "")
        if code and cid:
            mapping[code] = cid
    return mapping


def _program_from_source(source_file: str) -> Optional[str]:
    source = source_file.lower()

    # Order matters: specific before generic
    if any(k in source for k in ["m-tech", "mtech", "m_tech"]):
        return "prog_mtech_cse"

    if any(k in source for k in ["cse_ai", "ai-b-tech", "cseai", "s5-ct", "s7-ct", "s3-ct"]):
        return "prog_btech_ai"

    if any(k in source for k in ["-cs", "_cs", "cse_", "cse-"]):
        return "prog_btech_cse"

    return None


def build_nodes(
    faculty: List[dict],
    courses: List[dict],
    programs: List[dict],
    chunks: List[dict],
) -> List[dict]:
    nodes: Dict[str, dict] = {}

    def add_node(entity: dict, source_ref: str):
        node_id = entity.get("id")
        if not node_id:
            return
        aliases = list(dict.fromkeys([a for a in entity.get("aliases", []) if a]))
        node = {
            "id": node_id,
            "type": entity.get("type", ""),
            "name": entity.get("name", ""),
            "aliases": aliases,
            "source_refs": [source_ref],
        }
        if node_id in nodes:
            merged_aliases = list(dict.fromkeys(nodes[node_id]["aliases"] + aliases))
            merged_refs = list(dict.fromkeys(nodes[node_id]["source_refs"] + [source_ref]))
            nodes[node_id]["aliases"] = merged_aliases
            nodes[node_id]["source_refs"] = merged_refs
        else:
            nodes[node_id] = node

    for e in faculty:
        add_node(e, str(config.FACULTY_FILE))
    for e in courses:
        add_node(e, str(config.COURSES_FILE))
    for e in programs:
        add_node(e, str(config.PROGRAMS_FILE))

    semester_ids = set()
    for chunk in chunks:
        for ref in chunk.get("entity_refs", []):
            if re.match(r"^semester_[1-8]$", ref):
                semester_ids.add(ref)

    for sid in sorted(semester_ids):
        sem_num = sid.split("_")[-1]
        nodes[sid] = {
            "id": sid,
            "type": "semester",
            "name": f"Semester {sem_num}",
            "aliases": [f"S{sem_num}", f"Semester {sem_num}"],
            "source_refs": [str(config.CHUNKS_FILE)],
        }

    return sorted(nodes.values(), key=lambda n: n["id"])


def extract_course_part_of_program_edges(
    chunks: List[dict],
    valid_program_ids: set,
) -> Tuple[List[dict], Dict[str, int]]:
    edges: Dict[str, dict] = {}
    rejected = defaultdict(int)

    for chunk in chunks:
        refs = set(chunk.get("entity_refs", []))
        courses = sorted([r for r in refs if r.startswith("course_")])
        if not courses:
            continue

        explicit_programs = [r for r in refs if r in valid_program_ids]
        program_id: Optional[str] = None

        if len(explicit_programs) == 1:
            program_id = explicit_programs[0]
        elif len(explicit_programs) > 1:
            rejected["part_of_ambiguous_programs"] += 1
            continue
        else:
            program_id = _program_from_source(chunk.get("source_file", ""))

        if not program_id or program_id not in valid_program_ids:
            rejected["part_of_no_program"] += 1
            continue

        for course_id in courses:
            edge = {
                "id": _edge_id("part_of", course_id, program_id),
                "type": "part_of",
                "source": course_id,
                "target": program_id,
                "confidence": 1.0,
                "deterministic": True,
                "evidence": [
                    f"chunk_id:{chunk.get('chunk_id', '')}",
                    f"source_file:{chunk.get('source_file', '')}",
                    "rule:course_source_program_mapping",
                ],
            }
            edges[edge["id"]] = edge

    return sorted(edges.values(), key=lambda e: e["id"]), dict(rejected)


def extract_prerequisite_edges(
    chunks: List[dict],
    code_to_course_id: Dict[str, str],
) -> Tuple[List[dict], Dict[str, int]]:
    edges: Dict[str, dict] = {}
    rejected = defaultdict(int)

    prereq_pat = re.compile(r"pre[\s-]?requisites?\s*[:\-]\s*([^\n]+)", re.IGNORECASE)
    code_pat = re.compile(r"\b[A-Z0-9]{3,10}\b")

    for chunk in chunks:
        refs = set(chunk.get("entity_refs", []))
        source_courses = sorted([r for r in refs if r.startswith("course_")])
        if len(source_courses) != 1:
            if len(source_courses) > 1:
                rejected["prereq_ambiguous_source_course"] += 1
            continue

        source_course = source_courses[0]
        text = chunk.get("text", "")
        matches = prereq_pat.findall(text)
        if not matches:
            continue

        for matched in matches:
            found = False
            for token in code_pat.findall(matched.upper()):
                if not (any(c.isalpha() for c in token) and any(c.isdigit() for c in token)):
                    continue
                target_course = code_to_course_id.get(token)
                if not target_course or target_course == source_course:
                    continue

                edge = {
                    "id": _edge_id("has_prerequisite", source_course, target_course),
                    "type": "has_prerequisite",
                    "source": source_course,
                    "target": target_course,
                    "confidence": 1.0,
                    "deterministic": True,
                    "evidence": [
                        f"chunk_id:{chunk.get('chunk_id', '')}",
                        f"source_file:{chunk.get('source_file', '')}",
                        f"text_span:Prerequisite:{matched.strip()}",
                    ],
                }
                edges[edge["id"]] = edge
                found = True
            if not found:
                rejected["prereq_no_mapped_code"] += 1

    return sorted(edges.values(), key=lambda e: e["id"]), dict(rejected)


def extract_teaches_edges(
    chunks: List[dict],
    faculty: List[dict],
    courses: List[dict],
) -> Tuple[List[dict], Dict[str, int]]:
    edges: Dict[str, dict] = {}
    rejected = defaultdict(int)

    faculty_matchers = []
    for f in faculty:
        aliases = list(dict.fromkeys([f.get("name", "")] + f.get("aliases", [])))
        faculty_matchers.append((f.get("id", ""), aliases))

    course_matchers = []
    for c in courses:
        aliases = [c.get("name", "")]
        code = c.get("code", "")
        if code:
            aliases.append(code)
        course_matchers.append((c.get("id", ""), [a for a in aliases if a]))

    cue_words = [re.escape(word) for word in config.TEACHES_ASSIGNMENT_CUES]
    cue_pat = re.compile(rf"\b({'|'.join(cue_words)})\b", re.IGNORECASE)

    for chunk in chunks:
        lines = [ln.strip() for ln in chunk.get("text", "").splitlines() if ln.strip()]
        for line in lines:
            matched_faculty = [fid for fid, aliases in faculty_matchers if _contains_alias(line, aliases)]
            matched_courses = [cid for cid, aliases in course_matchers if _contains_alias(line, aliases)]

            if not matched_faculty or not matched_courses:
                continue

            has_table_row_signal = line.count("|") >= 2
            has_assignment_signal = bool(cue_pat.search(line))
            if not (has_table_row_signal or has_assignment_signal):
                rejected["teaches_no_assignment_signal"] += 1
                continue

            for fid in sorted(set(matched_faculty)):
                for cid in sorted(set(matched_courses)):
                    edge = {
                        "id": _edge_id("teaches", fid, cid),
                        "type": "teaches",
                        "source": fid,
                        "target": cid,
                        "confidence": 1.0,
                        "deterministic": True,
                        "evidence": [
                            f"chunk_id:{chunk.get('chunk_id', '')}",
                            f"source_file:{chunk.get('source_file', '')}",
                            f"text_span:{line[:200]}",
                        ],
                    }
                    edges[edge["id"]] = edge

    return sorted(edges.values(), key=lambda e: e["id"]), dict(rejected)


def validate_graph(graph: dict) -> List[str]:
    errors: List[str] = []

    for key in ["version", "generated_at", "nodes", "edges"]:
        if key not in graph:
            errors.append(f"Missing top-level key: {key}")

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    node_ids = set()
    for node in nodes:
        for key in ["id", "type", "name", "aliases", "source_refs"]:
            if key not in node:
                errors.append(f"Node missing key '{key}': {node}")
        nid = node.get("id")
        if not nid:
            errors.append(f"Node with empty id: {node}")
            continue
        if nid in node_ids:
            errors.append(f"Duplicate node id: {nid}")
        node_ids.add(nid)

    edge_ids = set()
    for edge in edges:
        for key in ["id", "type", "source", "target", "confidence", "deterministic", "evidence"]:
            if key not in edge:
                errors.append(f"Edge missing key '{key}': {edge}")
        eid = edge.get("id")
        if not eid:
            errors.append(f"Edge with empty id: {edge}")
            continue
        if eid in edge_ids:
            errors.append(f"Duplicate edge id: {eid}")
        edge_ids.add(eid)

        if edge.get("confidence") != 1.0:
            errors.append(f"Non-deterministic confidence for edge {eid}")
        if edge.get("deterministic") is not True:
            errors.append(f"Edge deterministic flag must be true: {eid}")
        if not edge.get("evidence"):
            errors.append(f"Edge evidence missing: {eid}")

        source = edge.get("source")
        target = edge.get("target")
        if source not in node_ids:
            errors.append(f"Dangling edge source '{source}' in edge {eid}")
        if target not in node_ids:
            errors.append(f"Dangling edge target '{target}' in edge {eid}")

    return errors


def build_knowledge_graph(
    faculty: List[dict],
    courses: List[dict],
    programs: List[dict],
    chunks: List[dict],
) -> Tuple[dict, dict]:
    nodes = build_nodes(faculty, courses, programs, chunks)
    node_ids = {n["id"] for n in nodes}
    valid_program_ids = {p["id"] for p in programs if p.get("id")}

    part_of_edges, part_of_rejected = extract_course_part_of_program_edges(
        chunks, valid_program_ids
    )
    prereq_edges, prereq_rejected = extract_prerequisite_edges(
        chunks, _course_code_tokens(courses)
    )
    teaches_edges, teaches_rejected = extract_teaches_edges(chunks, faculty, courses)

    all_edges_map = {}
    for edge in part_of_edges + prereq_edges + teaches_edges:
        all_edges_map[edge["id"]] = edge
    edges = sorted(all_edges_map.values(), key=lambda e: e["id"])

    graph = {
        "version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "nodes": nodes,
        "edges": edges,
    }

    errors = validate_graph(graph)

    nodes_by_type = defaultdict(int)
    for node in nodes:
        nodes_by_type[node["type"]] += 1

    edges_by_type = defaultdict(int)
    for edge in edges:
        edges_by_type[edge["type"]] += 1

    report = {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "nodes_by_type": dict(sorted(nodes_by_type.items())),
        "edges_by_type": dict(sorted(edges_by_type.items())),
        "rejected_candidates": {
            "part_of": part_of_rejected,
            "has_prerequisite": prereq_rejected,
            "teaches": teaches_rejected,
        },
        "validation_errors": errors,
        "valid": len(errors) == 0,
        "known_node_ids": len(node_ids),
    }

    return graph, report


def run_knowledge_graph_pipeline() -> Tuple[dict, dict]:
    config.ensure_directories()

    faculty = _load_json(config.FACULTY_FILE)
    courses = _load_json(config.COURSES_FILE)
    programs = _load_json(config.PROGRAMS_FILE)
    chunks = _load_json(config.CHUNKS_FILE)

    graph, report = build_knowledge_graph(faculty, courses, programs, chunks)

    config.KNOWLEDGE_GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.KNOWLEDGE_GRAPH_FILE, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)

    with open(config.KNOWLEDGE_GRAPH_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("🕸 Knowledge graph construction complete")
    print("=" * 50)
    print(f"Nodes: {report['total_nodes']}")
    print(f"Edges: {report['total_edges']}")
    print(f"Valid: {report['valid']}")
    print(f"Graph: {config.KNOWLEDGE_GRAPH_FILE}")
    print(f"Report: {config.KNOWLEDGE_GRAPH_REPORT_FILE}")
    print("=" * 50)

    return graph, report


if __name__ == "__main__":
    run_knowledge_graph_pipeline()
