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
from typing import Dict, List, Optional, Set, Tuple

import config


def _load_json(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return []


def _load_json_dict(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    return {}


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


def _course_id_to_code(courses: List[dict]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for course in courses:
        cid = course.get("id", "")
        code = (course.get("code") or "").strip().upper()
        if cid and code:
            mapping[cid] = code
    return mapping


def _course_alias_to_id(courses: List[dict]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for course in courses:
        cid = course.get("id", "")
        if not cid:
            continue
        code = (course.get("code") or "").strip().upper()
        name = _normalize(course.get("name", ""))
        aliases = [_normalize(a) for a in course.get("aliases", [])]

        if code:
            mapping[code] = cid
        if name:
            mapping[name] = cid
        for alias in aliases:
            if alias:
                mapping[alias] = cid
    return mapping


def _faculty_alias_to_id(faculty: List[dict]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for member in faculty:
        fid = member.get("id", "")
        if not fid:
            continue

        names: List[str] = []
        primary = member.get("name", "")
        if isinstance(primary, str) and primary.strip():
            names.append(primary)
        aliases = member.get("aliases", [])
        if isinstance(aliases, list):
            names.extend([a for a in aliases if isinstance(a, str) and a.strip()])

        for alias in names:
            key = _normalize(alias)
            if key and key not in mapping:
                mapping[key] = fid
    return mapping


def _extract_candidate_faculty_names(text: str) -> List[str]:
    return re.findall(
        r"(?:dr|prof|mr|ms|mrs)\.?\s+[a-z]+(?:\s+[a-z]+){0,3}",
        text,
        flags=re.IGNORECASE,
    )


def _normalize_assignments(
    raw_assignments: Dict,
    valid_faculty_ids: Set[str],
    valid_course_ids: Set[str],
) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    cleaned: Dict[str, Set[str]] = defaultdict(set)
    rejected = defaultdict(int)

    for faculty_id, course_ids in raw_assignments.items():
        if faculty_id not in valid_faculty_ids:
            rejected["manual_unknown_faculty"] += 1
            continue
        if not isinstance(course_ids, list):
            rejected["manual_invalid_course_list"] += 1
            continue

        for course_id in course_ids:
            if not isinstance(course_id, str):
                rejected["manual_invalid_course_entry"] += 1
                continue
            if course_id not in valid_course_ids:
                rejected["manual_unknown_course"] += 1
                continue
            cleaned[faculty_id].add(course_id)

    normalized = {
        fid: sorted(course_set)
        for fid, course_set in cleaned.items()
    }
    return normalized, dict(rejected)


def extract_timetable_teaching_links(
    chunks: List[dict],
    faculty: List[dict],
    courses: List[dict],
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, int]]:
    """
    Extract deterministic faculty-course links from timetable chunks.

    Returns:
        assignments_from_timetable: faculty_id -> {course_ids}
        pair_sources: "faculty|course" -> {"timetable"}
        audit: extraction stats including unresolved signals
    """
    code_to_course = _course_code_tokens(courses)
    faculty_alias_map = _faculty_alias_to_id(faculty)

    assignments: Dict[str, Set[str]] = defaultdict(set)
    pair_sources: Dict[str, Set[str]] = defaultdict(set)
    audit = defaultdict(int)
    code_pat = re.compile(r"\b[A-Z0-9]{5,10}\b")

    for chunk in chunks:
        content_type = chunk.get("content_type")
        metadata = chunk.get("metadata", {}) or {}
        table_kind = metadata.get("table_kind", "")
        is_timetable = (
            content_type == "table"
            and (
                table_kind == "timetable"
                or metadata.get("timetable_signal") == "true"
                or "timetable" in _normalize(" ".join(chunk.get("section_hierarchy", [])))
            )
        )
        if not is_timetable:
            continue

        audit["timetable_chunks_seen"] += 1
        refs = set(chunk.get("entity_refs", []))

        course_ids = {ref for ref in refs if isinstance(ref, str) and ref.startswith("course_")}
        course_ids_meta = [token.strip() for token in str(metadata.get("timetable_course_ids", "")).split(",") if token.strip()]
        for cid in course_ids_meta:
            if cid in code_to_course.values():
                course_ids.add(cid)
            else:
                audit["timetable_unmapped_course_ids"] += 1

        for token in code_pat.findall(chunk.get("text", "").upper()):
            mapped = code_to_course.get(token)
            if mapped:
                course_ids.add(mapped)

        faculty_ids = {ref for ref in refs if isinstance(ref, str) and ref.startswith("faculty_")}
        faculty_ids_meta = [token.strip() for token in str(metadata.get("timetable_faculty_ids", "")).split(",") if token.strip()]
        for fid in faculty_ids_meta:
            if fid in faculty_alias_map.values():
                faculty_ids.add(fid)
            else:
                audit["timetable_unmapped_faculty_ids"] += 1

        for token in _extract_candidate_faculty_names(chunk.get("text", "")):
            mapped = faculty_alias_map.get(_normalize(token))
            if mapped:
                faculty_ids.add(mapped)
            else:
                audit["timetable_unmatched_faculty_names"] += 1

        if not faculty_ids:
            audit["timetable_chunks_without_faculty"] += 1
            continue
        if not course_ids:
            audit["timetable_chunks_without_course"] += 1
            continue

        for fid in sorted(faculty_ids):
            for cid in sorted(course_ids):
                assignments[fid].add(cid)
                pair_key = f"{fid}|{cid}"
                pair_sources[pair_key].add("timetable")
                audit["timetable_pairs_added"] += 1

    return assignments, pair_sources, dict(audit)


def merge_teaching_assignments(
    manual_assignments: Dict[str, List[str]],
    timetable_assignments: Dict[str, Set[str]],
    valid_faculty_ids: Set[str],
    valid_course_ids: Set[str],
) -> Tuple[Dict[str, List[str]], Dict[str, Set[str]], Dict[str, int]]:
    """Deterministically union manual + timetable assignments."""
    merged: Dict[str, Set[str]] = defaultdict(set)
    pair_sources: Dict[str, Set[str]] = defaultdict(set)
    audit = defaultdict(int)

    normalized_manual, manual_rejected = _normalize_assignments(
        manual_assignments,
        valid_faculty_ids,
        valid_course_ids,
    )
    for key, value in manual_rejected.items():
        audit[key] += value

    for fid, course_ids in normalized_manual.items():
        for cid in course_ids:
            merged[fid].add(cid)
            pair_sources[f"{fid}|{cid}"].add("manual")
            audit["manual_pairs"] += 1

    for fid, course_set in timetable_assignments.items():
        if fid not in valid_faculty_ids:
            audit["timetable_unknown_faculty"] += 1
            continue
        for cid in sorted(course_set):
            if cid not in valid_course_ids:
                audit["timetable_unknown_course"] += 1
                continue
            merged[fid].add(cid)
            pair_sources[f"{fid}|{cid}"].add("timetable")
            audit["timetable_pairs"] += 1

    merged_sorted = {fid: sorted(list(courses_set)) for fid, courses_set in merged.items()}
    audit["merged_faculty_count"] = len(merged_sorted)
    audit["merged_pair_count"] = sum(len(v) for v in merged_sorted.values())

    return merged_sorted, pair_sources, dict(audit)


def extract_assignment_teaches_edges(
    assignments: Dict[str, List[str]],
    pair_sources: Dict[str, Set[str]],
) -> List[dict]:
    edges: Dict[str, dict] = {}

    for faculty_id, course_ids in assignments.items():
        for course_id in course_ids:
            pair_key = f"{faculty_id}|{course_id}"
            sources = sorted(pair_sources.get(pair_key, {"manual"}))
            evidence = [
                "source:teaching_assignments",
                "rule:assignment_union",
            ]
            if "manual" in sources:
                evidence.append("source:manual_assignments")
            if "timetable" in sources:
                evidence.append("source:timetable")
                evidence.append("rule:timetable_faculty_course")

            edge = {
                "id": _edge_id("teaches", faculty_id, course_id),
                "type": "teaches",
                "source": faculty_id,
                "target": course_id,
                "confidence": 1.0,
                "deterministic": True,
                "evidence": evidence,
            }
            edges[edge["id"]] = edge

    return sorted(edges.values(), key=lambda e: e["id"])


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
    courses: List[dict],
) -> Tuple[List[dict], Dict[str, int]]:
    edges: Dict[str, dict] = {}
    rejected = defaultdict(int)

    prereq_pat = re.compile(r"pre[\s-]?requisites?\s*[:\-]\s*([^\n]+)", re.IGNORECASE)
    code_pat = re.compile(r"\b[A-Z0-9]{3,10}\b")

    course_id_to_code = _course_id_to_code(courses)
    alias_to_course_id = _course_alias_to_id(courses)

    for chunk in chunks:
        refs = set(chunk.get("entity_refs", []))
        source_courses = sorted([r for r in refs if r.startswith("course_")])
        text = chunk.get("text", "")
        match_iter = list(prereq_pat.finditer(text))
        if not match_iter:
            continue

        for prereq_match in match_iter:
            matched = prereq_match.group(1)
            source_course: Optional[str] = None

            if len(source_courses) == 1:
                source_course = source_courses[0]
                resolution_rule = "source_course_single_ref"
            else:
                resolution_rule = ""

            if source_course is None and source_courses:
                # Try heading anchor from section hierarchy.
                hierarchy = chunk.get("section_hierarchy") or []
                for heading in hierarchy:
                    h_norm = _normalize(heading)
                    if not h_norm:
                        continue

                    heading_tokens = re.findall(r"\b[A-Z0-9]{3,10}\b", heading.upper())
                    for token in heading_tokens:
                        candidate = code_to_course_id.get(token)
                        if candidate in source_courses:
                            source_course = candidate
                            break
                    if source_course:
                        resolution_rule = "source_course_heading_code"
                        break

                    for alias, cid in alias_to_course_id.items():
                        if cid in source_courses and alias and alias in h_norm:
                            source_course = cid
                            resolution_rule = "source_course_heading_alias"
                            break
                    if source_course:
                        break

            if source_course is None and len(source_courses) > 1:
                # Use nearest preceding source-course code before prerequisite marker.
                nearest_pos = -1
                nearest_course = None
                marker_pos = prereq_match.start()
                upper_text = text.upper()

                for cid in source_courses:
                    code = course_id_to_code.get(cid, "")
                    if not code:
                        continue
                    pos = upper_text.rfind(code, 0, marker_pos)
                    if pos > nearest_pos:
                        nearest_pos = pos
                        nearest_course = cid

                if nearest_course:
                    source_course = nearest_course
                    resolution_rule = "source_course_nearest_preceding"

            if source_course is None:
                if len(source_courses) > 1:
                    rejected["prereq_ambiguous_source_course"] += 1
                else:
                    rejected["prereq_missing_source_course"] += 1
                continue

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
                            f"rule:{resolution_rule}",
                        f"text_span:Prerequisite:{matched.strip()}",
                    ],
                }
                edges[edge["id"]] = edge
                found = True
            if not found:
                rejected["prereq_no_mapped_code"] += 1

    return sorted(edges.values(), key=lambda e: e["id"]), dict(rejected)


def extract_course_semester_edges(
    chunks: List[dict],
    valid_semester_ids: set,
) -> Tuple[List[dict], Dict[str, int]]:
    edges: Dict[str, dict] = {}
    rejected = defaultdict(int)

    sem_pat = re.compile(r"\bsemester\s*([1-8])\b|\bs\s*([1-8])\b", re.IGNORECASE)

    for chunk in chunks:
        refs = set(chunk.get("entity_refs", []))
        courses = sorted([r for r in refs if r.startswith("course_")])
        if not courses:
            continue

        semesters = sorted([r for r in refs if r in valid_semester_ids])
        if not semesters:
            for heading in chunk.get("section_hierarchy") or []:
                match = sem_pat.search(heading)
                if not match:
                    continue
                sem_num = match.group(1) or match.group(2)
                sem_id = f"semester_{sem_num}"
                if sem_id in valid_semester_ids:
                    semesters.append(sem_id)

        if not semesters:
            rejected["course_sem_no_semester_signal"] += len(courses)
            continue

        sem_id = sorted(set(semesters))[0]
        for course_id in courses:
            edge = {
                "id": _edge_id("taught_in", course_id, sem_id),
                "type": "taught_in",
                "source": course_id,
                "target": sem_id,
                "confidence": 1.0,
                "deterministic": True,
                "evidence": [
                    f"chunk_id:{chunk.get('chunk_id', '')}",
                    f"source_file:{chunk.get('source_file', '')}",
                    f"rule:course_semester_link:{sem_id}",
                ],
            }
            edges[edge["id"]] = edge

    return sorted(edges.values(), key=lambda e: e["id"]), dict(rejected)


def extract_course_relation_edges(
    chunks: List[dict],
    code_to_course_id: Dict[str, str],
) -> Tuple[List[dict], Dict[str, int]]:
    edges: Dict[str, dict] = {}
    rejected = defaultdict(int)

    coreq_pat = re.compile(r"co[\s-]?requisites?\s*[:\-]\s*([^\n]+)", re.IGNORECASE)
    code_pat = re.compile(r"\b[A-Z0-9]{3,10}\b")

    for chunk in chunks:
        refs = set(chunk.get("entity_refs", []))
        source_courses = sorted([r for r in refs if r.startswith("course_")])
        if len(source_courses) != 1:
            continue

        source_course = source_courses[0]
        text = chunk.get("text", "")
        matches = coreq_pat.findall(text)
        if not matches:
            continue

        for matched in matches:
            found = False
            for token in code_pat.findall(matched.upper()):
                target_course = code_to_course_id.get(token)
                if not target_course or target_course == source_course:
                    continue

                edge = {
                    "id": _edge_id("corequisite", source_course, target_course),
                    "type": "corequisite",
                    "source": source_course,
                    "target": target_course,
                    "confidence": 1.0,
                    "deterministic": True,
                    "evidence": [
                        f"chunk_id:{chunk.get('chunk_id', '')}",
                        f"source_file:{chunk.get('source_file', '')}",
                        f"text_span:Corequisite:{matched.strip()}",
                    ],
                }
                edges[edge["id"]] = edge
                found = True

            if not found:
                rejected["coreq_no_mapped_code"] += 1

    return sorted(edges.values(), key=lambda e: e["id"]), dict(rejected)


def extract_teaches_edges(
    chunks: List[dict],
    faculty: List[dict],
    courses: List[dict],
) -> Tuple[List[dict], Dict[str, int]]:
    edges: Dict[str, dict] = {}
    rejected = defaultdict(int)

    cue_words = [re.escape(word) for word in config.TEACHES_ASSIGNMENT_CUES]
    cue_pat = re.compile(rf"\b({'|'.join(cue_words)})\b", re.IGNORECASE)

    for chunk in chunks:
        refs = set(chunk.get("entity_refs", []))
        matched_faculty = sorted([ref for ref in refs if ref.startswith("faculty_")])
        matched_courses = sorted([ref for ref in refs if ref.startswith("course_")])
        if not matched_faculty or not matched_courses:
            continue

        lines = [ln.strip() for ln in chunk.get("text", "").splitlines() if ln.strip()]
        has_chunk_signal = False
        for line in lines:
            has_table_row_signal = line.count("|") >= 2
            has_assignment_signal = bool(cue_pat.search(line))
            if has_table_row_signal or has_assignment_signal:
                has_chunk_signal = True
                break

        if not has_chunk_signal:
            rejected["teaches_no_assignment_signal"] += 1
            continue

        for fid in matched_faculty:
            for cid in matched_courses:
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
                        "rule:entity_ref_assignment_signal",
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
    valid_semester_ids = {n["id"] for n in nodes if n.get("type") == "semester"}
    code_map = _course_code_tokens(courses)
    valid_faculty_ids = {f.get("id") for f in faculty if f.get("id")}
    valid_course_ids = {c.get("id") for c in courses if c.get("id")}

    raw_assignments = _load_json_dict(config.TEACHING_ASSIGNMENTS_FILE)
    timetable_assignments, timetable_pair_sources, timetable_audit = extract_timetable_teaching_links(
        chunks,
        faculty,
        courses,
    )
    merged_assignments, merged_pair_sources, merge_audit = merge_teaching_assignments(
        raw_assignments,
        timetable_assignments,
        valid_faculty_ids,
        valid_course_ids,
    )

    for pair_key, sources in timetable_pair_sources.items():
        merged_pair_sources[pair_key].update(sources)

    part_of_edges, part_of_rejected = extract_course_part_of_program_edges(
        chunks, valid_program_ids
    )
    prereq_edges, prereq_rejected = extract_prerequisite_edges(
        chunks, code_map, courses
    )
    assignment_teaches_edges = extract_assignment_teaches_edges(
        merged_assignments,
        merged_pair_sources,
    )
    chunk_teaches_edges, teaches_rejected = extract_teaches_edges(chunks, faculty, courses)
    taught_in_edges, taught_in_rejected = extract_course_semester_edges(
        chunks, valid_semester_ids
    )
    coreq_edges, coreq_rejected = extract_course_relation_edges(chunks, code_map)

    all_edges_map = {}
    for edge in part_of_edges + prereq_edges + assignment_teaches_edges + chunk_teaches_edges + taught_in_edges + coreq_edges:
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
            "taught_in": taught_in_rejected,
            "corequisite": coreq_rejected,
        },
        "teaching_assignment_merge": {
            "manual_rows": len(raw_assignments),
            "merged_rows": len(merged_assignments),
            "audit": merge_audit,
            "timetable_extraction_audit": timetable_audit,
        },
        "validation_errors": errors,
        "valid": len(errors) == 0,
        "known_node_ids": len(node_ids),
        "merged_teaching_assignments": merged_assignments,
    }

    return graph, report


def run_knowledge_graph_pipeline() -> Tuple[dict, dict]:
    config.ensure_directories()

    faculty = _load_json(config.FACULTY_FILE)
    courses = _load_json(config.COURSES_FILE)
    programs = _load_json(config.PROGRAMS_FILE)
    chunks = _load_json(config.CHUNKS_FILE)

    graph, report = build_knowledge_graph(faculty, courses, programs, chunks)

    merged_assignments = report.get("merged_teaching_assignments", {})
    if isinstance(merged_assignments, dict) and merged_assignments:
        with open(config.TEACHING_ASSIGNMENTS_FILE, "w", encoding="utf-8") as f:
            json.dump(merged_assignments, f, indent=2, ensure_ascii=False)

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
