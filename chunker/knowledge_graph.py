import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import config

logger = logging.getLogger(__name__)


def _safe_load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return default


def _oxford_join(parts: List[str]) -> str:
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _safe_aliases(entity: Dict[str, Any]) -> List[str]:
    aliases = entity.get("aliases", [])
    if not isinstance(aliases, list):
        return []
    return [a for a in aliases if isinstance(a, str) and a.strip()]


def _make_node(entity: Dict[str, Any], node_type: str) -> Optional[Dict[str, Any]]:
    entity_id = entity.get("id")
    if not isinstance(entity_id, str) or not entity_id.strip():
        return None
    label = entity.get("name", entity_id)
    return {
        "id": entity_id,
        "type": node_type,
        "label": label,
        "aliases": _safe_aliases(entity),
        "attributes": {
            k: v for k, v in entity.items()
            if k not in {"id", "type", "name", "aliases"}
        },
    }


def _edge_id(relation: str, source: str, target: str) -> str:
    relation_norm = re.sub(r"[^A-Za-z0-9_]", "_", relation).lower()
    return f"edge_{relation_norm}_{source}_to_{target}"


def _extract_course_program_links(
    course: Dict[str, Any],
    program_ids: Set[str],
    program_by_name: Dict[str, str],
) -> List[str]:
    linked: Set[str] = set()

    def maybe_add(value: Any):
        if not isinstance(value, str):
            return
        token = value.strip()
        if not token:
            return
        if token in program_ids:
            linked.add(token)
            return
        normalized = token.lower()
        if normalized in program_by_name:
            linked.add(program_by_name[normalized])

    maybe_add(course.get("program_id"))
    maybe_add(course.get("program"))

    for key in ("program_ids", "programs"):
        raw = course.get(key, [])
        if isinstance(raw, list):
            for item in raw:
                maybe_add(item)

    return sorted(linked)


def build_knowledge_graph(data_dir: Path) -> Dict[str, Any]:
    """Build canonical knowledge graph JSON payload from entity registries."""
    entities_dir = data_dir / "entities"
    faculty_list = _safe_load_json(entities_dir / "faculty.json", default=[])
    courses_list = _safe_load_json(entities_dir / "courses.json", default=[])
    programs_list = _safe_load_json(entities_dir / "programs.json", default=[])
    assignments = _safe_load_json(entities_dir / "teaching_assignments.json", default={})

    if not isinstance(faculty_list, list):
        faculty_list = []
    if not isinstance(courses_list, list):
        courses_list = []
    if not isinstance(programs_list, list):
        programs_list = []
    if not isinstance(assignments, dict):
        assignments = {}

    nodes: Dict[str, Dict[str, Any]] = {}
    edges: Dict[str, Dict[str, Any]] = {}
    orphan_faculty_refs: Set[str] = set()
    orphan_course_refs: Set[str] = set()

    for entity in faculty_list:
        node = _make_node(entity, "faculty")
        if node:
            nodes[node["id"]] = node

    for entity in courses_list:
        node = _make_node(entity, "course")
        if node:
            nodes[node["id"]] = node

    for entity in programs_list:
        node = _make_node(entity, "program")
        if node:
            nodes[node["id"]] = node

    program_ids = {node_id for node_id, node in nodes.items() if node.get("type") == "program"}
    program_by_name: Dict[str, str] = {}
    for node_id in program_ids:
        node = nodes[node_id]
        label = str(node.get("label", "")).strip().lower()
        if label:
            program_by_name[label] = node_id
        for alias in node.get("aliases", []):
            if isinstance(alias, str) and alias.strip():
                program_by_name[alias.strip().lower()] = node_id

    for faculty_id, course_ids in assignments.items():
        if faculty_id not in nodes:
            orphan_faculty_refs.add(faculty_id)
            continue
        if not isinstance(course_ids, list):
            continue
        for course_id in course_ids:
            if not isinstance(course_id, str) or not course_id.strip():
                continue
            if course_id not in nodes:
                orphan_course_refs.add(course_id)
                continue
            edge = {
                "id": _edge_id("TEACHES", faculty_id, course_id),
                "source": faculty_id,
                "target": course_id,
                "relation": "TEACHES",
                "attributes": {},
            }
            edges[edge["id"]] = edge

    for course in courses_list:
        course_id = course.get("id")
        if not isinstance(course_id, str) or course_id not in nodes:
            continue
        linked_programs = _extract_course_program_links(course, program_ids, program_by_name)
        for program_id in linked_programs:
            edge = {
                "id": _edge_id("BELONGS_TO_PROGRAM", course_id, program_id),
                "source": course_id,
                "target": program_id,
                "relation": "BELONGS_TO_PROGRAM",
                "attributes": {},
            }
            edges[edge["id"]] = edge

    node_type_counts: Dict[str, int] = {}
    for node in nodes.values():
        node_type = node.get("type", "unknown")
        node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1

    edge_relation_counts: Dict[str, int] = {}
    for edge in edges.values():
        relation = edge.get("relation", "unknown")
        edge_relation_counts[relation] = edge_relation_counts.get(relation, 0) + 1

    summary = {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "node_counts_by_type": dict(sorted(node_type_counts.items())),
        "edge_counts_by_relation": dict(sorted(edge_relation_counts.items())),
        "orphan_references": {
            "faculty_ids": sorted(orphan_faculty_refs),
            "course_ids": sorted(orphan_course_refs),
        },
    }

    graph = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "nodes": sorted(nodes.values(), key=lambda x: x["id"]),
        "edges": sorted(edges.values(), key=lambda x: x["id"]),
        "summary": summary,
    }
    return graph


def save_knowledge_graph(
    graph: Dict[str, Any],
    output_path: Path,
    summary_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(graph.get("summary", {}), f, indent=2, ensure_ascii=False)


def build_and_save_knowledge_graph(
    data_dir: Path,
    output_path: Optional[Path] = None,
    summary_path: Optional[Path] = None,
) -> Dict[str, Any]:
    if output_path is None:
        output_path = config.KNOWLEDGE_GRAPH_FILE
    if summary_path is None:
        summary_path = config.KNOWLEDGE_GRAPH_SUMMARY_FILE

    graph = build_knowledge_graph(data_dir)
    save_knowledge_graph(graph, output_path, summary_path)
    logger.info("Knowledge graph saved to %s", output_path)
    return graph


def load_knowledge_graph(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    if path is None:
        path = config.KNOWLEDGE_GRAPH_FILE
    data = _safe_load_json(path, default=None)
    if isinstance(data, dict) and "nodes" in data and "edges" in data:
        return data
    return None


def query_knowledge_graph(graph: Dict[str, Any], query_text: str) -> str:
    """Answer simple relationship questions over TEACHES edges."""
    if not query_text.strip():
        return "Please provide a query."

    nodes = {n.get("id"): n for n in graph.get("nodes", []) if isinstance(n, dict)}
    teaches_edges = [
        e for e in graph.get("edges", [])
        if isinstance(e, dict) and e.get("relation") == "TEACHES"
    ]

    course_lookup: List[Tuple[str, str]] = []
    faculty_lookup: List[Tuple[str, str]] = []
    for node_id, node in nodes.items():
        node_type = node.get("type")
        label = str(node.get("label", ""))
        aliases = [a for a in node.get("aliases", []) if isinstance(a, str)]
        corpus = [label, *aliases]
        for token in corpus:
            norm = token.strip().lower()
            if not norm:
                continue
            if node_type == "course":
                course_lookup.append((norm, node_id))
            elif node_type == "faculty":
                faculty_lookup.append((norm, node_id))

    def normalize_phrase(text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()

    def phrase_matches(target: str, candidate: str) -> bool:
        target_norm = normalize_phrase(target)
        candidate_norm = normalize_phrase(candidate)
        if not target_norm or not candidate_norm:
            return False
        if target_norm == candidate_norm:
            return True
        return bool(re.search(rf"\b{re.escape(target_norm)}\b", candidate_norm))

    q = query_text.lower().strip()
    who_teaches_patterns = [
        r"who teaches (.+)\??$",
        r"instructor for (.+)\??$",
        r"who handles (.+)\??$",
    ]
    for pattern in who_teaches_patterns:
        match = re.search(pattern, q)
        if match:
            target = match.group(1).strip()
            matched_course_ids = {cid for token, cid in course_lookup if phrase_matches(target, token)}
            if not matched_course_ids:
                return "No matching course found in the knowledge graph."
            faculty_ids = sorted({
                edge["source"] for edge in teaches_edges
                if edge.get("target") in matched_course_ids
            })
            if not faculty_ids:
                return "No instructor mapping found for that course."
            names = [nodes[fid].get("label", fid) for fid in faculty_ids if fid in nodes]
            return f"Instructor(s): {_oxford_join(names)}."

    taught_by_patterns = [
        r"what does (.+) teach\??$",
        r"courses taught by (.+)\??$",
        r"what courses does (.+) handle\??$",
    ]
    for pattern in taught_by_patterns:
        match = re.search(pattern, q)
        if match:
            target = match.group(1).strip()
            matched_faculty_ids = {fid for token, fid in faculty_lookup if phrase_matches(target, token)}
            if not matched_faculty_ids:
                return "No matching faculty member found in the knowledge graph."
            course_ids = sorted({
                edge["target"] for edge in teaches_edges
                if edge.get("source") in matched_faculty_ids
            })
            if not course_ids:
                return "No courses mapped for that faculty member."
            course_labels = []
            for cid in course_ids:
                node = nodes.get(cid, {})
                label = node.get("label", cid)
                code = node.get("attributes", {}).get("code", "")
                course_labels.append(f"{code} ({label})" if code else label)
            return f"Course(s): {_oxford_join(course_labels)}."

    return "Supported queries: 'Who teaches <course>?' or 'What does <faculty> teach?'"


def generate_knowledge_graph_documents(data_dir: Path) -> List[Dict]:
    """
    Generate synthetic relationship documents from canonical graph edges.
    """
    graph_path = data_dir / "graph" / "knowledge_graph.json"
    graph = _safe_load_json(graph_path, default={})
    if not graph:
        graph = build_knowledge_graph(data_dir)

    nodes = {n.get("id"): n for n in graph.get("nodes", []) if isinstance(n, dict)}
    teaches_map: Dict[str, List[Dict[str, str]]] = {}
    relation_types_by_faculty: Dict[str, Set[str]] = {}

    for edge in graph.get("edges", []):
        if not isinstance(edge, dict) or edge.get("relation") != "TEACHES":
            continue
        faculty_id = edge.get("source")
        course_id = edge.get("target")
        if faculty_id not in nodes or course_id not in nodes:
            continue

        course_node = nodes[course_id]
        course_name = str(course_node.get("label", "Unknown"))
        course_code = str(course_node.get("attributes", {}).get("code", course_id))
        relation_types_by_faculty.setdefault(faculty_id, set()).add(str(edge.get("relation", "")))
        teaches_map.setdefault(faculty_id, []).append(
            {
                "course_id": course_id,
                "course_name": course_name,
                "course_code": course_code,
            }
        )

    documents = []
    for faculty_id, courses in sorted(teaches_map.items()):
        faculty_node = nodes.get(faculty_id, {})
        fac_name = str(faculty_node.get("label", "Unknown"))
        fac_attrs = faculty_node.get("attributes", {})
        fac_email = fac_attrs.get("email", "")
        fac_designation = fac_attrs.get("designation", "")

        sorted_courses = sorted(courses, key=lambda c: c["course_id"])
        course_text_parts = [
            f"{course['course_code']} ({course['course_name']})"
            for course in sorted_courses
        ]
        course_text = _oxford_join(course_text_parts)

        role_text = fac_designation if fac_designation else "faculty member"
        email_text = f" Contact email: {fac_email}." if fac_email else ""

        text = (
            f"[Context: Knowledge Graph] {fac_name} is a {role_text} in the CSE department."
            f" {fac_name} teaches {course_text}."
            f" The instructor for {course_text} is {fac_name}.{email_text}"
        )

        doc = {
            "id": f"kg_{faculty_id}",
            "text": text,
            "metadata": {
                "source_file": "data/entities/teaching_assignments.json",
                "content_type": "knowledge_graph",
                "main_topic": "Teaching Assignment",
                "relation_types": sorted(
                    [r for r in relation_types_by_faculty.get(faculty_id, set()) if r]
                ),
                "faculty_id": faculty_id,
                "faculty_name": fac_name,
                "faculty_designation": fac_designation or "",
                "faculty_email": fac_email or "",
                "course_ids": [course["course_id"] for course in sorted_courses],
                "course_codes": [course["course_code"] for course in sorted_courses],
                "course_names": [course["course_name"] for course in sorted_courses],
            },
        }
        documents.append(doc)

    logger.info("Generated %d synthetic Knowledge Graph documents.", len(documents))
    return documents
