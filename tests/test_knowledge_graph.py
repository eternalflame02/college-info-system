"""
Tests for phase-1 knowledge graph builder.
"""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_graph.builder import (
    build_nodes,
    build_knowledge_graph,
    extract_prerequisite_edges,
    validate_graph,
)
import config


def _sample_entities():
    faculty = [
        {
            "id": "faculty_dr_john_doe",
            "type": "faculty",
            "name": "Dr. John Doe",
            "aliases": ["John Doe"],
        }
    ]
    courses = [
        {
            "id": "course_cs101",
            "type": "course",
            "name": "Data Structures",
            "code": "CS101",
            "aliases": ["CS101", "Data Structures"],
        },
        {
            "id": "course_cs102",
            "type": "course",
            "name": "Algorithms",
            "code": "CS102",
            "aliases": ["CS102", "Algorithms"],
        },
    ]
    programs = [
        {
            "id": "prog_btech_cse",
            "type": "program",
            "name": "B.Tech Computer Science and Engineering",
            "aliases": ["B.Tech CSE", "CSE"],
        }
    ]
    return faculty, courses, programs


def test_build_nodes_includes_semester_nodes():
    faculty, courses, programs = _sample_entities()
    chunks = [{"chunk_id": "c1", "entity_refs": ["semester_3", "course_cs101"]}]

    nodes = build_nodes(faculty, courses, programs, chunks)
    node_ids = {n["id"] for n in nodes}

    assert "semester_3" in node_ids
    assert "faculty_dr_john_doe" in node_ids
    assert "course_cs101" in node_ids
    assert "prog_btech_cse" in node_ids


def test_extract_prerequisite_edges_from_explicit_statement():
    chunks = [
        {
            "chunk_id": "c-prereq",
            "source_file": "data/markdown/pdfs/sample.md",
            "entity_refs": ["course_cs101"],
            "text": "Prerequisite: CS102",
        }
    ]
    code_map = {"CS102": "course_cs102"}

    edges, rejected = extract_prerequisite_edges(chunks, code_map)
    assert len(edges) == 1
    assert edges[0]["type"] == "has_prerequisite"
    assert edges[0]["source"] == "course_cs101"
    assert edges[0]["target"] == "course_cs102"
    assert rejected == {}


def test_build_knowledge_graph_creates_deterministic_edges():
    faculty, courses, programs = _sample_entities()
    chunks = [
        {
            "chunk_id": "c-part-of",
            "source_file": "data/markdown/pdfs/CSE_B-Tech-2023_S3-S4-Syllabus.md",
            "entity_refs": ["course_cs101"],
            "text": "Course details",
        },
        {
            "chunk_id": "c-prereq",
            "source_file": "data/markdown/pdfs/sample.md",
            "entity_refs": ["course_cs101"],
            "text": "Prerequisite: CS102",
        },
        {
            "chunk_id": "c-teaches",
            "source_file": "data/markdown/pages/faculty.md",
            "entity_refs": ["course_cs101", "faculty_dr_john_doe"],
            "text": "| Data Structures | Dr. John Doe | Assistant Professor |",
        },
    ]

    graph, report = build_knowledge_graph(faculty, courses, programs, chunks)
    edge_types = {e["type"] for e in graph["edges"]}

    assert report["valid"] is True
    assert "part_of" in edge_types
    assert "has_prerequisite" in edge_types
    assert "teaches" in edge_types
    assert all(e["confidence"] == 1.0 for e in graph["edges"])
    assert all(e["deterministic"] is True for e in graph["edges"])


def test_validation_fails_for_dangling_edge():
    graph = {
        "version": "1.0",
        "generated_at": "2026-01-01T00:00:00Z",
        "nodes": [
            {
                "id": "n1",
                "type": "course",
                "name": "X",
                "aliases": [],
                "source_refs": ["s"],
            }
        ],
        "edges": [
            {
                "id": "e1",
                "type": "part_of",
                "source": "n1",
                "target": "missing",
                "confidence": 1.0,
                "deterministic": True,
                "evidence": ["rule:x"],
            }
        ],
    }

    errors = validate_graph(graph)
    assert any("Dangling edge target" in err for err in errors)


def test_real_data_graph_structure_smoke():
    required = [
        config.FACULTY_FILE,
        config.COURSES_FILE,
        config.PROGRAMS_FILE,
        config.CHUNKS_FILE,
    ]
    if not all(p.exists() for p in required):
        return

    with open(config.FACULTY_FILE, "r", encoding="utf-8") as f:
        faculty = json.load(f)
    with open(config.COURSES_FILE, "r", encoding="utf-8") as f:
        courses = json.load(f)
    with open(config.PROGRAMS_FILE, "r", encoding="utf-8") as f:
        programs = json.load(f)
    with open(config.CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    graph, report = build_knowledge_graph(faculty, courses, programs, chunks)

    assert isinstance(graph, dict)
    assert isinstance(report, dict)
    assert "nodes" in graph and "edges" in graph
    assert report["total_nodes"] >= 0
    assert report["total_edges"] >= 0
