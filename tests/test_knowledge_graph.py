"""
Tests for synthetic knowledge graph generation.
"""

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from chunker.knowledge_graph import (
    build_and_save_knowledge_graph,
    build_knowledge_graph,
    generate_knowledge_graph_documents,
    query_knowledge_graph,
)
from rag_ingestion import append_knowledge_graph_chunks


def _write_entities(tmp_path, faculty, courses, programs, assignments):
    entities_dir = tmp_path / "entities"
    entities_dir.mkdir(parents=True)
    (entities_dir / "faculty.json").write_text(json.dumps(faculty), encoding="utf-8")
    (entities_dir / "courses.json").write_text(json.dumps(courses), encoding="utf-8")
    (entities_dir / "programs.json").write_text(json.dumps(programs), encoding="utf-8")
    (entities_dir / "teaching_assignments.json").write_text(json.dumps(assignments), encoding="utf-8")


def test_build_knowledge_graph_nodes_edges_and_summary(tmp_path):
    faculty = [
        {
            "id": "faculty_dr_jisha_john",
            "name": "Dr. Jisha John",
            "designation": "Professor",
            "email": "jisha@mbcet.ac.in",
            "aliases": ["Jisha John"],
        }
    ]
    courses = [
        {
            "id": "course_cs0u20a",
            "name": "Artificial Intelligence",
            "code": "CS0U20A",
            "program_id": "prog_btech_cse",
        }
    ]
    programs = [
        {
            "id": "prog_btech_cse",
            "name": "B.Tech Computer Science and Engineering",
            "aliases": ["B.Tech CSE"],
        }
    ]
    assignments = {"faculty_dr_jisha_john": ["course_cs0u20a"]}
    _write_entities(tmp_path, faculty, courses, programs, assignments)

    graph = build_knowledge_graph(tmp_path)
    node_ids = {n["id"] for n in graph["nodes"]}
    edge_ids = {e["id"] for e in graph["edges"]}

    assert "faculty_dr_jisha_john" in node_ids
    assert "course_cs0u20a" in node_ids
    assert "prog_btech_cse" in node_ids
    assert any(e["relation"] == "TEACHES" for e in graph["edges"])
    assert any(e["relation"] == "BELONGS_TO_PROGRAM" for e in graph["edges"])
    assert len(edge_ids) == len(graph["edges"])  # deterministic dedupe
    assert graph["summary"]["node_counts_by_type"]["faculty"] == 1
    assert graph["summary"]["edge_counts_by_relation"]["TEACHES"] == 1


def test_build_knowledge_graph_missing_reference_and_dedup(tmp_path):
    faculty = [{"id": "faculty_dr_jisha_john", "name": "Dr. Jisha John"}]
    courses = [{"id": "course_cs0u20a", "name": "Artificial Intelligence", "code": "CS0U20A"}]
    programs = []
    assignments = {
        "faculty_dr_jisha_john": ["course_cs0u20a", "course_cs0u20a", "course_missing"],
        "faculty_missing": ["course_cs0u20a"],
    }
    _write_entities(tmp_path, faculty, courses, programs, assignments)

    graph = build_knowledge_graph(tmp_path)
    teaches_edges = [e for e in graph["edges"] if e["relation"] == "TEACHES"]

    assert len(teaches_edges) == 1
    assert graph["summary"]["orphan_references"]["faculty_ids"] == ["faculty_missing"]
    assert graph["summary"]["orphan_references"]["course_ids"] == ["course_missing"]


def test_generate_knowledge_graph_documents(tmp_path):
    faculty = [
        {
            "id": "faculty_dr_jisha_john",
            "name": "Dr. Jisha John",
            "designation": "Professor",
            "email": "jisha@mbcet.ac.in",
        }
    ]
    courses = [{"id": "course_cs0u20a", "name": "Artificial Intelligence", "code": "CS0U20A"}]
    programs = []
    assignments = {"faculty_dr_jisha_john": ["course_cs0u20a"]}
    _write_entities(tmp_path, faculty, courses, programs, assignments)

    docs = generate_knowledge_graph_documents(tmp_path)

    assert len(docs) == 1
    assert docs[0]["metadata"]["content_type"] == "knowledge_graph"
    assert docs[0]["metadata"]["faculty_id"] == "faculty_dr_jisha_john"
    assert docs[0]["metadata"]["course_codes"] == ["CS0U20A"]
    assert "relation_types" in docs[0]["metadata"]
    assert "teaches" in docs[0]["text"].lower()


def test_query_knowledge_graph_for_basic_relationships(tmp_path):
    faculty = [{"id": "faculty_dr_jisha_john", "name": "Dr. Jisha John"}]
    courses = [{"id": "course_cs0u20a", "name": "Artificial Intelligence", "code": "CS0U20A"}]
    programs = []
    assignments = {"faculty_dr_jisha_john": ["course_cs0u20a"]}
    _write_entities(tmp_path, faculty, courses, programs, assignments)

    graph = build_knowledge_graph(tmp_path)
    answer_1 = query_knowledge_graph(graph, "Who teaches Artificial Intelligence?")
    answer_2 = query_knowledge_graph(graph, "What does Jisha teach?")

    assert "jisha" in answer_1.lower()
    assert "artificial intelligence" in answer_2.lower()


def test_build_and_save_and_ingestion_append_integration(tmp_path):
    faculty = [{"id": "faculty_dr_jisha_john", "name": "Dr. Jisha John"}]
    courses = [{"id": "course_cs0u20a", "name": "Artificial Intelligence", "code": "CS0U20A"}]
    programs = []
    assignments = {"faculty_dr_jisha_john": ["course_cs0u20a"]}
    _write_entities(tmp_path, faculty, courses, programs, assignments)

    graph_output = tmp_path / "graph" / "knowledge_graph.json"
    summary_output = tmp_path / "graph" / "knowledge_graph_summary.json"
    graph = build_and_save_knowledge_graph(
        tmp_path,
        output_path=graph_output,
        summary_path=summary_output,
    )

    assert graph_output.exists()
    assert summary_output.exists()
    assert graph["summary"]["total_edges"] == 1

    chunks = [
        {
            "chunk_id": "seed_1",
            "text": "seed",
            "source_type": "html",
            "source_file": "seed.md",
            "section_hierarchy": [],
            "content_type": "section",
            "entity_refs": [],
            "page_range": None,
            "word_count": 1,
            "hash": "seedhash",
            "metadata": {},
        }
    ]
    chunk_report = {"total_chunks": 1, "chunks_by_type": {"section": 1}}
    updated_chunks, updated_report, added = append_knowledge_graph_chunks(
        chunks, chunk_report, tmp_path
    )

    assert added == 1
    assert len(updated_chunks) == 2
    assert updated_report["total_chunks"] == 2
    assert updated_report["chunks_by_type"]["knowledge_graph"] == 1
