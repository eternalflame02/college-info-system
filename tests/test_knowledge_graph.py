"""
Tests for synthetic knowledge graph generation.
"""

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from chunker.knowledge_graph import generate_knowledge_graph_documents


def test_generate_knowledge_graph_documents(tmp_path):
    entities_dir = tmp_path / "entities"
    entities_dir.mkdir(parents=True)

    faculty = [
        {
            "id": "faculty_dr_jisha_john",
            "name": "Dr. Jisha John",
            "designation": "Professor",
            "email": "jisha@mbcet.ac.in",
        }
    ]
    courses = [
        {
            "id": "course_cs0u20a",
            "name": "Artificial Intelligence",
            "code": "CS0U20A",
        }
    ]
    assignments = {
        "faculty_dr_jisha_john": ["course_cs0u20a"],
    }

    (entities_dir / "faculty.json").write_text(json.dumps(faculty), encoding="utf-8")
    (entities_dir / "courses.json").write_text(json.dumps(courses), encoding="utf-8")
    (entities_dir / "teaching_assignments.json").write_text(json.dumps(assignments), encoding="utf-8")

    docs = generate_knowledge_graph_documents(tmp_path)

    assert len(docs) == 1
    assert docs[0]["metadata"]["content_type"] == "knowledge_graph"
    assert docs[0]["metadata"]["faculty_id"] == "faculty_dr_jisha_john"
    assert docs[0]["metadata"]["course_codes"] == ["CS0U20A"]
    assert "teaches" in docs[0]["text"].lower()
