"""
Tests for query classification and routing hints.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_ingestion import classify_query_type


def test_classify_teaching_query():
    assert classify_query_type("Who teaches Artificial Intelligence?") == "teaching"


def test_classify_faculty_query():
    assert classify_query_type("Who is the HOD?") == "faculty"
