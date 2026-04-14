"""
Tests for query classification and routing hints.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_ingestion import classify_query_type, query_chromadb_with_fallback, apply_adaptive_distance_threshold


def test_classify_teaching_query():
    assert classify_query_type("Who teaches Artificial Intelligence?") == "teaching"


def test_classify_faculty_query():
    assert classify_query_type("Who is the HOD?") == "faculty"


def test_classify_regulation_query_by_attendance_keyword():
    assert classify_query_type("What is the attendance requirement for exams?") == "regulation"


class _FakeCollection:
    def __init__(self):
        self.calls = 0

    def query(self, **kwargs):
        self.calls += 1
        n = kwargs.get("n_results", 5)

        # First call: strict-route empty (forces fallback)
        if self.calls == 1:
            return {
                "ids": [[]],
                "distances": [[]],
                "documents": [[]],
                "metadatas": [[]],
            }

        # Fallback call: mixed content
        ids = [f"id_{i}" for i in range(n)]
        dists = [0.45 + (i * 0.02) for i in range(n)]
        docs = [f"doc {i}" for i in range(n)]
        metas = [
            {"content_type": "table" if i % 2 == 0 else "section"}
            for i in range(n)
        ]
        return {
            "ids": [ids],
            "distances": [dists],
            "documents": [docs],
            "metadatas": [metas],
        }


class _FakeCollectionPrimaryNonEmpty:
    def __init__(self):
        self.calls = 0

    def query(self, **kwargs):
        self.calls += 1
        # Primary call returns one weak but present result.
        if self.calls == 1:
            return {
                "ids": [["id_primary"]],
                "distances": [[0.75]],
                "documents": [["doc primary"]],
                "metadatas": [[{"content_type": "profile"}]],
            }

        return {
            "ids": [["id_fallback"]],
            "distances": [[0.5]],
            "documents": [["doc fallback"]],
            "metadatas": [[{"content_type": "table"}]],
        }


class _FakeCollectionTeachingSparsePoor:
    def __init__(self):
        self.calls = 0

    def query(self, **kwargs):
        self.calls += 1
        # Primary call: sparse and poor teaching route.
        if self.calls == 1:
            return {
                "ids": [["id_kg_only"]],
                "distances": [[1.05]],
                "documents": [["kg statement"]],
                "metadatas": [[{"content_type": "knowledge_graph"}]],
            }

        # Fallback call: mixed results.
        return {
            "ids": [["id_fb1", "id_fb2"]],
            "distances": [[0.6, 0.7]],
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"content_type": "table"}, {"content_type": "section"}]],
        }


class _FakeCollectionRegulationPoor:
    def __init__(self):
        self.calls = 0

    def query(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return {
                "ids": [["id_reg_1"]],
                "distances": [[1.02]],
                "documents": [["reg snippet"]],
                "metadatas": [[{"content_type": "regulation"}]],
            }

        return {
            "ids": [["id_fb1", "id_fb2"]],
            "distances": [[0.62, 0.66]],
            "documents": [["fallback 1", "fallback 2"]],
            "metadatas": [[{"content_type": "table"}, {"content_type": "section"}]],
        }


def test_general_fallback_triggered(monkeypatch):
    class _FakeModel:
        def encode(self, *_args, **_kwargs):
            import numpy as np
            return np.array([[0.1, 0.2, 0.3]], dtype=float)

    monkeypatch.setattr("rag_ingestion.load_embedding_model", lambda device="auto": _FakeModel())

    fake = _FakeCollection()
    results = query_chromadb_with_fallback(fake, "Tell me about department facilities", query_type="general")
    assert results["fallback_triggered"] is True
    assert results["filtered_count"] > 0
    assert len(results["content_type_distribution"]) >= 1


def test_non_general_keeps_primary_when_non_empty(monkeypatch):
    class _FakeModel:
        def encode(self, *_args, **_kwargs):
            import numpy as np
            return np.array([[0.1, 0.2, 0.3]], dtype=float)

    monkeypatch.setattr("rag_ingestion.load_embedding_model", lambda device="auto": _FakeModel())

    fake = _FakeCollectionPrimaryNonEmpty()
    results = query_chromadb_with_fallback(fake, "Who is the HOD?", query_type="faculty")
    assert results["fallback_triggered"] is False
    assert fake.calls == 1
    assert results["filtered_count"] == 1


def test_teaching_sparse_poor_can_fallback(monkeypatch):
    class _FakeModel:
        def encode(self, *_args, **_kwargs):
            import numpy as np
            return np.array([[0.1, 0.2, 0.3]], dtype=float)

    monkeypatch.setattr("rag_ingestion.load_embedding_model", lambda device="auto": _FakeModel())

    fake = _FakeCollectionTeachingSparsePoor()
    results = query_chromadb_with_fallback(
        fake,
        "Who teaches Advanced AI?",
        query_type="teaching",
    )
    assert results["fallback_triggered"] is True
    assert fake.calls == 2


def test_regulation_poor_can_fallback(monkeypatch):
    class _FakeModel:
        def encode(self, *_args, **_kwargs):
            import numpy as np
            return np.array([[0.1, 0.2, 0.3]], dtype=float)

    monkeypatch.setattr("rag_ingestion.load_embedding_model", lambda device="auto": _FakeModel())

    fake = _FakeCollectionRegulationPoor()
    results = query_chromadb_with_fallback(
        fake,
        "Explain grading rules in regulation",
        query_type="regulation",
    )
    assert results["fallback_triggered"] is True
    assert fake.calls == 2


def test_faculty_threshold_allows_sparse_profile_hits():
    raw = {
        "ids": [["id1"]],
        "distances": [[1.30]],
        "documents": [["profile text"]],
        "metadatas": [[{"content_type": "profile"}]],
    }
    filtered = apply_adaptive_distance_threshold(raw, "faculty")
    assert filtered["filtered_count"] == 1
