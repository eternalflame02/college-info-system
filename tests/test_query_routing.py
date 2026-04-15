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


class _FakeCollectionRegulationMixedPrimary:
    def __init__(self):
        self.calls = 0

    def query(self, **kwargs):
        self.calls += 1
        return {
            "ids": [["reg_1", "sec_1", "reg_2"]],
            "distances": [[1.2, 1.1, 1.25]],
            "documents": [["regulation doc 1", "general section", "regulation doc 2"]],
            "metadatas": [[
                {"content_type": "regulation"},
                {"content_type": "section"},
                {"content_type": "regulation"},
            ]],
        }


class _FakeCollectionFacultyMixedPrimary:
    def query(self, **kwargs):
        return {
            "ids": [["fac_match", "fac_other"]],
            "distances": [[1.3, 1.0]],
            "documents": [["profile dr tessy", "profile someone else"]],
            "metadatas": [[
                {"content_type": "profile", "faculty_id": "faculty_dr_tessy_mathew"},
                {"content_type": "profile", "faculty_id": "faculty_dr_jisha_john"},
            ]],
        }


class _FakeCollectionFacultyNoisy:
    def __init__(self):
        self.calls = 0

    def query(self, **kwargs):
        self.calls += 1
        return {
            "ids": [["noise_1"]],
            "distances": [[1.2]],
            "documents": [["workshop mentions many faculty names"]],
            "metadatas": [[
                {
                    "content_type": "profile",
                    "faculty_id": "faculty_dr_tessy_mathew",
                    "entity_refs": "faculty_dr_tessy_mathew,faculty_dr_jisha_john",
                    "source_file": "data/markdown/pages/computer-science-engineering_workshops-seminars_8fc87b.md",
                }
            ]],
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
    assert results["fallback_triggered"] is False
    assert fake.calls == 1


def test_faculty_threshold_allows_sparse_profile_hits():
    raw = {
        "ids": [["id1"]],
        "distances": [[1.30]],
        "documents": [["profile text"]],
        "metadatas": [[{"content_type": "profile"}]],
    }
    filtered = apply_adaptive_distance_threshold(raw, "faculty")
    assert filtered["filtered_count"] == 1


def test_regulation_query_prefers_regulation_content(monkeypatch):
    class _FakeModel:
        def encode(self, *_args, **_kwargs):
            import numpy as np
            return np.array([[0.1, 0.2, 0.3]], dtype=float)

    monkeypatch.setattr("rag_ingestion.load_embedding_model", lambda device="auto": _FakeModel())

    fake = _FakeCollectionRegulationMixedPrimary()
    results = query_chromadb_with_fallback(
        fake,
        "R2023 regulations",
        query_type="regulation",
    )

    assert results["fallback_triggered"] is False
    assert results["filtered_count"] >= 1
    assert any((m or {}).get("content_type") == "regulation" for m in results["metadatas"][0])
    assert fake.calls == 1


def test_faculty_query_filters_to_name_signal(monkeypatch):
    class _FakeModel:
        def encode(self, *_args, **_kwargs):
            import numpy as np
            return np.array([[0.1, 0.2, 0.3]], dtype=float)

    class _Registry:
        lookup = {
            "dr tessy mathew": "faculty_dr_tessy_mathew",
            "tessy mathew": "faculty_dr_tessy_mathew",
        }

        def load_all(self):
            return 0

        def find_exact_match(self, text):
            return None

        def find_fuzzy_match(self, _text, entity_type="faculty"):
            return None

    monkeypatch.setattr("rag_ingestion.load_embedding_model", lambda device="auto": _FakeModel())
    monkeypatch.setattr("chunker.entity_registry.EntityRegistry", lambda: _Registry())

    fake = _FakeCollectionFacultyMixedPrimary()
    results = query_chromadb_with_fallback(
        fake,
        "Who is Dr Tessy?",
        query_type="faculty",
    )

    assert results["fallback_triggered"] is False
    assert results["filtered_count"] == 1
    assert results["metadatas"][0][0]["faculty_id"] == "faculty_dr_tessy_mathew"


def test_faculty_query_rejects_noisy_aggregate_match(monkeypatch):
    class _FakeModel:
        def encode(self, *_args, **_kwargs):
            import numpy as np
            return np.array([[0.1, 0.2, 0.3]], dtype=float)

    class _Registry:
        lookup = {
            "dr tessy mathew": "faculty_dr_tessy_mathew",
            "tessy mathew": "faculty_dr_tessy_mathew",
        }

        def load_all(self):
            return 0

        def find_exact_match(self, _text):
            return None

        def find_fuzzy_match(self, _text, entity_type="faculty"):
            return None

    monkeypatch.setattr("rag_ingestion.load_embedding_model", lambda device="auto": _FakeModel())
    monkeypatch.setattr("chunker.entity_registry.EntityRegistry", lambda: _Registry())

    fake = _FakeCollectionFacultyNoisy()
    results = query_chromadb_with_fallback(
        fake,
        "Who is Dr Tessy?",
        query_type="faculty",
    )

    assert results["fallback_triggered"] is True
    assert results["filtered_count"] == 0


class _FakeCollectionTrackCalls:
    def __init__(self, name, content_type):
        self.name = name
        self.content_type = content_type
        self.calls = 0

    def query(self, **kwargs):
        self.calls += 1
        n = kwargs.get("n_results", 5)
        ids = [f"{self.name}_{i}" for i in range(n)]
        dists = [0.45 + (i * 0.02) for i in range(n)]
        docs = [f"doc {self.name} {i}" for i in range(n)]
        metas = [{"content_type": self.content_type} for _ in range(n)]
        return {
            "ids": [ids],
            "distances": [dists],
            "documents": [docs],
            "metadatas": [metas],
        }


class _FakeCollectionFacultyPrimaryWrong:
    def __init__(self):
        self.calls = 0

    def query(self, **kwargs):
        self.calls += 1
        return {
            "ids": [["non_table_wrong"]],
            "distances": [[0.72]],
            "documents": [["profile someone else"]],
            "metadatas": [[
                {
                    "content_type": "section",
                    "faculty_id": "faculty_dr_jisha_john",
                    "entity_refs": "faculty_dr_jisha_john",
                    "source_file": "data/markdown/pages/computer-science-engineering_department-advisory-board_accaa6.md",
                }
            ]],
        }


class _FakeCollectionFacultyTableCorrect:
    def __init__(self):
        self.calls = 0

    def query(self, **kwargs):
        self.calls += 1
        return {
            "ids": [["table_tessy"]],
            "distances": [[0.84]],
            "documents": [["[Context: Dr. Tessy Mathew] profile table"]],
            "metadatas": [[
                {
                    "content_type": "table",
                    "faculty_id": "faculty_dr_tessy_mathew",
                    "entity_refs": "faculty_dr_tessy_mathew",
                    "source_file": "data/markdown/pages/faculty_dr-tessy-mathew_febe6c.md",
                }
            ]],
        }


def test_course_query_prefers_table_collection(monkeypatch):
    class _FakeModel:
        def encode(self, *_args, **_kwargs):
            import numpy as np
            return np.array([[0.1, 0.2, 0.3]], dtype=float)

    monkeypatch.setattr("rag_ingestion.load_embedding_model", lambda device="auto": _FakeModel())

    table = _FakeCollectionTrackCalls("table", "table")
    non_table = _FakeCollectionTrackCalls("non_table", "section")
    collection_map = {
        "table": table,
        "non_table": non_table,
    }

    results = query_chromadb_with_fallback(
        collection_map,
        "What courses are in semester 3?",
        query_type="course",
    )

    assert results["primary_collection"] == "table"
    assert table.calls >= 1


def test_faculty_query_fallbacks_when_primary_has_wrong_faculty(monkeypatch):
    class _FakeModel:
        def encode(self, *_args, **_kwargs):
            import numpy as np
            return np.array([[0.1, 0.2, 0.3]], dtype=float)

    class _Registry:
        lookup = {
            "dr tessy mathew": "faculty_dr_tessy_mathew",
            "tessy mathew": "faculty_dr_tessy_mathew",
        }

        def load_all(self):
            return 0

        def find_exact_match(self, _text):
            return None

        def find_fuzzy_match(self, _text, entity_type="faculty"):
            return None

    monkeypatch.setattr("rag_ingestion.load_embedding_model", lambda device="auto": _FakeModel())
    monkeypatch.setattr("chunker.entity_registry.EntityRegistry", lambda: _Registry())

    non_table = _FakeCollectionFacultyPrimaryWrong()
    table = _FakeCollectionFacultyTableCorrect()

    collection_map = {
        "non_table": non_table,
        "table": table,
    }

    results = query_chromadb_with_fallback(
        collection_map,
        "Who is Dr Tessy?",
        query_type="faculty",
    )

    assert results["fallback_triggered"] is True
    assert results["filtered_count"] == 1
    assert results["metadatas"][0][0]["faculty_id"] == "faculty_dr_tessy_mathew"
    assert non_table.calls >= 1
    assert table.calls >= 1
