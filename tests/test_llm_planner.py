from chatbot import (
    RetrievalPlan,
    _default_retrieval_plan,
    _extract_json_block,
    _normalize_planner_output,
    _plan_retrieval_with_llm,
    _execute_retrieval_plan,
)


def test_extract_json_block_plain_object():
    payload = '{"query_type":"teaching","source_mode":"kg_only","rewritten_query":"who teaches ai","rationale":"relation lookup"}'
    parsed = _extract_json_block(payload)
    assert parsed is not None
    assert parsed["query_type"] == "teaching"
    assert parsed["source_mode"] == "kg_only"


def test_extract_json_block_fenced_object():
    payload = """```json
{
  "query_type": "general",
  "source_mode": "hybrid",
  "rewritten_query": "labs and research in cse",
  "rationale": "broad query"
}
```"""
    parsed = _extract_json_block(payload)
    assert parsed is not None
    assert parsed["query_type"] == "general"
    assert parsed["source_mode"] == "hybrid"


def test_normalize_planner_output_invalid_values(monkeypatch):
    monkeypatch.setattr("chatbot.classify_query", lambda _q: "faculty")
    plan = _normalize_planner_output(
        {
            "query_type": "unknown_type",
            "source_mode": "weird_mode",
            "rewritten_query": "",
            "rationale": "",
        },
        original_query="Who is the HOD?",
    )
    assert plan.query_type == "faculty"
    assert plan.source_mode == "vector_only"
    assert plan.rewritten_query == "Who is the HOD?"


def test_default_retrieval_plan_teaching_hybrid(monkeypatch):
    monkeypatch.setattr("chatbot.classify_query", lambda _q: "teaching")
    plan = _default_retrieval_plan("Who teaches AI?")
    assert plan.query_type == "teaching"
    assert plan.source_mode == "hybrid"


def test_plan_retrieval_with_llm_falls_back_without_client(monkeypatch):
    monkeypatch.setattr("chatbot.classify_query", lambda _q: "general")
    monkeypatch.setattr("chatbot._get_groq_client", lambda: None)
    plan = _plan_retrieval_with_llm("Tell me about labs and research")
    assert plan.query_type == "general"
    assert plan.source_mode == "vector_only"


def test_execute_retrieval_plan_kg_only_uses_kg(monkeypatch):
    monkeypatch.setattr("chatbot._retrieve_from_kg", lambda _q: "Dr. X teaches Y")
    monkeypatch.setattr("chatbot._retrieve_from_chromadb", lambda *_args, **_kwargs: ["unused"])

    plan = RetrievalPlan(query_type="teaching", source_mode="kg_only", rewritten_query="who teaches y")
    kg_answer, chunks = _execute_retrieval_plan("Who teaches Y?", plan)

    assert kg_answer == "Dr. X teaches Y"
    assert chunks == []


def test_execute_retrieval_plan_hybrid_runs_both(monkeypatch):
    monkeypatch.setattr("chatbot._retrieve_from_kg", lambda _q: "KG answer")
    monkeypatch.setattr("chatbot._retrieve_from_chromadb", lambda *_args, **_kwargs: ["chunk1", "chunk2"])

    plan = RetrievalPlan(query_type="regulation", source_mode="hybrid", rewritten_query="attendance exam rules")
    kg_answer, chunks = _execute_retrieval_plan("What are attendance rules?", plan)

    assert kg_answer == "KG answer"
    assert chunks == ["chunk1", "chunk2"]


def test_default_retrieval_plan_greeting_no_retrieval():
    plan = _default_retrieval_plan("hello")
    assert plan.source_mode == "no_retrieval"
    assert plan.query_type == "general"


def test_normalize_planner_output_accepts_no_retrieval(monkeypatch):
    monkeypatch.setattr("chatbot.classify_query", lambda _q: "general")
    plan = _normalize_planner_output(
        {
            "query_type": "general",
            "source_mode": "no_retrieval",
            "rewritten_query": "hello",
            "rationale": "smalltalk",
        },
        original_query="hello",
    )
    assert plan.source_mode == "no_retrieval"


def test_execute_retrieval_plan_no_retrieval():
    plan = RetrievalPlan(query_type="general", source_mode="no_retrieval", rewritten_query="hi")
    kg_answer, chunks = _execute_retrieval_plan("hi", plan)
    assert kg_answer is None
    assert chunks == []
