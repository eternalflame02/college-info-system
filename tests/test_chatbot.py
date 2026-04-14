from chatbot import RetrievedChunk, _synthesize_answer_with_llm


def test_synthesize_falls_back_when_llm_returns_none(monkeypatch):
    class _FakeMessage:
        content = "None"

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeCompletion:
        choices = [_FakeChoice()]

    class _FakeChatCompletions:
        @staticmethod
        def create(**_kwargs):
            return _FakeCompletion()

    class _FakeChat:
        completions = _FakeChatCompletions()

    class _FakeClient:
        chat = _FakeChat()

    monkeypatch.setattr("chatbot._get_groq_client", lambda: _FakeClient())

    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            text="Dr. Tessy Mathew is Assistant Professor in CSE.",
            content_type="profile",
            source_file="data/markdown/pages/faculty_dr-tessy-mathew_febe6c.md",
            section_hierarchy="Dr. Tessy Mathew",
            distance=1.0,
        )
    ]

    result = _synthesize_answer_with_llm(
        query="Who is Dr Tessy",
        query_type="faculty",
        chunks=chunks,
        kg_answer=None,
    )

    assert "Relevant information found" in result
    assert "Dr. Tessy Mathew" in result