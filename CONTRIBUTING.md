# Contributing Guide

This document defines how to contribute to the College Info System repository effectively and safely.

## 1. Project Context

Project title:

Hybrid Knowledge-Graph-Enhanced Retrieval-Augmented Generation for Academic Information Systems

Academic context:

- B.Tech CSE Mini Project
- Institutional domain: MBCET CSE department information retrieval

Team:

- Julia Mariam John (B23CS2137)
- Nirmel B Joseph (B23CS2148)
- Rohith NS (B23CS2156)

Guide:

- Mr. Praveen J.S, Assistant Professor, Department of CSE

## 2. Contribution Principles

All contributions should prioritize:

1. Determinism in generated structured artifacts.
2. Grounded retrieval and evidence-backed output.
3. Reproducibility of pipeline runs.
4. Clear documentation for operational handoff.

## 3. Development Environment Setup

## 3.1 Clone

```bash
git clone https://github.com/eternalflame02/college-info-system.git
cd college-info-system
```

## 3.2 Virtual environment

Windows PowerShell:

```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python -m venv venv
source venv/bin/activate
```

## 3.3 Install dependencies

```bash
pip install -r requirements.txt
```

## 3.4 Prepare local configuration

```bash
cp .env.example .env
```

Set values as needed (Groq key, model options, routing flags, warmup option).

## 4. Branching Strategy

Use concise branch names based on purpose:

- `feature/<short-name>`: new feature work.
- `fix/<short-name>`: bug fixes.
- `docs/<short-name>`: documentation-only updates.
- `refactor/<short-name>`: internal restructuring without behavior change.

Examples:

- `feature/frontend-chat-controls`
- `fix/chroma-routing-threshold`
- `docs/runbook-refresh`

## 5. Commit Standards

Use Conventional Commit style where possible:

- `feat: ...`
- `fix: ...`
- `docs: ...`
- `refactor: ...`
- `test: ...`
- `chore: ...`

Good examples:

- `feat: add maximize toggle to frontend chat widget`
- `fix: pin transformers and huggingface-hub compatibility range`
- `docs: expand README with API runtime and troubleshooting`

Keep commits focused:

- Separate code behavior changes from generated-data churn when possible.
- Prefer one logical concern per commit.

## 6. Required Validation Before Push

Run these checks before opening a PR:

1. Unit tests:

```bash
pytest -q
```

2. Syntax checks for core entry points:

```bash
python -m py_compile main.py api_server.py
```

3. Query smoke test:

```bash
python main.py --stage query --text "Who is the HOD?"
```

4. Runtime launch check:

```bash
python main.py --stage serve
```

Confirm:

- frontend loads at `http://127.0.0.1:8000`
- chat endpoint responds
- markdown table rendering is correct
- clear/maximize/close behavior works
- long responses can complete within 180s client timeout

Optional public validation (ngrok):

```bash
ngrok http --domain=hyo-gymnocarpous-lingeringly.ngrok-free.dev 8000
```

Then verify the public URL and one `/chat` request.

## 7. Pull Request Requirements

PR description should include:

1. Problem statement.
2. Scope of changes.
3. Files/modules affected.
4. Validation commands run.
5. Results summary.
6. Screenshots/GIFs for frontend changes (if applicable).

Suggested PR template sections:

- Summary
- Technical changes
- Testing performed
- Risks and mitigations
- Follow-ups

## 8. Data and Artifact Policy

Generated artifacts can be intentionally updated, but runtime noise should be avoided.

### 8.1 Usually acceptable to commit

- Deterministic JSON artifacts intentionally regenerated for functional updates.
- Documentation updates.
- Source code and tests.

### 8.2 Usually not acceptable to commit

- Ephemeral local DB lock-state files.
- Accidental runtime binary drift.

Runtime DB files to keep out of routine commits:

- `chroma_db/chroma.sqlite3`
- `chroma_db/chroma.sqlite3-shm`
- `chroma_db/chroma.sqlite3-wal`

If historically tracked and causing churn, one-time untracking can be used:

```bash
git rm --cached chroma_db/chroma.sqlite3 chroma_db/chroma.sqlite3-shm chroma_db/chroma.sqlite3-wal
```

## 8.3 Secret handling policy

- Never commit real credentials (for example, `GROQ_API_KEY`, `HF_TOKEN`) to tracked files.
- Keep secrets in local `.env` only.
- Use placeholders in `.env.example` and docs.

## 9. Knowledge Graph Contribution Rules

For phase-1 graph updates:

- Keep edges deterministic and evidence-grounded.
- Avoid speculative relationships.
- Validate endpoint IDs against entity registries.
- Preserve stable node/edge IDs where possible.

Allowed edge families in current phase:

- `course -> part_of -> program`
- `course -> taught_in -> semester`
- `faculty -> teaches -> course`
- `course -> has_prerequisite -> course` (explicit evidence only)
- `course -> corequisite -> course` (explicit evidence only)

If evidence is ambiguous, omit the edge.

## 10. Runtime Interface Guidelines

Primary runtime interface:

- FastAPI server (`main.py --stage serve`)

Compatibility note:

- `main.py --stage chat` is a backward-compatible alias to `serve`.
- Streamlit is retained as test-only support.

API endpoints:

- `GET /`
- `POST /chat`
- `GET /stats`

When changing API schema or behavior, update README and workflow docs in the same PR.

## 11. Documentation Requirements

Any contribution that changes behavior must also update documentation:

- User-facing behavior: README
- Operational process: WORKFLOW
- Contributor process: CONTRIBUTING
- Environment surface: `.env.example`

Documentation updates are mandatory for:

- new env vars
- stage behavior changes
- endpoint changes
- frontend UX feature changes

## 12. Issue Triage and Ownership

Assign issues to one of these tracks:

1. Data ingestion/scraping
2. Chunking/entity linking
3. Graph building
4. Retrieval/runtime
5. Frontend UX
6. Testing/evaluation
7. Documentation

Each issue should define:

- acceptance criteria
- expected test coverage
- artifacts expected to change

## 13. Common Pitfalls and Fixes

### 13.1 Slow startup during UI iteration

Set:

```env
CHAT_WARMUP_ON_STARTUP=0
```

### 13.2 Chat runtime failures due dependency drift

Ensure `requirements.txt` ranges are respected, especially:

- `huggingface-hub>=0.26.0,<1.0`
- `transformers>=4.44.0,<5.0`

### 13.3 Windows file lock errors on sqlite files

- Stop all processes using ChromaDB.
- Retry file operations after process shutdown.

### 13.4 `serve` and ngrok lifecycle conflicts

- If one-command auto-ngrok startup is unstable in your terminal, set `AUTO_START_NGROK=0`.
- Run `serve` and `ngrok` in separate terminals for maximum reliability.

## 14. Code Review Checklist

Reviewer should confirm:

1. Behavior matches PR summary.
2. Tests are adequate and pass.
3. Docs are updated for behavior changes.
4. No accidental runtime DB noise is included.
5. No regression in serve/chat startup path.

## 15. Conduct and Collaboration

- Use respectful and constructive review feedback.
- Prefer objective comments tied to correctness, maintainability, and reproducibility.
- Resolve review threads with clear change notes.

## 16. License

By contributing, you agree your contribution is licensed under the repository MIT License.
