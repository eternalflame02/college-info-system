# College Info System Workflow (Detailed)

This document is the operational runbook for this repository. It explains how data flows through the system, how to run and validate each stage, and how to maintain stable developer workflows.

## 1. Workflow Goals

The workflow is designed to ensure:

1. Deterministic, reproducible graph and chunk artifacts.
2. Grounded retrieval from high-quality evidence.
3. Clear separation between source code and generated runtime data.
4. Safe day-to-day development with predictable test and release steps.

## 2. Modes of Use

There are three common working modes.

### A. Full rebuild mode

Use when source website content has significantly changed or when preparing a major demo.

### B. Incremental update mode

Use when only part of the content or logic changed.

### C. Runtime-only mode

Use when you only want to run the assistant against an already ingested database.

## 3. High-Level Stage Order

Canonical order for a full fresh build:

1. `scrape`
2. `entities`
3. `kg`
4. `chunk`
5. `graph`
6. `embed`
7. `serve` (or `chat` alias)

Equivalent single command:

```bash
python main.py --stage all
```

## 4. Detailed Stage Operations

## 4.1 Scraping stage

Command:

```bash
python main.py --stage scrape
```

Responsibilities:

- URL discovery with BFS crawling.
- HTML extraction and markdown conversion.
- PDF extraction including tables and OCR fallback.

Primary outputs:

- `data/markdown/pages/*.md`
- `data/markdown/pdfs/*.md`

Operational checks:

- Verify new markdown files are generated.
- Spot-check at least one PDF-derived markdown table for formatting integrity.

## 4.2 Entity stage

Command:

```bash
python main.py --stage entities
```

Responsibilities:

- Build faculty entity registry with alias normalization.
- Ensure course/program registry files exist.

Primary outputs:

- `data/entities/faculty.json`
- `data/entities/courses.json`
- `data/entities/programs.json`

Operational checks:

- Confirm IDs are deterministic and stable across runs.
- Ensure aliases include title-stripped variants where applicable.

## 4.3 Canonical KG stage (`kg`)

Command:

```bash
python main.py --stage kg
```

Responsibilities:

- Build canonical graph used by runtime support components.

Primary outputs:

- `data/graph/knowledge_graph.json`
- `data/graph/knowledge_graph_summary.json`

Operational checks:

- Validate node and edge counts are reasonable.
- Confirm no malformed IDs or missing references.

## 4.4 Chunk stage

Command:

```bash
python main.py --stage chunk
```

Responsibilities:

- Convert markdown corpus to semantic chunks.
- Classify content types (table/profile/regulation/list/section).
- Add metadata and entity links.
- Deduplicate with SHA-256 text hashing.

Primary outputs:

- `data/chunks/chunks.json`
- `data/chunks/chunk_report.json`
- `data/chunks/errors.log`

Operational checks:

- Verify chunk counts by content type.
- Check that chunk IDs remain stable if source text did not change.

## 4.5 Phase-1 graph stage (`graph`)

Command:

```bash
python main.py --stage graph
```

Responsibilities:

- Build deterministic graph with strict edge policies.
- Merge timetable-derived teaching links into assignment map.
- Validate schema and endpoint integrity.

Primary outputs:

- `data/knowledge_graph/graph.json`
- `data/knowledge_graph/graph_report.json`
- `data/entities/teaching_assignments.json`

Operational checks:

- Ensure only deterministic edges are emitted.
- Confirm evidence strings exist for supported edge families.

## 4.6 Embedding/ingestion stage

Command:

```bash
python main.py --stage embed
```

Force rebuild command:

```bash
python main.py --stage embed --force
```

Responsibilities:

- Generate embeddings (with cache reuse where possible).
- Ingest into ChromaDB (single or multi-collection mode).
- Produce ingestion and validation reports.

Primary outputs:

- `chroma_db/*`
- `data/embeddings/embedding_cache.npz`
- `data/validation/chromadb_ingestion_report.json`
- `data/validation/chromadb_validation_report.json`

Operational checks:

- Verify collection counts align with chunk totals.
- Run one or more query smoke tests immediately after ingestion.

## 4.7 Query stage

Command:

```bash
python main.py --stage query --text "Who teaches AI?"
```

Responsibilities:

- Execute retrieval-only diagnostics with query routing and fallback behaviors.

Operational checks:

- Inspect top retrieved chunks and confidence/distance behavior.
- Ensure route/fallback mode appears appropriate for the query class.

## 4.8 Serve/chat stage

Commands:

```bash
python main.py --stage serve
python main.py --stage chat
```

Notes:

- `chat` is a backward-compatible alias for `serve`.
- Primary runtime is FastAPI + integrated frontend.
- Streamlit is retained as test-only support.

Responsibilities:

- Serve frontend at `GET /`.
- Handle chat requests at `POST /chat`.
- Expose metrics at `GET /stats`.

Online exposure options:

1. Two-terminal mode (recommended):
   - terminal A: `python main.py --stage serve`
   - terminal B: `ngrok http --domain=<your-domain> 8000`
2. One-command mode (optional):
   - set `AUTO_START_NGROK=1`
   - set `NGROK_DOMAIN=<your-domain>`
   - run `python main.py --stage serve`

Notes:

- Two-terminal mode is generally more stable when debugging process lifecycle issues.
- One-command mode is convenient but may behave differently across terminal hosts.

Operational checks:

- Open `http://127.0.0.1:8000` and verify chat widget loads.
- Submit a query and confirm answer + source metadata render.
- Verify markdown tables are rendered properly in bot responses.
- Verify clear/maximize/close controls render and function in chat window.
- Verify long-running prompts can complete within the 180-second client timeout.
- If ngrok is enabled, verify public URL responds and `/chat` calls succeed.

## 5. Recommended Developer Flows

## 5.1 First-time setup

```bash
python -m venv venv
# activate venv
pip install -r requirements.txt
cp .env.example .env
```

Then run:

```bash
python main.py --stage all
python main.py --stage serve
```

## 5.2 Daily incremental flow

When source data changed:

1. Re-run `scrape`.
2. Re-run `entities` if person/course metadata changed.
3. Re-run `chunk`, `graph`, `embed`.
4. Run query smoke tests.
5. Start `serve` and test the UI.

When only runtime/frontend/backend code changed:

1. Skip data stages.
2. Run tests and syntax checks.
3. Start `serve` and run manual chat checks.

## 5.3 Fast UI iteration flow

Use this to reduce startup delay while adjusting frontend and API behavior:

```env
CHAT_WARMUP_ON_STARTUP=0
```

Then:

```bash
python main.py --stage serve
```

For public UI testing:

```bash
ngrok http --domain=hyo-gymnocarpous-lingeringly.ngrok-free.dev 8000
```

## 6. Quality Gates

Before merging:

1. Run unit tests:

```bash
pytest -q
```

2. Run syntax sanity checks for critical entry points:

```bash
python -m py_compile main.py api_server.py
```

3. Run retrieval smoke tests:

```bash
python main.py --stage query --text "Who is the HOD?"
python main.py --stage query --text "List two CSE courses with credits"
```

4. Start server and validate frontend behaviors:

- chat open/close
- markdown table rendering
- maximize/restore
- source list rendering

## 7. Data and Git Hygiene

Generated data can be large and/or runtime-volatile. Treat it carefully.

Guidelines:

- Do not commit transient runtime DB noise.
- Keep source code changes separate from generated data churn.
- Commit deterministic artifacts only when intentionally updated.

Important runtime files:

- `chroma_db/chroma.sqlite3`
- `chroma_db/chroma.sqlite3-shm`
- `chroma_db/chroma.sqlite3-wal`

If these were historically tracked, one-time untracking may be required:

```bash
git rm --cached chroma_db/chroma.sqlite3 chroma_db/chroma.sqlite3-shm chroma_db/chroma.sqlite3-wal
```

## 8. Windows Lock Handling for SQLite

On Windows, SQLite files may be locked by running processes, causing restore/unlink failures.

If `git restore` fails with unlink errors:

1. Stop server and any process using ChromaDB.
2. Retry restore or use one-time index cleanup (`git rm --cached ...`).
3. Confirm ignore rules are present for runtime DB artifacts.

## 9. Runtime Startup and Port Handling

If `serve` fails with port conflict:

1. Check port ownership:

```powershell
Get-NetTCPConnection -LocalPort 8000 | Select-Object OwningProcess, LocalAddress, LocalPort, State
```

2. Stop stale process if required:

```powershell
Stop-Process -Id <PID> -Force
```

3. Restart server:

```bash
python main.py --stage serve
```

## 10. Branch and PR Workflow

1. Create branch from `main`.
2. Keep PRs focused by concern:
   - feature/runtime changes
   - data artifact changes
   - documentation updates
3. Include validation evidence in PR description:
   - test command outputs summary
   - query smoke test results
   - UI verification notes

## 11. Release Readiness Checklist

Before tagging a release/demo build:

1. `.env.example` matches required configuration surface.
2. README and workflow docs match current runtime behavior.
3. Unit tests pass.
4. Query smoke tests pass.
5. FastAPI server launches cleanly from `main.py --stage serve`.
6. Frontend chat UX checks are complete.
7. No unintended binary/runtime files remain staged.

## 12. Known Design Notes

- The repository intentionally maintains two graph outputs:
  - Canonical graph (`data/graph/*`)
  - Phase-1 deterministic graph (`data/knowledge_graph/*`)
- Keep semantic assumptions and entity ID conventions aligned across both builders.
- Runtime retrieval quality depends heavily on chunk metadata integrity and collection routing.
