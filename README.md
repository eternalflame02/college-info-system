# College Info System (MBCET CSE)

A hybrid academic information system that combines semantic chunking, deterministic knowledge graphs, and retrieval-augmented generation (RAG) to answer department-related queries.

This repository contains the full pipeline from scraping source content to serving a web chat assistant.

![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Table of Contents

1. Project Overview
2. What Is Implemented
3. System Architecture
4. Repository Structure
5. Prerequisites
6. Installation
7. Configuration
8. Running the System
9. Pipeline Stages (Detailed)
10. Web Assistant and API
11. Data Artifacts
12. Testing and Validation
13. Troubleshooting
14. Development Notes
15. License

## Project Overview

The goal of this project is to provide reliable answers for CSE department information by grounding responses in structured institutional data.

The system workflow is:

1. Crawl and scrape department web content and PDFs.
2. Convert content into normalized Markdown.
3. Build entity registries (faculty, courses, programs).
4. Build semantic chunks with rich metadata.
5. Build deterministic knowledge graph artifacts.
6. Embed chunks and ingest them into ChromaDB.
7. Retrieve relevant evidence at query time.
8. Synthesize grounded responses through the chatbot runtime.

## What Is Implemented

### Data processing and indexing

- HTML and PDF scraping with table-aware extraction.
- OCR-compatible pipeline support for scanned PDFs.
- Semantic chunking with metadata and deduplication.
- Entity registry generation and normalized aliases.
- Two graph artifacts:
  - Canonical graph used by runtime support paths.
  - Deterministic phase-1 graph with schema validation.
- Embedding cache and ChromaDB ingestion.
- Query routing with multi-collection retrieval.

### Runtime assistant

- FastAPI backend (`api_server.py`) with:
  - `GET /` serving the integrated frontend.
  - `POST /chat` for chat inference.
  - `GET /stats` for local KB metrics.
- Frontend chat widget integrated into `frontend/cse_department.html`.
- Markdown rendering in bot messages (including tables) with sanitization.
- Chat window maximize/restore support.
- Main CLI now treats Streamlit as test-only; `chat` stage aliases `serve`.

## System Architecture

```text
                +-------------------------+
                |  MBCET CSE Web Sources  |
                |  HTML Pages + PDFs      |
                +------------+------------+
                             |
                             v
                +-------------------------+
                |  Scraper Layer          |
                |  url_discovery          |
                |  html_scraper           |
                |  pdf_handler            |
                +------------+------------+
                             |
                             v
                +-------------------------+
                |  Markdown Artifacts     |
                |  data/markdown          |
                +------------+------------+
                             |
          +------------------+------------------+
          |                                     |
          v                                     v
+-------------------------+         +--------------------------+
| Entity Registry         |         | Semantic Chunker         |
| faculty/courses/program |         | chunk metadata + links   |
+------------+------------+         +------------+-------------+
             |                                   |
             +------------------+----------------+
                                |
                                v
                 +-------------------------------+
                 | Knowledge Graph Builders      |
                 | canonical + phase-1 graph     |
                 +---------------+---------------+
                                 |
                                 v
                 +-------------------------------+
                 | Embeddings + ChromaDB         |
                 | table / non-table / legacy    |
                 +---------------+---------------+
                                 |
                                 v
                 +-------------------------------+
                 | Chat Runtime                  |
                 | retrieval + synthesis         |
                 +---------------+---------------+
                                 |
                                 v
                 +-------------------------------+
                 | FastAPI + Frontend UI         |
                 | GET /, POST /chat             |
                 +-------------------------------+
```

## Repository Structure

```text
.
├── main.py                       # CLI orchestrator for all stages
├── api_server.py                 # FastAPI server (primary runtime interface)
├── chatbot.py                    # Retrieval + answer synthesis runtime
├── rag_ingestion.py              # Embeddings, ingestion, retrieval helpers
├── app.py                        # Streamlit interface (test-only)
├── config.py                     # Global settings and paths
├── evaluate_rag.py               # RAG quality evaluation script
├── extract_entities.py           # Entity extraction utility
├── visualize_embeddings.py       # Embedding visualization utility
├── requirements.txt
├── .env.example
│
├── scraper/                      # Discovery/scraping/conversion modules
├── chunker/                      # Chunking/entity linking/canonical KG modules
├── knowledge_graph/              # Deterministic phase-1 KG builder
├── tests/                        # Unit tests
├── frontend/                     # Integrated web frontend
│
├── data/
│   ├── markdown/                 # Scraped Markdown content
│   ├── entities/                 # Entity registries
│   ├── chunks/                   # Chunk payloads and reports
│   ├── knowledge_graph/          # Phase-1 graph outputs
│   ├── embeddings/               # Embedding cache and visualizations
│   └── validation/               # Ingestion/validation reports
│
└── chroma_db/                    # Persistent local vector DB files
```

## Prerequisites

- Python 3.11+
- Git
- Recommended: CUDA-capable GPU (optional, CPU fallback supported)
- Optional: Tesseract OCR for scanned PDFs

## Installation

### 1. Clone

```bash
git clone https://github.com/eternalflame02/college-info-system.git
cd college-info-system
```

### 2. Create and activate virtual environment

Windows (PowerShell):

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

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Configuration

Create `.env` from `.env.example` and update as needed.

```bash
cp .env.example .env
```

### Core variables

- `BASE_URL`: website root for scraping.
- `CSE_DEPARTMENT_URL`: crawler seed URL.
- `REQUEST_DELAY`, `MAX_RETRIES`: scraping controls.
- `MIN_CHUNK_WORDS`, `MAX_CHUNK_WORDS`, `PREFERRED_CHUNK_WORDS`: chunk sizing.
- `EMBEDDING_MODEL`: embedding model name.
- `GROQ_API_KEY`, `GROQ_MODEL`: LLM synthesis config.

### Retrieval-related variables

- `CHROMADB_MULTI_COLLECTION_ENABLED`: enable routed collections.
- `CHROMADB_ENABLE_LEGACY_FALLBACK`: fallback to legacy collection.
- `CHROMADB_TABLE_COLLECTION`: collection name for table-heavy chunks.
- `CHROMADB_NON_TABLE_COLLECTION`: collection name for non-table chunks.

Recommended values:

```env
CHROMADB_MULTI_COLLECTION_ENABLED=1
CHROMADB_ENABLE_LEGACY_FALLBACK=1
CHROMADB_TABLE_COLLECTION=mbcet_cse_table
CHROMADB_NON_TABLE_COLLECTION=mbcet_cse_non_table
```

### API runtime variables

- `CHAT_WARMUP_ON_STARTUP`: `1` (default) warms heavy resources on startup; set to `0` for faster dev startup.
- `CORS_ALLOW_ORIGINS`: comma-separated allowlist for frontend origins.

Example:

```env
CHAT_WARMUP_ON_STARTUP=0
CORS_ALLOW_ORIGINS=http://127.0.0.1:8000,http://localhost:8000
```

## Running the System

## Full pipeline

```bash
python main.py --stage all
```

## Stage-based runs

```bash
python main.py --stage scrape
python main.py --stage entities
python main.py --stage kg
python main.py --stage chunk
python main.py --stage graph
python main.py --stage embed --force
```

## Query smoke test

```bash
python main.py --stage query --text "Who teaches Artificial Intelligence?"
```

## Start web assistant (primary runtime)

```bash
python main.py --stage serve
```

or (backward-compatible alias):

```bash
python main.py --stage chat
```

Server URL: `http://127.0.0.1:8000`

## Pipeline Stages (Detailed)

### 1. `scrape`

Purpose:

- Crawl department content and collect HTML/PDF assets.

Operations:

- BFS URL discovery with include/exclude pattern controls.
- HTML conversion to Markdown.
- PDF text/table extraction with OCR fallback support.

Outputs:

- `data/markdown/pages/*.md`
- `data/markdown/pdfs/*.md`

### 2. `entities`

Purpose:

- Build base registries for named entities used in chunking and graph edges.

Operations:

- Extract faculty names from department sources.
- Normalize names and create aliases.
- Ensure course/program registries exist.

Outputs:

- `data/entities/faculty.json`
- `data/entities/courses.json`
- `data/entities/programs.json`

### 3. `kg` (canonical graph)

Purpose:

- Build canonical graph artifact for runtime support and consistency.

Outputs:

- `data/graph/knowledge_graph.json`
- `data/graph/knowledge_graph_summary.json`

### 4. `chunk`

Purpose:

- Create semantically meaningful chunks for retrieval.

Operations:

- Structure-aware segmentation.
- Content typing (table/profile/regulation/list/section).
- Entity linking and metadata enrichment.
- SHA-256 deduplication.

Outputs:

- `data/chunks/chunks.json`
- `data/chunks/chunk_report.json`

### 5. `graph` (phase-1 deterministic graph)

Purpose:

- Build deterministic graph with explicit edge rules and validation.

Outputs:

- `data/knowledge_graph/graph.json`
- `data/knowledge_graph/graph_report.json`
- `data/entities/teaching_assignments.json` (merged deterministic map)

### 6. `embed`

Purpose:

- Generate embeddings, ingest into ChromaDB, and run validation checks.

Outputs:

- `chroma_db/*`
- `data/embeddings/embedding_cache.npz`
- `data/validation/chromadb_ingestion_report.json`
- `data/validation/chromadb_validation_report.json`

### 7. `query`

Purpose:

- Execute a retrieval-only diagnostic query from CLI.

### 8. `serve` / `chat`

Purpose:

- Run the web assistant stack through FastAPI.

## Web Assistant and API

### Backend endpoints

- `GET /`: serves `frontend/cse_department.html`.
- `POST /chat`: accepts `{ "message": "..." }`, returns structured chatbot response.
- `GET /stats`: returns simple local metrics (chunk and faculty counts where available).

### Frontend capabilities

- In-page chat widget with source rendering.
- Bot markdown rendering with sanitization (supports tables/code/blocks/lists).
- Maximize/restore chat window support.
- Mobile-responsive behavior.

### Example request

```bash
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message":"List two CSE courses with credits in a markdown table"}'
```

## Data Artifacts

### Important generated files

- `data/chunks/chunks.json`: retrieval corpus with metadata.
- `data/entities/*.json`: registries and assignment mappings.
- `data/knowledge_graph/graph.json`: deterministic phase-1 graph.
- `data/validation/*.json`: ingestion and query validation summaries.
- `data/embeddings/embedding_cache.npz`: cache for re-embedding speed.

### Runtime DB artifact note

The following are runtime files and should not be committed as data changes:

- `chroma_db/chroma.sqlite3`
- `chroma_db/chroma.sqlite3-shm`
- `chroma_db/chroma.sqlite3-wal`

The repository ignore rules are configured to prevent repeated runtime DB noise.

## Testing and Validation

Run full unit tests:

```bash
pytest -q
```

Run targeted tests:

```bash
pytest tests/test_query_routing.py -q
pytest tests/test_knowledge_graph.py -q
```

Optional syntax sanity:

```bash
python -m py_compile main.py api_server.py
```

## Troubleshooting

### 1. FastAPI server starts slowly

Cause:

- Startup warmup loads model resources.

Fix:

```env
CHAT_WARMUP_ON_STARTUP=0
```

### 2. Chat endpoint returns runtime model errors

Cause:

- Incompatible `transformers`/`huggingface-hub` versions.

Fix:

- Ensure dependencies match `requirements.txt`:
  - `huggingface-hub>=0.26.0,<1.0`
  - `transformers>=4.44.0,<5.0`

### 3. `git restore` fails for `chroma_db/chroma.sqlite3*` on Windows

Cause:

- SQLite files locked by active process.

Fix:

1. Stop running API/processes that access Chroma.
2. Use one-time index cleanup if files were tracked historically:

```bash
git rm --cached chroma_db/chroma.sqlite3 chroma_db/chroma.sqlite3-shm chroma_db/chroma.sqlite3-wal
```

3. Keep ignore rules enabled to prevent recurrence.

### 4. Chat endpoint returns 500

Steps:

1. Check server logs for `/chat request failed` stack trace.
2. Verify `data/chunks/chunks.json` and Chroma artifacts exist.
3. Run pipeline stages up to `embed`.

## Development Notes

- Primary runtime path is FastAPI + integrated frontend.
- Streamlit interface remains test-only.
- `main.py --stage chat` is intentionally a backward-compatible alias to `serve`.
- Keep graph outputs deterministic and evidence-grounded.
- Prefer rerunning stage-specific pipelines instead of editing generated artifacts manually.

## License

MIT License. See `LICENSE`.
