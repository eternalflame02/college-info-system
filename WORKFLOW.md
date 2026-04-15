# College Info System - Detailed Workflow

This document explains the complete repository workflow from data acquisition to chatbot response generation, including development, testing, and documentation tracks.

## 1. Repository Intent

The project builds a CSE department knowledge system using:

1. Web and PDF scraping
2. Markdown normalization
3. Entity extraction
4. Semantic chunking
5. Knowledge graph construction (two graph artifacts)
6. Embedding + ChromaDB ingestion
7. Query routing + chatbot answer synthesis
8. Evaluation and visualization utilities

Core entrypoint: `main.py`

## 2. Folder Responsibilities

### Runtime code

- `scraper/`: URL discovery, HTML scraping, PDF parsing/OCR, HTML to Markdown conversion
- `chunker/`: semantic chunking, content classification, entity linking, canonical graph generation
- `knowledge_graph/`: phase-1 deterministic knowledge graph builder and validator
- `app.py`: Streamlit UI
- `chatbot.py`: retrieval + KG + LLM orchestration
- `rag_ingestion.py`: embeddings, ChromaDB ingestion, query routing, validation
- `config.py`: all paths and runtime settings
- `main.py`: CLI orchestrator for all stages

### Data and artifacts

- `data/raw/`: downloaded PDFs and raw timetable assets
- `data/markdown/pages/`: HTML-derived markdown
- `data/markdown/pdfs/`: PDF-derived markdown
- `data/entities/`: faculty/course/program/teaching-assignment JSONs
- `data/chunks/`: chunk payloads and chunk reports
- `data/knowledge_graph/`: phase-1 graph output + report
- `data/embeddings/`: cached embeddings + embedding visual outputs
- `data/validation/`: ingestion and retrieval validation reports
- `chroma_db/`: persistent Chroma database

### Documentation and planning

- `README.md`: high-level architecture and usage
- `CONTRIBUTING.md`: collaboration and contribution flow
- `Docs/knowledge-graph/`: phase-1 schema and graph notes
- `Agent_instructions/`: implementation guidance and instruction artifacts
- `Docs/* presentation and abstract folders`: academic deliverables

### Quality and experiments

- `tests/`: unit and behavior tests for scraper/chunker/entity/KG/query routing
- `evaluate_rag.py`: direct query sanity script for retrieval quality checks
- `tmp_semantic_test.py`: ad-hoc semantic retrieval testing
- `visualize_embeddings.py`: dimensionality reduction and embedding visualization

## 3. End-to-End Pipeline (Operational)

Run full pipeline:

```bash
python main.py --stage all
```

Execution order in `--stage all`:

1. `scrape`
2. `entities`
3. `kg` (canonical graph in `chunker/knowledge_graph.py`)
4. `chunk`
5. `graph` (phase-1 deterministic graph in `knowledge_graph/builder.py`)
6. `embed`

### Stage A - Scrape (`main.py --stage scrape`)

1. Discover URLs with BFS (`scraper/url_discovery.py`)
2. Split into HTML pages and PDF links
3. Scrape HTML pages (`scraper/html_scraper.py`)
4. Convert HTML to markdown + frontmatter (`scraper/markdown_converter.py`)
5. Download/process PDFs (`scraper/pdf_handler.py`)
6. Use table extraction (pdfplumber) and OCR fallback where needed

Primary outputs:

- `data/markdown/pages/*.md`
- `data/markdown/pdfs/*.md`

### Stage B - Entities (`main.py --stage entities` + optional `extract_entities.py`)

Path 1 (CLI stage):

1. Extract faculty links from department page
2. Normalize names and aliases
3. Write `faculty.json`
4. Ensure placeholder `courses.json` and `programs.json`

Path 2 (course/program extraction utility):

1. Parse markdown syllabus-like tables from PDF markdown
2. Detect course code/name/credits heuristically
3. Write course/program registries

Primary outputs:

- `data/entities/faculty.json`
- `data/entities/courses.json`
- `data/entities/programs.json`
- `data/entities/teaching_assignments.json` (if maintained manually or by future automation)

### Stage C - Canonical Graph (`main.py --stage kg`)

Builder: `chunker/knowledge_graph.py`

1. Load entity registries and optional teaching assignments
2. Build node set for faculty/course/program
3. Build TEACHES and BELONGS_TO_PROGRAM relationships where deterministic
4. Capture orphan references and summary stats

Outputs:

- `data/graph/knowledge_graph.json`
- `data/graph/knowledge_graph_summary.json`

This canonical graph is used by chatbot KG queries and synthetic KG document generation during ingestion.

### Stage D - Semantic Chunking (`main.py --stage chunk`)

Pipeline: `chunker/semantic_chunker.py`

1. Load entity registry + teaching assignments
2. Read all markdown files from pages and pdf folders
3. Chunk by headings/tables/size thresholds
4. Classify content type (`table`, `profile`, `regulation`, `list`, `section`)
5. Link entity references (exact + fuzzy faculty + semester tokens)
6. Generate deterministic chunk IDs + text hash
7. Skip duplicates globally by SHA-256 hash
8. Persist chunk output and reports

Outputs:

- `data/chunks/chunks.json`
- `data/chunks/chunk_report.json`
- `data/chunks/errors.log`
- `logs/chunker.log`

### Stage E - Phase-1 Graph (`main.py --stage graph`)

Builder: `knowledge_graph/builder.py`

1. Build graph nodes from faculty/course/program plus deterministic semester nodes
2. Extract deterministic edges only:
	- `part_of`
	- `teaches` (from merged manual assignments + timetable-derived links)
	- `taught_in`
	- `has_prerequisite` (when explicitly present and source course is grounded)
	- `corequisite` (when explicitly present)
3. Build timetable-derived faculty-course links from timetable chunks and merge into `data/entities/teaching_assignments.json`
3. Validate schema integrity, deterministic constraints, and node endpoint integrity
4. Write graph + report

Outputs:

- `data/knowledge_graph/graph.json`
- `data/knowledge_graph/graph_report.json`
- `data/entities/teaching_assignments.json` (merged deterministic assignment map)

### Stage F - Embedding + Chroma Ingestion (`main.py --stage embed`)

Orchestrator: `rag_ingestion.py`

1. Load chunk artifacts
2. Append synthetic knowledge-graph chunks generated from canonical graph
3. Use embedding cache when chunk IDs match
4. Else generate embeddings (GPU auto detect, OOM fallback)
5. Initialize persistent ChromaDB client and collections (`table`, `non_table`, optional `legacy`)
6. Batch ingest documents into routed collections by content family
7. Run validation suite with representative queries
8. Save ingestion and validation reports

Outputs:

- `chroma_db/` database files
- `data/embeddings/embedding_cache.npz`
- `data/validation/chromadb_ingestion_report.json`
- `data/validation/chromadb_validation_report.json`

## 4. Query and Chat Runtime Flow

### CLI query mode

```bash
python main.py --stage query --text "Who teaches Artificial Intelligence?"
```

Flow:

1. Classify query type
2. Run routed retrieval (`query_chromadb`) with adaptive distance threshold
3. Apply query-signal reranking (for example semester/faculty hints)
4. If route is empty or poor, trigger mixed-content fallback retrieval
5. Re-rank fallback results for diversity and print confidence diagnostics
6. Print ranked results, fallback metadata, and content-type mix

### Streamlit chatbot mode

```bash
python main.py --stage chat
```

Flow in `chatbot.py`:

1. Warm singleton resources (embedding model, Chroma collection, canonical graph)
2. Classify query (`teaching`, `faculty`, `course`, `timetable`, `regulation`, `general`)
3. If `teaching`, try direct KG answer first
4. Retrieve candidate chunks through routed fallback retrieval path
5. Sort evidence by confidence/distance and build grounded context window
6. Synthesize final answer with Groq model when available
7. Fall back to chunk-based formatted response if LLM unavailable
8. Return answer + sources + quality + response latency

UI in `app.py`:

1. Loads cached backend callable
2. Maintains conversation state
3. Displays sources and metadata badges
4. Supports quick example prompts from sidebar

## 5. Developer Workflow

### Recommended setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Then choose one of these sequences.

### Fresh rebuild sequence

1. `python main.py --stage scrape`
2. `python main.py --stage entities`
3. `python extract_entities.py` (if you want richer course/program extraction)
4. `python main.py --stage kg`
5. `python main.py --stage chunk`
6. `python main.py --stage graph`
7. `python main.py --stage embed --force`

Before step 7, enable multi-collection mode in `.env`:

- `CHROMADB_MULTI_COLLECTION_ENABLED=1`
- `CHROMADB_ENABLE_LEGACY_FALLBACK=1`

### Incremental update sequence (content only)

1. Re-run scrape for updated website/PDF content
2. Re-run entities and extractors if entities changed
3. Re-run chunk, graph, and embed
4. Launch chat and run query smoke tests

## 6. Validation and Quality Gates

### Unit tests

```bash
pytest -q
```

Coverage focus:

- scraper conversion and URL handling
- entity matching and fuzzy behavior
- chunking determinism and metadata behavior
- query routing logic
- phase-1 graph determinism and validation

### Retrieval quality checks

1. Run `main.py --stage query` with representative question sets
2. Run `evaluate_rag.py` for direct embedding/chroma sanity checks
3. Run `wide_quality_eval.py` for broad benchmark and diagnostics
4. Use validation reports in `data/validation/`

### Embedding-space diagnostics

1. Run `visualize_embeddings.py`
2. Review generated HTML in `data/embeddings/embedding_visualization.html`

## 7. Known Dual-Graph Design (Important)

The repo currently maintains two graph artifacts with different purposes:

1. Canonical graph in `data/graph/` from `chunker/knowledge_graph.py`
2. Phase-1 graph in `data/knowledge_graph/` from `knowledge_graph/builder.py`

This is intentional in current structure but should be kept synchronized conceptually, especially around relationship semantics and IDs used by chatbot and ingestion.

## 8. Operational Checklist

Before demo or release:

1. Verify `.env` values in use (Groq key, HF token, optional Tesseract path)
2. Confirm markdown and entity files are regenerated after source content changes
3. Confirm chunk counts and chunk type distribution are sensible
4. Confirm Chroma collection count matches expected chunk total (+ synthetic KG chunks)
5. Confirm top retrieval quality for faculty/course/teaching/regulation queries
6. Confirm chatbot source display corresponds to retrieved evidence

## 9. Suggested Ownership Map

1. Scraping + source conversion: `scraper/`
2. Knowledge representation and chunking: `chunker/` and `knowledge_graph/`
3. Retrieval and runtime: `rag_ingestion.py`, `chatbot.py`, `app.py`
4. Testing and evaluation: `tests/`, `evaluate_rag.py`, `visualize_embeddings.py`
5. Academic documentation: `Docs/` and `Agent_instructions/`

