---
description: How to run the complete end-to-end pipeline (scrape, graph, embed, serve)
---

# Running the Complete Pipeline

This workflow documents how to run the full end-to-end pipeline used by the current repository.

## Prerequisites

- Python 3.11+ installed
- Virtual environment activated
- Dependencies installed (`pip install -r requirements.txt`)
- `.env` file configured (copy from `.env.example`)

## Workflow Steps

### 1. Activate Virtual Environment

```bash
# Windows
.\venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 2. Run Complete Pipeline (Recommended)

```bash
python main.py --stage all -v
```

This runs all major stages in sequence:
1. Scraping (HTML + PDFs)
2. Entity registry
3. Canonical KG (`kg`)
4. Semantic chunking
5. Phase-1 deterministic graph (`graph`)
6. Embedding + ChromaDB ingestion (`embed`)

### 3. Run Individual Stages (Optional)

If you need to run stages separately:

```bash
# Stage 1: Scrape website
python main.py --stage scrape

# Stage 2: Build entity registry  
python main.py --stage entities

# Stage 3: Build canonical KG
python main.py --stage kg

# Stage 4: Generate chunks
python main.py --stage chunk

# Stage 5: Build deterministic phase-1 graph
python main.py --stage graph

# Stage 6: Embed and ingest
python main.py --stage embed
```

### 4. Start Runtime Server

```bash
python main.py --stage serve
```

Alternative alias:

```bash
python main.py --stage chat
```

### 5. Optional Public Exposure (ngrok)

Recommended in separate terminal:

```bash
ngrok http --domain=hyo-gymnocarpous-lingeringly.ngrok-free.dev 8000
```

Optional one-command mode (env-driven):

- `AUTO_START_NGROK=1`
- `NGROK_DOMAIN=hyo-gymnocarpous-lingeringly.ngrok-free.dev`
- run `python main.py --stage serve`

### 6. Verify Output

After pipeline completion, check:

```bash
# Scraped Markdown files
ls data/markdown/pages/
ls data/markdown/pdfs/

# Entity registries
cat data/entities/faculty.json

# Generated chunks
ls data/chunks/

# Deterministic graph artifacts
ls data/knowledge_graph/

# Validation artifacts
ls data/validation/
```

## Expected Output

| Stage | Output Location | Typical Outcome |
|-------|-----------------|-----------------|
| HTML Scraping | `data/markdown/pages/` | page markdown artifacts |
| PDF Processing | `data/markdown/pdfs/` | pdf markdown artifacts |
| Entity Registry | `data/entities/*.json` | faculty/course/program files |
| Chunking | `data/chunks/chunks.json` | semantic chunk corpus |
| Phase-1 Graph | `data/knowledge_graph/graph.json` | deterministic graph |
| Validation | `data/validation/*.json` | ingestion and retrieval reports |

## Troubleshooting

### Network Issues
If scraping fails, check:
- Internet connection
- MBCET website availability
- Increase `REQUEST_TIMEOUT` in `.env`

### PDF Processing Errors
If PDFs fail to process:
- Ensure Tesseract OCR is installed (for scanned PDFs)
- Check `TESSERACT_CMD` path in `.env`

### Server Port Conflicts

If port 8000 is occupied:

```powershell
Get-NetTCPConnection -LocalPort 8000 | Select-Object OwningProcess, State
Stop-Process -Id <PID> -Force
```

Then restart `python main.py --stage serve`.

### ngrok Interstitial Warning

On free ngrok, browser may show a one-time warning page.
API calls can bypass via header:

`ngrok-skip-browser-warning: 1`

### Missing Dependencies
```bash
pip install -r requirements.txt --upgrade
```
