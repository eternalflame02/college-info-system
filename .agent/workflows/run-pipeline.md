---
description: How to run the complete semantic chunking pipeline
---

# Running the Complete Pipeline

This workflow documents how to run the full semantic chunking pipeline.

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

This runs all three stages in sequence:
1. Web scraping (HTML + PDFs)
2. Entity registry building
3. Semantic chunking

### 3. Run Individual Stages (Optional)

If you need to run stages separately:

```bash
# Stage 1: Scrape website
python main.py --stage scrape

# Stage 2: Build entity registry  
python main.py --stage entities

# Stage 3: Generate chunks
python main.py --stage chunk
```

### 4. Verify Output

After pipeline completion, check:

```bash
# Scraped Markdown files
ls data/markdown/pages/
ls data/markdown/pdfs/

# Entity registries
cat data/entities/faculty.json

# Generated chunks
ls data/chunks/
```

## Expected Output

| Stage | Output Location | Expected Count |
|-------|-----------------|----------------|
| HTML Scraping | `data/markdown/pages/` | ~60 files |
| PDF Processing | `data/markdown/pdfs/` | ~10 files |
| Entity Registry | `data/entities/faculty.json` | ~43 entities |
| Chunking | `data/chunks/chunks.json` | ~640 chunks |

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

### Missing Dependencies
```bash
pip install -r requirements.txt --upgrade
```
