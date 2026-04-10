# MBCET CSE Semantic Chunking Pipeline

A Python-based pipeline for scraping, processing, and semantically chunking MBCET CSE department content for RAG (Retrieval-Augmented Generation) and Knowledge Graph applications.

![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Pipeline Stages](#-pipeline-stages)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Output Formats](#-output-formats)
- [Testing](#-testing)
- [Contributing](#-contributing)

---

## ✨ Features

- **Web Scraping**: Crawls MBCET website with intelligent URL discovery
- **PDF Processing**: Extracts text and tables from syllabus PDFs using `pdfplumber`
- **OCR Support**: Handles scanned documents with Tesseract OCR
- **Table Extraction**: Preserves tabular data as structured Markdown tables
- **Entity Registry**: Extracts and normalizes faculty names with aliases
- **Semantic Chunking**: Creates semantically meaningful chunks for RAG
- **Duplicate Detection**: SHA-256 based deduplication

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MBCET Website                            │
│                    (mbcet.ac.in/cse)                            │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     URL Discovery                               │
│              (Crawls pages, identifies PDFs)                    │
└─────────────────┬───────────────────────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
┌───────────────┐   ┌───────────────┐
│ HTML Scraper  │   │ PDF Handler   │
│ (markdownify) │   │ (pdfplumber)  │
└───────┬───────┘   └───────┬───────┘
        │                   │
        └─────────┬─────────┘
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Markdown Files                               │
│                  (data/markdown/)                               │
└─────────────────┬───────────────────────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
┌───────────────┐   ┌───────────────┐
│Entity Registry│   │   Semantic    │
│   Builder     │   │   Chunker     │
└───────┬───────┘   └───────┬───────┘
        │                   │
        ▼                   ▼
┌───────────────┐   ┌───────────────┐
│ faculty.json  │   │ chunks.json   │
│ courses.json  │   │ 640 chunks    │
└───────────────┘   └───────────────┘
```

---

## 🚀 Installation

### Prerequisites

- Python 3.11 or higher
- Tesseract OCR (for scanned PDF support)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/mbcet-chunking-pipeline.git
cd mbcet-chunking-pipeline
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install Tesseract OCR (Optional - for scanned PDFs)

**Windows:**
- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Add to PATH or set in `.env`

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

### Step 5: Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

---

## 📖 Usage

### Run Complete Pipeline

```bash
python main.py --stage all
```

### Run Individual Stages

```bash
# Stage 1: Scrape website and PDFs
python main.py --stage scrape

# Stage 2: Build entity registry
python main.py --stage entities

# Stage 3: Run semantic chunker
python main.py --stage chunk
```

### Verbose Mode

```bash
python main.py --stage all -v
```

---

## 🔄 Pipeline Stages

### 1. Web Scraping (`--stage scrape`)

**Purpose:** Crawls MBCET CSE website and converts content to Markdown.

**What it does:**
- Discovers all pages under `/cse` using BFS crawling
- Identifies PDF links (syllabi, regulations)
- Converts HTML pages to clean Markdown
- Extracts tables from PDFs using `pdfplumber`
- Falls back to OCR for scanned documents

**Output:**
- `data/markdown/pages/` - 60+ HTML-derived Markdown files
- `data/markdown/pdfs/` - 10 PDF-derived Markdown files

### 2. Entity Registry (`--stage entities`)

**Purpose:** Extracts named entities for knowledge graph construction.

**What it does:**
- Parses faculty list from department page
- Normalizes names (removes titles like Dr., Prof.)
- Generates aliases for entity linking
- Creates unique entity IDs

**Output:**
- `data/entities/faculty.json` - 43 faculty entities
- `data/entities/courses.json` - Course entities (planned)
- `data/entities/programs.json` - Program entities (planned)

### 3. Semantic Chunking (`--stage chunk`)

**Purpose:** Splits documents into semantically meaningful chunks.

**What it does:**
- Parses Markdown structure (headers, lists, tables, paragraphs)
- Classifies content types (profile, regulation, table, section, list)
- Links chunks to entities (faculty references)
- Detects and skips duplicates (SHA-256 hashing)
- Tracks page ranges for PDF sources

**Output:**
- `data/chunks/chunks.json` - 640 semantic chunks
- `data/chunks/chunk_report.json` - Statistics and summary

---

## 📁 Project Structure

```
mbcet-chunking-pipeline/
├── main.py                 # CLI entry point
├── config.py               # Configuration and paths
├── requirements.txt        # Python dependencies
├── .env.example            # Environment template
│
├── scraper/                # Web scraping module
│   ├── __init__.py
│   ├── url_discovery.py    # BFS URL crawler
│   ├── html_scraper.py     # HTML to Markdown converter
│   ├── pdf_handler.py      # PDF processing with pdfplumber
│   └── markdown_converter.py # HTML cleaning and conversion
│
├── chunker/                # Semantic chunking module
│   ├── __init__.py
│   ├── semantic_chunker.py # Main chunking logic
│   ├── entity_registry.py  # Entity extraction and normalization
│   └── chunk_classifiers.py # Content type classification
│
├── tests/                  # Unit tests
│   ├── test_scraper.py
│   ├── test_chunker.py
│   └── test_entities.py
│
├── data/                   # Generated data (gitignored)
│   ├── raw/                # Downloaded PDFs
│   ├── markdown/           # Converted Markdown files
│   │   ├── pages/          # From HTML pages
│   │   └── pdfs/           # From PDF documents
│   ├── entities/           # Entity registries
│   └── chunks/             # Final chunks
│
└── Docs/                   # Reference documents
```

---

## ⚙️ Configuration

Configuration is managed via `config.py` and `.env`:

### Environment Variables (`.env`)

```bash
# Base URL for scraping
BASE_URL=https://mbcet.ac.in
CSE_DEPARTMENT_URL=https://mbcet.ac.in/cse

# Request settings
REQUEST_TIMEOUT=30
REQUEST_DELAY=1.0
MAX_RETRIES=3

# Tesseract path (Windows)
TESSERACT_CMD=C:/Program Files/Tesseract-OCR/tesseract.exe
```

### Chunking Settings (`config.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `MAX_CHUNK_WORDS` | 500 | Maximum words per chunk |
| `MIN_CHUNK_WORDS` | 50 | Minimum words per chunk |
| `OVERLAP_SENTENCES` | 2 | Sentence overlap between chunks |

---

## 📤 Output Formats

### Chunks JSON Schema

```json
{
  "chunk_id": "pdf_cse_syllabus_root_a1b2c3d4",
  "text": "## Data Structures\n\nModule 1: Arrays and linked lists...",
  "source_type": "pdf",
  "source_file": "data/markdown/pdfs/CSE_Syllabus.md",
  "section_hierarchy": ["Semester 3", "Data Structures"],
  "content_type": "regulation",
  "entity_refs": ["faculty_dr_john_doe"],
  "page_range": [15, 17],
  "word_count": 245,
  "hash": "sha256:abc123..."
}
```

### Entity JSON Schema

```json
{
  "id": "faculty_dr_john_doe",
  "name": "Dr. John Doe",
  "aliases": ["John Doe", "Dr John Doe"],
  "type": "faculty",
  "url": "https://mbcet.ac.in/cse/faculty/john-doe"
}
```

---

## 🧪 Testing

### Run All Tests

```bash
pytest
```

### Run with Coverage

```bash
pytest --cov=scraper --cov=chunker --cov-report=html
```

### Run Specific Tests

```bash
pytest tests/test_scraper.py -v
pytest tests/test_chunker.py -v
```

---

## 🔧 Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `requests` | ≥2.31.0 | HTTP client |
| `beautifulsoup4` | ≥4.12.0 | HTML parsing |
| `markdownify` | ≥0.12.0 | HTML to Markdown |
| `pdfplumber` | ≥0.11.0 | PDF table extraction |
| `pypdf` | ≥4.0.0 | PDF text extraction |
| `pytesseract` | ≥0.3.10 | OCR integration |
| `pyyaml` | ≥6.0.0 | YAML frontmatter |
| `tqdm` | ≥4.66.0 | Progress bars |
| `pytest` | ≥8.0.0 | Testing framework |

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Total pages scraped | 60 |
| Total PDFs processed | 10 |
| Chunks generated | 640 |
| Duplicates detected | 1 |
| Faculty entities | 43 |
| Test coverage | 50 tests passing |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Rohith** - *Initial work* - MBCET CSE Department

---

## 🙏 Acknowledgments

- MBCET CSE Department for the source content
- pdfplumber team for excellent table extraction
- Tesseract OCR community
