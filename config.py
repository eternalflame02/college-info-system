"""
Configuration module for MBCET CSE Semantic Chunking Pipeline.
Loads settings from environment variables and provides defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ========== OCR Configuration ==========
TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if TESSERACT_CMD:
    # Configure pytesseract if available
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    except ImportError:
        pass

# ========== Base Paths ==========
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MARKDOWN_DIR = DATA_DIR / "markdown"
MARKDOWN_PAGES_DIR = MARKDOWN_DIR / "pages"
MARKDOWN_PDFS_DIR = MARKDOWN_DIR / "pdfs"
ENTITIES_DIR = DATA_DIR / "entities"
CHUNKS_DIR = DATA_DIR / "chunks"
LOGS_DIR = PROJECT_ROOT / "logs"

# ========== URL Configuration ==========
BASE_URL = os.getenv("BASE_URL", "https://mbcet.ac.in")
CSE_DEPARTMENT_URL = os.getenv(
    "CSE_DEPARTMENT_URL",
    "https://mbcet.ac.in/departments/computer-science-engineering/"
)

# URLs to scrape (CSE focused)
SCRAPE_URLS = [
    CSE_DEPARTMENT_URL,
    # Faculty pages will be discovered dynamically
]

# URL patterns to include during crawling
INCLUDE_PATTERNS = [
    r"^https://mbcet\.ac\.in/departments/computer-science-engineering",
    r"^https://mbcet\.ac\.in/faculty/",
    r"^https://mbcet\.ac\.in/programmes/",
]

# URL patterns to exclude
EXCLUDE_PATTERNS = [
    r"/wp-content/",
    r"/wp-admin/",
    r"\.(jpg|jpeg|png|gif|svg|css|js)$",
]

# ========== Scraping Configuration ==========
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "1.0"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
REQUEST_TIMEOUT = 30
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# ========== Chunking Configuration ==========
MIN_CHUNK_WORDS = int(os.getenv("MIN_CHUNK_WORDS", "80"))
MAX_CHUNK_WORDS = int(os.getenv("MAX_CHUNK_WORDS", "700"))
PREFERRED_CHUNK_WORDS = int(os.getenv("PREFERRED_CHUNK_WORDS", "450"))
SOFT_LIMIT_WORDS = int(os.getenv("SOFT_LIMIT_WORDS", "600"))
MAX_TABLE_ROWS_PER_CHUNK = 50

# ========== Entity Registry Files ==========
FACULTY_FILE = ENTITIES_DIR / "faculty.json"
COURSES_FILE = ENTITIES_DIR / "courses.json"
PROGRAMS_FILE = ENTITIES_DIR / "programs.json"

# ========== Output Files ==========
CHUNKS_FILE = CHUNKS_DIR / "chunks.json"
CHUNK_REPORT_FILE = CHUNKS_DIR / "chunk_report.json"
ERRORS_LOG_FILE = CHUNKS_DIR / "errors.log"
CHUNKER_LOG_FILE = LOGS_DIR / "chunker.log"

# ========== Faculty Detection Patterns ==========
FACULTY_SECTION_KEYWORDS = [
    "faculty",
    "the people",
    "our team",
    "staff",
    "professors",
    "teaching staff",
]

FACULTY_DESIGNATION_PATTERN = (
    r"(Professor|Associate Professor|Assistant Professor|"
    r"Head of Department|Lecturer|HOD|Head ITMS)"
)

FACULTY_NAME_PATTERN = (
    r"(Dr\.?\s*|Prof\.?\s*|Mr\.?\s*|Ms\.?\s*|Mrs\.?\s*)"
    r"[A-Z][a-z]+(\s+[A-Z]\.?\s*)*(\s+[A-Z][a-z]+)+"
)


def ensure_directories():
    """Create all required directories if they don't exist."""
    directories = [
        DATA_DIR,
        RAW_DIR,
        MARKDOWN_DIR,
        MARKDOWN_PAGES_DIR,
        MARKDOWN_PDFS_DIR,
        ENTITIES_DIR,
        CHUNKS_DIR,
        LOGS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# Create directories on import
ensure_directories()
