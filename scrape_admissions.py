"""
Admissions Data Scraper for MBCET website.
Scrapes admission pages, downloads linked PDFs, and converts everything to Markdown.

Uses existing scraper utilities from the `scraper` package.
Outputs:
  - data/admissions/  -> scraped web page markdown + extracted PDF markdown
  - data/raw/         -> downloaded raw PDF files
"""

import sys
import re
import logging
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# Ensure project root is on sys.path so `config` and `scraper` resolve
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from scraper.html_scraper import HTMLScraper
from scraper.pdf_handler import PDFHandler

# --------------- Configuration ---------------
ADMISSION_URLS = [
    "https://mbcet.ac.in/admissions/b-tech/",
    "https://mbcet.ac.in/admissions/m-tech/",
    "https://mbcet.ac.in/admissions/ph-d/",
]

ADMISSIONS_DIR = config.DATA_DIR / "admissions"
RAW_PDF_DIR = config.RAW_DIR  # data/raw

# Friendly filenames for the scraped web pages
PAGE_FILENAMES = {
    "https://mbcet.ac.in/admissions/b-tech/": "btech_admissions.md",
    "https://mbcet.ac.in/admissions/m-tech/": "mtech_admissions.md",
    "https://mbcet.ac.in/admissions/ph-d/":   "phd_admissions.md",
}

# --------------- Logging ---------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("admissions_scraper")


# --------------- Helpers ---------------

def extract_pdf_links(html: str, base_url: str) -> list[dict]:
    """
    Extract all PDF links from an HTML page.
    Returns a list of dicts with 'url' and 'label' keys.
    Only includes PDFs that are admission-related (from the page content area).
    """
    soup = BeautifulSoup(html, "lxml")
    pdf_links = []
    seen_urls = set()

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        absolute_url = urljoin(base_url, href)

        # Only include .pdf links
        if not absolute_url.lower().endswith(".pdf"):
            continue

        # Deduplicate
        if absolute_url in seen_urls:
            continue
        seen_urls.add(absolute_url)

        label = a_tag.get_text(strip=True) or absolute_url.split("/")[-1]
        pdf_links.append({"url": absolute_url, "label": label})

    return pdf_links


def filter_admission_pdfs(pdf_links: list[dict], page_url: str) -> list[dict]:
    """
    Filter PDF links to only include those relevant to admissions.
    Excludes nav-bar / footer PDFs that appear on every page (e.g. NBA reports).
    """
    # Keywords that indicate admission-related PDFs
    admission_keywords = [
        "admission", "prospectus", "scholarship", "challan", "fee",
        "fitness", "physical", "brochure", "flyer", "curriculum",
        "document", "duty", "schedule", "allotment", "verification",
        "mtech", "m.tech", "m-tech", "btech", "b.tech", "b-tech",
        "phd", "ph.d", "seat", "rank",
    ]

    # URLs that are clearly NOT admission content (navbar/footer links)
    exclude_patterns = [
        "NBA-MBCET", "NBA-2019", "NB-A-2016", "WAR-ROOM",
    ]

    filtered = []
    for link in pdf_links:
        url_lower = link["url"].lower()
        label_lower = link["label"].lower()

        # Skip excluded patterns
        if any(pat.lower() in url_lower for pat in exclude_patterns):
            continue

        # Include if URL or label contains any admission keyword
        combined = url_lower + " " + label_lower
        if any(kw in combined for kw in admission_keywords):
            filtered.append(link)
            continue

        # Also include PDFs served from wp-content that don't match excludes
        # (they're likely admission-related if linked from the admissions page)
        if "wp-content/uploads" in url_lower:
            filtered.append(link)

    return filtered


def pdf_filename_to_md_name(pdf_filename: str) -> str:
    """Convert a PDF filename to a descriptive markdown filename."""
    stem = Path(pdf_filename).stem
    # Clean up the stem
    clean = re.sub(r"[_\-]+", "_", stem)
    clean = clean.strip("_")
    return f"pdf_{clean}.md"


# --------------- Main Pipeline ---------------

def main():
    logger.info("=" * 60)
    logger.info("MBCET Admissions Data Scraper")
    logger.info("=" * 60)

    # Ensure output directories exist
    ADMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize scrapers (reusing existing utilities)
    html_scraper = HTMLScraper()
    pdf_handler = PDFHandler(use_ocr=False)  # OCR not needed for these PDFs

    all_pdf_links = []

    # ──────────────────────────────────────────────
    # STEP 1: Scrape admission web pages to Markdown
    # ──────────────────────────────────────────────
    logger.info("\n── Step 1: Scraping admission web pages ──")

    for url in ADMISSION_URLS:
        logger.info(f"Scraping: {url}")

        # Fetch HTML
        html = html_scraper.fetch_page(url)
        if html is None:
            logger.error(f"FAILED to fetch {url}")
            continue

        # Convert to Markdown using existing converter
        from scraper.markdown_converter import html_to_markdown
        markdown = html_to_markdown(html, url)

        # Save with friendly filename
        filename = PAGE_FILENAMES.get(url, f"admission_page.md")
        output_path = ADMISSIONS_DIR / filename

        output_path.write_text(markdown, encoding="utf-8")
        logger.info(f"  ✓ Saved: {output_path.relative_to(PROJECT_ROOT)}")

        # Extract PDF links from this page
        page_pdfs = extract_pdf_links(html, url)
        admission_pdfs = filter_admission_pdfs(page_pdfs, url)
        logger.info(f"  Found {len(page_pdfs)} total PDF links, {len(admission_pdfs)} admission-related")

        for pdf in admission_pdfs:
            pdf["source_page"] = url
        all_pdf_links.extend(admission_pdfs)

        # Rate limiting
        time.sleep(config.REQUEST_DELAY)

    # Deduplicate PDF links across pages
    seen = set()
    unique_pdfs = []
    for pdf in all_pdf_links:
        if pdf["url"] not in seen:
            seen.add(pdf["url"])
            unique_pdfs.append(pdf)

    logger.info(f"\nTotal unique admission PDFs found: {len(unique_pdfs)}")
    for i, pdf in enumerate(unique_pdfs, 1):
        logger.info(f"  {i}. {pdf['label'][:60]}  →  {pdf['url'].split('/')[-1]}")

    # ──────────────────────────────────────────────
    # STEP 2: Download PDFs to data/raw/
    # ──────────────────────────────────────────────
    logger.info("\n── Step 2: Downloading PDFs to data/raw/ ──")

    downloaded_pdfs = []  # (url, local_path) pairs

    for pdf_info in unique_pdfs:
        url = pdf_info["url"]
        logger.info(f"Downloading: {url.split('/')[-1]}")

        pdf_path = pdf_handler.download_pdf(url, RAW_PDF_DIR)
        if pdf_path is not None:
            downloaded_pdfs.append((url, pdf_path))
            logger.info(f"  ✓ Saved: {pdf_path.relative_to(PROJECT_ROOT)}")
        else:
            logger.error(f"  ✗ FAILED: {url}")

        time.sleep(config.REQUEST_DELAY)

    logger.info(f"\nDownloaded {len(downloaded_pdfs)}/{len(unique_pdfs)} PDFs")

    # ──────────────────────────────────────────────
    # STEP 3: Extract PDF content to Markdown
    # ──────────────────────────────────────────────
    logger.info("\n── Step 3: Extracting PDF content to Markdown ──")

    converted_count = 0

    for url, pdf_path in downloaded_pdfs:
        logger.info(f"Converting: {pdf_path.name}")

        try:
            md_path = pdf_handler.pdf_to_markdown(pdf_path, ADMISSIONS_DIR)
            if md_path is not None:
                converted_count += 1
                logger.info(f"  ✓ Saved: {md_path.relative_to(PROJECT_ROOT)}")
            else:
                logger.warning(f"  ✗ No content extracted from {pdf_path.name}")
        except Exception as e:
            logger.error(f"  ✗ Error converting {pdf_path.name}: {e}")

        time.sleep(0.5)

    # ──────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    admissions_files = list(ADMISSIONS_DIR.glob("*.md"))
    raw_pdfs = [p for p in RAW_PDF_DIR.glob("*.pdf")
                if any(p.name == Path(dl[1]).name for dl in downloaded_pdfs)]

    logger.info(f"  data/admissions/  : {len(admissions_files)} Markdown files")
    for f in sorted(admissions_files):
        logger.info(f"    - {f.name}")

    logger.info(f"  data/raw/         : {len(downloaded_pdfs)} PDF files downloaded")
    for _, p in downloaded_pdfs:
        logger.info(f"    - {p.name}")

    logger.info(f"  PDF→Markdown      : {converted_count}/{len(downloaded_pdfs)} converted")
    logger.info("=" * 60)
    logger.info("Done!")


if __name__ == "__main__":
    main()
