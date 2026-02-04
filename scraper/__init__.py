"""
Scraper module for MBCET CSE Semantic Chunking Pipeline.
"""

from scraper.html_scraper import scrape_page, scrape_all_pages
from scraper.markdown_converter import html_to_markdown
from scraper.pdf_handler import download_pdf, pdf_to_markdown
from scraper.url_discovery import discover_urls

__all__ = [
    "scrape_page",
    "scrape_all_pages",
    "html_to_markdown",
    "download_pdf",
    "pdf_to_markdown",
    "discover_urls",
]
