"""
URL Discovery module for crawling MBCET website.
Discovers all relevant URLs for scraping.
"""

import re
import time
import logging
from urllib.parse import urljoin, urlparse
from typing import Set, List, Optional

import requests
from bs4 import BeautifulSoup

import config

logger = logging.getLogger(__name__)


class URLDiscovery:
    """
    Crawls the MBCET website to discover all relevant URLs.
    Focuses on CSE department, faculty pages, and academic content.
    """

    def __init__(
        self,
        base_url: str = config.BASE_URL,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ):
        self.base_url = base_url
        self.include_patterns = include_patterns or config.INCLUDE_PATTERNS
        self.exclude_patterns = exclude_patterns or config.EXCLUDE_PATTERNS
        
        # Compile regex patterns
        self._include_compiled = [re.compile(p, re.IGNORECASE) for p in self.include_patterns]
        self._exclude_compiled = [re.compile(p, re.IGNORECASE) for p in self.exclude_patterns]
        
        self.discovered_urls: Set[str] = set()
        self.pdf_urls: Set[str] = set()
        self.visited_urls: Set[str] = set()
        
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": config.USER_AGENT})

    def _normalize_url(self, url: str) -> str:
        """Normalize URL by removing fragments and trailing slashes."""
        parsed = urlparse(url)
        # Remove fragment and normalize path
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"
        if parsed.query:
            normalized += f"?{parsed.query}"
        return normalized

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL should be included in scraping."""
        # Must be from the same domain
        if not url.startswith(self.base_url):
            return False
        
        # Check exclude patterns first
        for pattern in self._exclude_compiled:
            if pattern.search(url):
                return False
        
        # Check include patterns
        for pattern in self._include_compiled:
            if pattern.search(url):
                return True
        
        return False

    def _is_pdf_url(self, url: str) -> bool:
        """Check if URL points to a PDF file."""
        return url.lower().endswith('.pdf')

    def _extract_links(self, html: str, base_url: str) -> Set[str]:
        """Extract all links from HTML content."""
        soup = BeautifulSoup(html, 'lxml')
        links = set()
        
        for anchor in soup.find_all('a', href=True):
            href = anchor['href']
            # Convert relative URLs to absolute
            absolute_url = urljoin(base_url, href)
            normalized = self._normalize_url(absolute_url)
            links.add(normalized)
        
        return links

    def _fetch_page(self, url: str) -> Optional[str]:
        """Fetch a page with retry logic."""
        for attempt in range(config.MAX_RETRIES):
            try:
                response = self.session.get(
                    url,
                    timeout=config.REQUEST_TIMEOUT,
                    allow_redirects=True
                )
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < config.MAX_RETRIES - 1:
                    time.sleep(config.REQUEST_DELAY * (attempt + 1))
        
        logger.error(f"Failed to fetch {url} after {config.MAX_RETRIES} attempts")
        return None

    def discover(
        self,
        start_urls: Optional[List[str]] = None,
        max_depth: int = 3,
        max_urls: int = 200
    ) -> dict:
        """
        Discover all relevant URLs starting from the given URLs.
        
        Args:
            start_urls: List of URLs to start crawling from
            max_depth: Maximum crawl depth
            max_urls: Maximum number of URLs to discover
            
        Returns:
            Dictionary with 'pages' and 'pdfs' lists
        """
        if start_urls is None:
            start_urls = [config.CSE_DEPARTMENT_URL]
        
        # Queue: (url, depth)
        queue = [(url, 0) for url in start_urls]
        
        while queue and len(self.discovered_urls) < max_urls:
            url, depth = queue.pop(0)
            url = self._normalize_url(url)
            
            if url in self.visited_urls:
                continue
            
            self.visited_urls.add(url)
            
            # Check if it's a PDF
            if self._is_pdf_url(url):
                if self._is_valid_url(url) or 'syllabus' in url.lower() or 'regulation' in url.lower():
                    self.pdf_urls.add(url)
                    logger.info(f"Found PDF: {url}")
                continue
            
            # Check if URL is valid for our scope
            if not self._is_valid_url(url):
                continue
            
            self.discovered_urls.add(url)
            logger.info(f"Discovered: {url} (depth={depth})")
            
            # Don't go deeper than max_depth
            if depth >= max_depth:
                continue
            
            # Fetch and extract links
            html = self._fetch_page(url)
            if html is None:
                continue
            
            # Respect rate limiting
            time.sleep(config.REQUEST_DELAY)
            
            # Extract and queue new links
            links = self._extract_links(html, url)
            for link in links:
                if link not in self.visited_urls:
                    queue.append((link, depth + 1))
        
        return {
            'pages': sorted(list(self.discovered_urls)),
            'pdfs': sorted(list(self.pdf_urls))
        }


def discover_urls(
    start_urls: Optional[List[str]] = None,
    max_depth: int = 3,
    max_urls: int = 200
) -> dict:
    """
    Convenience function to discover URLs.
    
    Returns:
        Dictionary with 'pages' and 'pdfs' lists
    """
    discovery = URLDiscovery()
    return discovery.discover(start_urls, max_depth, max_urls)


if __name__ == "__main__":
    # Test URL discovery
    logging.basicConfig(level=logging.INFO)
    result = discover_urls(max_depth=2, max_urls=50)
    print(f"\nDiscovered {len(result['pages'])} pages and {len(result['pdfs'])} PDFs")
    print("\nPages:")
    for url in result['pages'][:10]:
        print(f"  - {url}")
    if result['pdfs']:
        print("\nPDFs:")
        for url in result['pdfs']:
            print(f"  - {url}")
