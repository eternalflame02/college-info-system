"""
HTML Page Scraper for MBCET website.
Fetches pages and saves them as Markdown files.
"""

import time
import logging
import hashlib
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional, List, Dict

import requests

import config
from scraper.markdown_converter import html_to_markdown, extract_faculty_data

logger = logging.getLogger(__name__)


class HTMLScraper:
    """
    Scrapes HTML pages from MBCET website and converts to Markdown.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": config.USER_AGENT})
        self.scraped_pages: Dict[str, str] = {}  # url -> output_path

    def _get_output_filename(self, url: str) -> str:
        """Generate a filename from URL."""
        parsed = urlparse(url)
        path = parsed.path.strip('/')
        
        if not path:
            return "index.md"
        
        # Convert path to filename
        filename = path.replace('/', '_')
        
        # Remove common prefixes
        filename = filename.replace('departments_', '')
        filename = filename.replace('faculty_', 'faculty_')
        
        # Add hash suffix for uniqueness
        url_hash = hashlib.md5(url.encode()).hexdigest()[:6]
        
        # Ensure .md extension
        if not filename.endswith('.md'):
            filename = f"{filename}_{url_hash}.md"
        
        return filename

    def fetch_page(self, url: str) -> Optional[str]:
        """
        Fetch HTML content from a URL.
        
        Args:
            url: URL to fetch
            
        Returns:
            HTML content or None if failed
        """
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

    def scrape_page(
        self,
        url: str,
        output_dir: Optional[Path] = None
    ) -> Optional[str]:
        """
        Scrape a single page and save as Markdown.
        
        Args:
            url: URL to scrape
            output_dir: Directory to save Markdown file
            
        Returns:
            Path to saved Markdown file or None if failed
        """
        output_dir = output_dir or config.MARKDOWN_PAGES_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Scraping: {url}")
        
        # Fetch HTML
        html = self.fetch_page(url)
        if html is None:
            return None
        
        # Convert to Markdown
        try:
            markdown = html_to_markdown(html, url)
        except Exception as e:
            logger.error(f"Failed to convert {url} to Markdown: {e}")
            return None
        
        # Save to file
        filename = self._get_output_filename(url)
        output_path = output_dir / filename
        
        try:
            output_path.write_text(markdown, encoding='utf-8')
            logger.info(f"Saved: {output_path}")
            self.scraped_pages[url] = str(output_path)
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to save {output_path}: {e}")
            return None

    def scrape_pages(
        self,
        urls: List[str],
        output_dir: Optional[Path] = None
    ) -> Dict[str, str]:
        """
        Scrape multiple pages.
        
        Args:
            urls: List of URLs to scrape
            output_dir: Directory to save Markdown files
            
        Returns:
            Dictionary mapping URLs to output paths
        """
        results = {}
        
        for i, url in enumerate(urls):
            logger.info(f"Progress: {i + 1}/{len(urls)}")
            
            output_path = self.scrape_page(url, output_dir)
            if output_path:
                results[url] = output_path
            
            # Rate limiting
            if i < len(urls) - 1:
                time.sleep(config.REQUEST_DELAY)
        
        logger.info(f"Scraped {len(results)}/{len(urls)} pages successfully")
        return results

    def extract_faculty_list(self, department_url: str) -> List[Dict]:
        """
        Extract list of faculty from a department page.
        
        Args:
            department_url: URL of the department page
            
        Returns:
            List of faculty data dictionaries
        """
        html = self.fetch_page(department_url)
        if html is None:
            return []
        
        from bs4 import BeautifulSoup
        import re
        
        soup = BeautifulSoup(html, 'lxml')
        faculty_list = []
        
        # Find faculty links
        for link in soup.find_all('a', href=True):
            href = link['href']
            if '/faculty/' in href and href.startswith(config.BASE_URL):
                name = link.get_text(strip=True)
                if name and len(name) > 3:  # Filter out empty or very short names
                    faculty_list.append({
                        'name': name,
                        'url': href
                    })
        
        # Remove duplicates
        seen = set()
        unique_faculty = []
        for f in faculty_list:
            if f['url'] not in seen:
                seen.add(f['url'])
                unique_faculty.append(f)
        
        return unique_faculty


def scrape_page(url: str) -> Optional[str]:
    """Convenience function to scrape a single page."""
    scraper = HTMLScraper()
    return scraper.scrape_page(url)


def scrape_all_pages(urls: List[str]) -> Dict[str, str]:
    """Convenience function to scrape multiple pages."""
    scraper = HTMLScraper()
    return scraper.scrape_pages(urls)


if __name__ == "__main__":
    # Test scraping
    logging.basicConfig(level=logging.INFO)
    
    # Test with CSE department page
    result = scrape_page(config.CSE_DEPARTMENT_URL)
    if result:
        print(f"Saved to: {result}")
    else:
        print("Scraping failed")
