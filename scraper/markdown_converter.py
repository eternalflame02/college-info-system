"""
HTML to Markdown converter for MBCET pages.
Converts scraped HTML content to structured Markdown with frontmatter.
"""

import re
from datetime import datetime
from urllib.parse import urlparse
from typing import Optional

from bs4 import BeautifulSoup, NavigableString, Tag
from markdownify import markdownify, MarkdownConverter

import config


class MBCETMarkdownConverter(MarkdownConverter):
    """
    Custom Markdown converter optimized for MBCET pages.
    Preserves headers, tables, and lists while cleaning up noise.
    Compatible with markdownify 1.2.2+
    """
    pass  # Use base class, custom conversion done via options


def _clean_small_images(soup: BeautifulSoup) -> BeautifulSoup:
    """Remove small decorative images before conversion."""
    for img in soup.find_all('img'):
        # Skip tiny icons and decorative images
        width = img.get('width', '')
        height = img.get('height', '')
        alt = img.get('alt', '')
        
        try:
            if width and int(width) < 50:
                img.decompose()
                continue
            if height and int(height) < 50:
                img.decompose()
                continue
        except (ValueError, TypeError):
            pass
        
        # Remove images without alt text (likely decorative)
        if not alt:
            img.decompose()
    
    return soup


def _clean_unwanted_links(soup: BeautifulSoup) -> BeautifulSoup:
    """Clean up unwanted links."""
    for a in soup.find_all('a'):
        href = a.get('href', '')
        # Remove javascript and mailto links but keep the text
        if href.startswith(('javascript:', 'mailto:', 'tel:')):
            a.unwrap()
    return soup



def _clean_html(soup: BeautifulSoup) -> BeautifulSoup:
    """Remove unwanted elements from HTML."""
    # Remove scripts, styles, and other non-content elements
    for tag in soup.find_all(['script', 'style', 'noscript', 'iframe', 'nav', 'footer']):
        tag.decompose()
    
    # Remove common noise elements
    noise_classes = [
        'cookie-notice', 'popup', 'modal', 'social-share',
        'sidebar', 'advertisement', 'ad-', 'menu', 'navigation'
    ]
    for class_pattern in noise_classes:
        for tag in soup.find_all(class_=re.compile(class_pattern, re.IGNORECASE)):
            tag.decompose()
    
    # Remove elements with IDs that indicate noise
    noise_ids = ['cookie', 'popup', 'modal', 'sidebar', 'menu', 'nav']
    for id_pattern in noise_ids:
        for tag in soup.find_all(id=re.compile(id_pattern, re.IGNORECASE)):
            tag.decompose()
    
    return soup


def _extract_main_content(soup: BeautifulSoup) -> Optional[Tag]:
    """Extract the main content area from the page."""
    # Try common content selectors
    selectors = [
        ('main', {}),
        ('article', {}),
        ('div', {'class': re.compile(r'content|main|entry|post', re.IGNORECASE)}),
        ('div', {'id': re.compile(r'content|main|entry|post', re.IGNORECASE)}),
        ('div', {'class': 'elementor-widget-container'}),
    ]
    
    for tag_name, attrs in selectors:
        content = soup.find(tag_name, attrs)
        if content:
            return content
    
    # Fallback to body
    return soup.find('body')


def _extract_title(soup: BeautifulSoup, url: str) -> str:
    """Extract page title from HTML."""
    # Try h1 first
    h1 = soup.find('h1')
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)
    
    # Try title tag
    title_tag = soup.find('title')
    if title_tag and title_tag.get_text(strip=True):
        title = title_tag.get_text(strip=True)
        # Remove site name suffix
        title = re.sub(r'\s*[-–|]\s*Mar Baselios.*$', '', title, flags=re.IGNORECASE)
        return title.strip()
    
    # Fallback to URL path
    parsed = urlparse(url)
    path = parsed.path.strip('/').split('/')[-1]
    return path.replace('-', ' ').replace('_', ' ').title()


def _generate_frontmatter(
    title: str,
    url: str,
    scraped_at: Optional[str] = None
) -> str:
    """Generate YAML frontmatter for the Markdown file."""
    scraped_at = scraped_at or datetime.now().isoformat()
    
    frontmatter = f"""---
title: "{title}"
source_url: "{url}"
source_type: html
scraped_at: "{scraped_at}"
---

"""
    return frontmatter


def _clean_markdown(markdown: str) -> str:
    """Clean up markdown output."""
    # Remove excessive blank lines
    markdown = re.sub(r'\n{3,}', '\n\n', markdown)
    
    # Remove lines that are just whitespace
    lines = markdown.split('\n')
    cleaned_lines = []
    for line in lines:
        if line.strip() or (cleaned_lines and cleaned_lines[-1].strip()):
            cleaned_lines.append(line.rstrip())
    
    markdown = '\n'.join(cleaned_lines)
    
    # Fix header formatting
    markdown = re.sub(r'^(#+)([^\s#])', r'\1 \2', markdown, flags=re.MULTILINE)
    
    # Remove trailing whitespace
    markdown = '\n'.join(line.rstrip() for line in markdown.split('\n'))
    
    return markdown.strip()


def html_to_markdown(
    html: str,
    source_url: str,
    include_frontmatter: bool = True
) -> str:
    """
    Convert HTML content to structured Markdown.
    
    Args:
        html: Raw HTML content
        source_url: Original URL of the page
        include_frontmatter: Whether to include YAML frontmatter
        
    Returns:
        Markdown string with optional frontmatter
    """
    soup = BeautifulSoup(html, 'lxml')
    
    # Clean HTML
    soup = _clean_html(soup)
    
    # Extract title
    title = _extract_title(soup, source_url)
    
    # Extract main content
    content = _extract_main_content(soup)
    if content is None:
        content = soup.find('body') or soup
    
    # Additional cleaning for images and links
    content = _clean_small_images(content)
    content = _clean_unwanted_links(content)
    
    # Convert to Markdown using plain markdownify function
    markdown = markdownify(
        str(content),
        heading_style='atx',
        bullets='-',
        strong_em_symbol='*',
        strip=['script', 'style'],
    )
    
    # Clean up markdown
    markdown = _clean_markdown(markdown)
    
    # Add title as H1 if not present
    if not markdown.startswith('#'):
        markdown = f"# {title}\n\n{markdown}"
    
    # Add frontmatter
    if include_frontmatter:
        frontmatter = _generate_frontmatter(title, source_url)
        markdown = frontmatter + markdown
    
    return markdown


def extract_faculty_data(html: str, url: str) -> dict:
    """
    Extract structured faculty data from a faculty profile page.
    
    Returns:
        Dictionary with faculty information
    """
    soup = BeautifulSoup(html, 'lxml')
    
    data = {
        'name': '',
        'designation': '',
        'qualification': '',
        'email': '',
        'aicte_id': '',
        'department': '',
        'url': url,
    }
    
    # Extract name from h1 or title
    h1 = soup.find('h1')
    if h1:
        data['name'] = h1.get_text(strip=True)
    
    # Look for structured profile data
    text = soup.get_text()
    
    # Extract designation
    designation_match = re.search(
        r'(Professor\s*(?:&|and)?\s*Head\s*\w*|'
        r'Associate\s*Professor|'
        r'Assistant\s*Professor|'
        r'Professor|'
        r'Lecturer)',
        text,
        re.IGNORECASE
    )
    if designation_match:
        data['designation'] = designation_match.group(1).strip()
    
    # Extract qualification
    qual_match = re.search(
        r'Qualification\s*:?\s*([^\n]+)',
        text,
        re.IGNORECASE
    )
    if qual_match:
        data['qualification'] = qual_match.group(1).strip()
    
    # Extract AICTE ID
    aicte_match = re.search(
        r'AICTE\s*ID\s*:?-?\s*(\d[\d-]+)',
        text,
        re.IGNORECASE
    )
    if aicte_match:
        data['aicte_id'] = aicte_match.group(1).strip()
    
    return data


if __name__ == "__main__":
    # Test with sample HTML
    sample_html = """
    <html>
    <head><title>Computer Science - MBCET</title></head>
    <body>
        <main>
            <h1>Computer Science & Engineering</h1>
            <h2>Vision</h2>
            <p>To be a Centre of Excellence in Computer Science.</p>
            <h2>Faculty</h2>
            <h3>Dr. Jisha John</h3>
            <p>Professor & Head</p>
            <p>Qualification: Ph.D</p>
        </main>
    </body>
    </html>
    """
    
    markdown = html_to_markdown(sample_html, "https://mbcet.ac.in/test")
    print(markdown)
