"""
Tests for scraper module.
"""

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scraper.markdown_converter import (
    html_to_markdown,
    extract_faculty_data,
)
from scraper.url_discovery import URLDiscovery


class TestHTMLToMarkdown:
    """Test HTML to Markdown conversion."""

    def test_basic_conversion(self):
        html = """
        <html>
        <head><title>Test Page</title></head>
        <body>
            <h1>Hello World</h1>
            <p>This is a paragraph.</p>
        </body>
        </html>
        """
        result = html_to_markdown(html, "https://example.com/test")
        
        assert "# " in result or "Hello World" in result
        assert "paragraph" in result

    def test_preserves_headings(self):
        html = """
        <h1>Title</h1>
        <h2>Subtitle</h2>
        <h3>Section</h3>
        """
        result = html_to_markdown(html, "https://example.com", include_frontmatter=False)
        
        # Should have heading markers
        assert "#" in result

    def test_preserves_tables(self):
        html = """
        <table>
            <tr><th>Name</th><th>Age</th></tr>
            <tr><td>John</td><td>30</td></tr>
        </table>
        """
        result = html_to_markdown(html, "https://example.com", include_frontmatter=False)
        
        # Tables should contain pipe characters
        assert "|" in result

    def test_includes_frontmatter(self):
        html = "<h1>Test</h1>"
        result = html_to_markdown(html, "https://example.com/page")
        
        assert "---" in result
        assert "source_url:" in result
        assert "source_type: html" in result

    def test_removes_scripts(self):
        html = """
        <h1>Content</h1>
        <script>alert('evil');</script>
        <p>Safe content</p>
        """
        result = html_to_markdown(html, "https://example.com", include_frontmatter=False)
        
        assert "alert" not in result
        assert "Safe content" in result


class TestFacultyDataExtraction:
    """Test faculty data extraction from HTML."""

    def test_extract_basic_faculty(self):
        html = """
        <h1>Dr. John Doe</h1>
        <p>Professor & Head</p>
        <p>Qualification: Ph.D</p>
        <p>AICTE ID: 1-123456</p>
        """
        data = extract_faculty_data(html, "https://example.com/faculty/john")
        
        assert "John" in data['name'] or "Doe" in data['name']
        assert data['url'] == "https://example.com/faculty/john"

    def test_extract_designation(self):
        html = """
        <h1>Dr. Jane Smith</h1>
        <p>Associate Professor</p>
        """
        data = extract_faculty_data(html, "https://example.com/faculty/jane")
        
        assert "Associate Professor" in data['designation']


class TestURLDiscovery:
    """Test URL discovery logic."""

    def test_normalize_url(self):
        discovery = URLDiscovery()
        
        url1 = discovery._normalize_url("https://example.com/path/")
        url2 = discovery._normalize_url("https://example.com/path")
        
        assert url1 == url2

    def test_is_pdf_url(self):
        discovery = URLDiscovery()
        
        assert discovery._is_pdf_url("https://example.com/doc.pdf")
        assert discovery._is_pdf_url("https://example.com/doc.PDF")
        assert not discovery._is_pdf_url("https://example.com/page.html")

    def test_url_validation(self):
        discovery = URLDiscovery(
            base_url="https://mbcet.ac.in",
            include_patterns=[r"^https://mbcet\.ac\.in/departments/"],
            exclude_patterns=[r"\.jpg$"]
        )
        
        # Should be valid
        assert discovery._is_valid_url("https://mbcet.ac.in/departments/cse/")
        
        # Should be invalid (different domain)
        assert not discovery._is_valid_url("https://other.com/page")
        
        # Should be invalid (excluded pattern)
        assert not discovery._is_valid_url("https://mbcet.ac.in/image.jpg")


class TestURLDiscoveryEdgeCases:
    """Test edge cases for URL discovery."""

    def test_empty_html_link_extraction(self):
        discovery = URLDiscovery()
        links = discovery._extract_links("", "https://example.com")
        assert len(links) == 0

    def test_relative_url_resolution(self):
        discovery = URLDiscovery()
        html = '<a href="/page">Link</a>'
        links = discovery._extract_links(html, "https://example.com")
        
        # Should resolve to absolute URL
        assert any("example.com/page" in link for link in links)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
