"""
Tests for semantic chunker module.
"""

import pytest
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from chunker.chunk_models import Chunk, ChunkReport
from chunker.semantic_chunker import (
    normalize_for_hash,
    compute_hash,
    slugify,
    generate_chunk_id,
    remove_frontmatter,
    MarkdownChunker,
)


class TestNormalization:
    """Test text normalization functions."""

    def test_normalize_removes_frontmatter(self):
        text = "---\ntitle: Test\n---\n\n# Hello"
        result = normalize_for_hash(text)
        assert "title" not in result
        assert "hello" in result

    def test_normalize_lowercase(self):
        text = "HELLO World"
        result = normalize_for_hash(text)
        assert result == "hello world"

    def test_normalize_collapses_whitespace(self):
        text = "hello    world\n\n\ntest"
        result = normalize_for_hash(text)
        assert result == "hello world test"


class TestHashing:
    """Test hash computation."""

    def test_hash_deterministic(self):
        text = "Hello World"
        hash1 = compute_hash(text)
        hash2 = compute_hash(text)
        assert hash1 == hash2

    def test_hash_different_for_different_text(self):
        hash1 = compute_hash("Hello")
        hash2 = compute_hash("World")
        assert hash1 != hash2

    def test_hash_ignores_case(self):
        hash1 = compute_hash("Hello")
        hash2 = compute_hash("hello")
        assert hash1 == hash2


class TestSlugify:
    """Test slug generation."""

    def test_slugify_basic(self):
        result = slugify("Hello World")
        assert result == "hello_world"

    def test_slugify_removes_special_chars(self):
        result = slugify("Hello! World?")
        assert result == "hello_world"

    def test_slugify_max_length(self):
        result = slugify("This is a very long string", max_length=10)
        assert len(result) <= 10


class TestChunkIdGeneration:
    """Test chunk ID generation."""

    def test_chunk_id_format(self):
        chunk_id = generate_chunk_id(
            "html",
            "faculty",
            ["CSE", "Faculty"],
            "abc123def456"
        )
        assert chunk_id.startswith("html_")
        assert "faculty" in chunk_id
        assert "abc123de" in chunk_id  # First 8 chars of hash

    def test_chunk_id_deterministic(self):
        id1 = generate_chunk_id("html", "test", ["A", "B"], "hash123")
        id2 = generate_chunk_id("html", "test", ["A", "B"], "hash123")
        assert id1 == id2


class TestFrontmatterRemoval:
    """Test frontmatter removal."""

    def test_removes_frontmatter(self):
        text = "---\ntitle: Test\nauthor: Me\n---\n\n# Content"
        result = remove_frontmatter(text)
        assert "title" not in result
        assert "Content" in result

    def test_preserves_content_without_frontmatter(self):
        text = "# Just content\nNo frontmatter here"
        result = remove_frontmatter(text)
        assert result == text


class TestChunkModel:
    """Test Chunk dataclass."""

    def test_chunk_creation(self):
        chunk = Chunk(
            chunk_id="test_123",
            text="Hello world",
            source_type="html",
            source_file="test.md",
        )
        assert chunk.chunk_id == "test_123"
        assert chunk.word_count == 2  # Calculated in __post_init__

    def test_chunk_to_dict(self):
        chunk = Chunk(
            chunk_id="test_123",
            text="Hello world",
            source_type="html",
            source_file="test.md",
        )
        d = chunk.to_dict()
        assert d["chunk_id"] == "test_123"
        assert d["text"] == "Hello world"

    def test_chunk_from_dict(self):
        data = {
            "chunk_id": "test_123",
            "text": "Hello world",
            "source_type": "html",
            "source_file": "test.md",
            "section_hierarchy": ["A", "B"],
            "content_type": "section",
            "entity_refs": [],
            "page_range": None,
            "word_count": 2,
            "hash": "abc123"
        }
        chunk = Chunk.from_dict(data)
        assert chunk.chunk_id == "test_123"
        assert chunk.section_hierarchy == ["A", "B"]


class TestMarkdownChunker:
    """Test Markdown chunking logic."""

    def test_chunk_simple_markdown(self):
        markdown = """# Title

This is a paragraph with enough words to meet the minimum chunk size.
We need at least eighty words to create a valid chunk according to our
configuration settings. So let me add more content here to make sure
we have enough. This is paragraph one of the content. Adding more text
to ensure we reach the minimum word count requirement. Almost there now.
Just a few more words should do it. Yes this should be enough now.
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(markdown)
            temp_path = Path(f.name)
        
        try:
            chunker = MarkdownChunker()
            chunks = chunker.chunk_file(temp_path)
            assert len(chunks) >= 1
            assert chunks[0].source_type == "html"  # Not in pdfs folder
        finally:
            temp_path.unlink()

    def test_chunk_with_table(self):
        markdown = """# Data

| Name | Age |
|------|-----|
| John | 30  |
| Jane | 25  |

Some additional context about the table above. This needs more words
to meet the minimum chunk requirement. Adding more text here to ensure
we have enough content for a valid chunk.
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(markdown)
            temp_path = Path(f.name)
        
        try:
            chunker = MarkdownChunker()
            chunks = chunker.chunk_file(temp_path)
            # Should have at least one table chunk
            table_chunks = [c for c in chunks if c.content_type == "table"]
            assert len(table_chunks) >= 1
        finally:
            temp_path.unlink()


class TestChunkReport:
    """Test ChunkReport dataclass."""

    def test_report_creation(self):
        report = ChunkReport(
            total_chunks=100,
            duplicates_skipped=5,
            chunks_by_type={"section": 80, "table": 20}
        )
        assert report.total_chunks == 100
        assert report.chunks_by_type["section"] == 80

    def test_report_to_json(self):
        report = ChunkReport(total_chunks=10)
        json_str = report.to_json()
        assert "total_chunks" in json_str
        assert "10" in json_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
