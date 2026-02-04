"""
Chunk data models for semantic chunking.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json


@dataclass
class Chunk:
    """
    Represents a semantic chunk of content.
    
    Attributes:
        chunk_id: Deterministic ID (format: {source_type}_{file_stem}_{section_slug}_{hash[:8]})
        text: Cleaned chunk text (no YAML frontmatter)
        source_type: "html" or "pdf"
        source_file: Relative path from project root
        section_hierarchy: List of heading titles [H1, H2, H3]
        content_type: "section" | "table" | "list" | "profile" | "regulation"
        entity_refs: List of entity IDs found in the chunk
        page_range: [start_page, end_page] for PDFs, None for HTML
        word_count: Number of words in the chunk
        hash: SHA-256 hash of normalized text
    """
    chunk_id: str
    text: str
    source_type: str
    source_file: str
    section_hierarchy: List[str] = field(default_factory=list)
    content_type: str = "section"
    entity_refs: List[str] = field(default_factory=list)
    page_range: Optional[List[int]] = None
    word_count: int = 0
    hash: str = ""

    def to_dict(self) -> dict:
        """Convert chunk to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        """Create chunk from dictionary."""
        return cls(**data)

    def __post_init__(self):
        """Calculate word count if not set."""
        if self.word_count == 0 and self.text:
            self.word_count = len(self.text.split())


@dataclass
class ChunkReport:
    """
    Summary report of chunking results.
    """
    total_chunks: int = 0
    duplicates_skipped: int = 0
    too_small_skipped: int = 0
    tables_split: int = 0
    chunks_by_type: dict = field(default_factory=dict)
    chunks_by_source: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert report to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def save_chunks(chunks: List[Chunk], output_path: str) -> None:
    """
    Save chunks to JSON file.
    
    Args:
        chunks: List of Chunk objects
        output_path: Path to output JSON file
    """
    data = [chunk.to_dict() for chunk in chunks]
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_chunks(input_path: str) -> List[Chunk]:
    """
    Load chunks from JSON file.
    
    Args:
        input_path: Path to input JSON file
        
    Returns:
        List of Chunk objects
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [Chunk.from_dict(item) for item in data]
