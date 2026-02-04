"""
Semantic Chunker for MBCET Markdown files.
Implements the core chunking logic as specified.
"""

import re
import hashlib
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from markdown_it import MarkdownIt

import config
from chunker.chunk_models import Chunk, ChunkReport, save_chunks
from chunker.entity_registry import EntityRegistry, find_entity_refs, load_entity_registry
from chunker.content_classifier import classify_content

logger = logging.getLogger(__name__)


# ============================================================
# Text Normalization and Hashing
# ============================================================

def normalize_for_hash(text: str) -> str:
    """
    Normalize text for hashing/deduplication.
    
    - Remove YAML frontmatter
    - Lowercase
    - Collapse whitespace
    - Strip leading/trailing whitespace
    """
    # Remove YAML frontmatter
    text = re.sub(r'^---\s*\n.*?\n---\s*\n', '', text, flags=re.DOTALL)
    
    # Lowercase
    text = text.lower()
    
    # Collapse whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def compute_hash(text: str) -> str:
    """Compute SHA-256 hash of normalized text."""
    normalized = normalize_for_hash(text)
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def slugify(text: str, max_length: int = 50) -> str:
    """
    Convert text to URL-safe slug.
    
    - Lowercase
    - Replace spaces with underscores
    - Remove special characters
    """
    slug = text.lower()
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[\s-]+', '_', slug)
    slug = slug.strip('_')
    return slug[:max_length]


def generate_chunk_id(
    source_type: str,
    file_stem: str,
    section_hierarchy: List[str],
    text_hash: str
) -> str:
    """
    Generate deterministic chunk ID.
    
    Format: {source_type}_{file_stem}_{section_slug}_{hash[:8]}
    """
    # Create section slug from hierarchy
    if section_hierarchy:
        section_slug = slugify('_'.join(section_hierarchy[-2:]))  # Last 2 levels
    else:
        section_slug = "root"
    
    file_slug = slugify(file_stem, max_length=30)
    
    return f"{source_type}_{file_slug}_{section_slug}_{text_hash[:8]}"


# ============================================================
# Markdown Parsing
# ============================================================

def remove_frontmatter(text: str) -> str:
    """Remove YAML frontmatter from markdown."""
    return re.sub(r'^---\s*\n.*?\n---\s*\n', '', text, flags=re.DOTALL)


def extract_page_markers(text: str) -> Dict[int, int]:
    """
    Extract page markers from markdown.
    
    Returns:
        Dictionary mapping line number to page number
    """
    markers = {}
    for i, line in enumerate(text.split('\n')):
        match = re.search(r'<!--\s*page:\s*(\d+)\s*-->', line)
        if match:
            markers[i] = int(match.group(1))
    return markers


def get_page_range(
    start_line: int,
    end_line: int,
    page_markers: Dict[int, int]
) -> Optional[List[int]]:
    """
    Get page range for a chunk based on line numbers.
    
    Returns:
        [start_page, end_page] or None if no markers
    """
    if not page_markers:
        return None
    
    pages = []
    sorted_markers = sorted(page_markers.items())
    
    current_page = None
    for line_num, page_num in sorted_markers:
        if line_num <= start_line:
            current_page = page_num
        elif line_num <= end_line:
            pages.append(page_num)
    
    if current_page:
        pages.insert(0, current_page)
    
    if not pages:
        return None
    
    return [min(pages), max(pages)]


# ============================================================
# Chunk Boundary Detection
# ============================================================

class MarkdownChunker:
    """
    Parse and chunk Markdown files according to semantic rules.
    """

    def __init__(self, entity_registry: Optional[EntityRegistry] = None):
        self.md = MarkdownIt()
        self.entity_registry = entity_registry
        
        # Heading stack for hierarchy tracking
        self.heading_stack: List[Tuple[int, str]] = []  # (level, text)
        
        # Size constraints
        self.min_words = config.MIN_CHUNK_WORDS
        self.max_words = config.MAX_CHUNK_WORDS
        self.soft_limit = config.SOFT_LIMIT_WORDS
        
    def _get_section_hierarchy(self) -> List[str]:
        """Get current section hierarchy as list of titles."""
        return [title for _, title in self.heading_stack]

    def _update_heading_stack(self, level: int, title: str):
        """Update heading stack when encountering a new heading."""
        # Remove all headings at same or lower level
        self.heading_stack = [
            (lvl, txt) for lvl, txt in self.heading_stack
            if lvl < level
        ]
        self.heading_stack.append((level, title))

    def _count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())

    def _split_large_table(self, table_text: str) -> List[str]:
        """
        Split large table into smaller chunks.
        
        - Preserve header row in each chunk
        - Max 50 rows per chunk
        """
        lines = table_text.strip().split('\n')
        
        if len(lines) <= 3:  # Header + separator + 1 row minimum
            return [table_text]
        
        # Find header and separator
        header_lines = []
        data_lines = []
        
        for i, line in enumerate(lines):
            if re.match(r'^\s*\|[\s\-:]+\|', line):
                # This is the separator line
                header_lines = lines[:i+1]
                data_lines = lines[i+1:]
                break
        
        if not header_lines:
            return [table_text]
        
        # Split data rows into chunks of MAX_TABLE_ROWS
        max_rows = config.MAX_TABLE_ROWS_PER_CHUNK
        chunks = []
        
        for i in range(0, len(data_lines), max_rows):
            chunk_rows = data_lines[i:i + max_rows]
            chunk_text = '\n'.join(header_lines + chunk_rows)
            chunks.append(chunk_text)
        
        return chunks if chunks else [table_text]

    def _extract_tokens(self, markdown: str) -> list:
        """Parse markdown and return tokens."""
        return self.md.parse(markdown)

    def _process_token_stream(
        self,
        tokens: list,
        source_file: str,
        source_type: str
    ) -> List[dict]:
        """
        Process token stream and identify chunk boundaries.
        
        Returns list of raw chunk data (not yet Chunk objects).
        """
        raw_chunks = []
        current_content = []
        current_start_line = 0
        
        for token in tokens:
            if token.type == 'heading_open':
                level = int(token.tag[1])
                
                # Check if we should start a new chunk
                should_split = False
                
                if level <= 2:  # H1 or H2 always start new chunk
                    should_split = True
                elif level == 3:
                    # H3 starts new chunk if current would exceed soft limit
                    current_text = '\n'.join(current_content)
                    if self._count_words(current_text) > self.soft_limit:
                        should_split = True
                
                if should_split and current_content:
                    # Save current chunk
                    raw_chunks.append({
                        'text': '\n'.join(current_content),
                        'hierarchy': self._get_section_hierarchy(),
                        'start_line': current_start_line,
                    })
                    current_content = []
                    current_start_line = token.map[0] if token.map else 0
                
            elif token.type == 'heading_close':
                pass
                
            elif token.type == 'inline' and self.heading_stack:
                # This is heading text, update stack
                parent_tokens = [t for t in tokens if t.type == 'heading_open']
                if parent_tokens:
                    last_heading = parent_tokens[-1]
                    if hasattr(last_heading, 'tag'):
                        level = int(last_heading.tag[1])
                        self._update_heading_stack(level, token.content)
                
            elif token.type == 'table_open':
                # Table always becomes its own chunk
                if current_content:
                    raw_chunks.append({
                        'text': '\n'.join(current_content),
                        'hierarchy': self._get_section_hierarchy(),
                        'start_line': current_start_line,
                    })
                    current_content = []
                    
            elif token.type in ('paragraph_open', 'bullet_list_open', 'ordered_list_open'):
                if token.map:
                    current_start_line = token.map[0]
                    
            elif token.type == 'inline':
                current_content.append(token.content)
        
        # Don't forget the last chunk
        if current_content:
            raw_chunks.append({
                'text': '\n'.join(current_content),
                'hierarchy': self._get_section_hierarchy(),
                'start_line': current_start_line,
            })
        
        return raw_chunks

    def chunk_file(
        self,
        filepath: Path,
        entity_registry: Optional[Dict] = None
    ) -> List[Chunk]:
        """
        Chunk a single Markdown file.
        
        Args:
            filepath: Path to Markdown file
            entity_registry: Optional entity lookup dictionary
            
        Returns:
            List of Chunk objects
        """
        self.heading_stack = []  # Reset for new file
        
        # Read file
        try:
            content = filepath.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Failed to read {filepath}: {e}")
            return []
        
        # Determine source type
        source_type = "pdf" if "pdfs" in str(filepath) else "html"
        
        # Remove frontmatter for chunking
        clean_content = remove_frontmatter(content)
        
        # Extract page markers
        page_markers = extract_page_markers(content)
        
        # Simple line-based chunking (more reliable than token-based for complex markdown)
        chunks = self._chunk_by_structure(
            clean_content,
            filepath,
            source_type,
            page_markers,
            entity_registry
        )
        
        return chunks

    def _chunk_by_structure(
        self,
        content: str,
        filepath: Path,
        source_type: str,
        page_markers: Dict[int, int],
        entity_registry: Optional[Dict]
    ) -> List[Chunk]:
        """
        Chunk content by structural elements (headings, tables, lists).
        """
        chunks = []
        lines = content.split('\n')
        
        current_chunk_lines = []
        current_start_line = 0
        heading_stack: List[Tuple[int, str]] = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check for heading
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                
                # Decide if we should start a new chunk
                should_split = False
                
                if level <= 2:
                    should_split = True
                elif level == 3:
                    current_text = '\n'.join(current_chunk_lines)
                    if self._count_words(current_text) > self.soft_limit:
                        should_split = True
                
                if should_split and current_chunk_lines:
                    # Create chunk from current content
                    chunk = self._create_chunk(
                        '\n'.join(current_chunk_lines),
                        filepath,
                        source_type,
                        [t for _, t in heading_stack],
                        current_start_line,
                        i - 1,
                        page_markers,
                        entity_registry
                    )
                    if chunk:
                        chunks.append(chunk)
                    current_chunk_lines = []
                    current_start_line = i
                
                # Update heading stack
                heading_stack = [(lv, t) for lv, t in heading_stack if lv < level]
                heading_stack.append((level, title))
                current_chunk_lines.append(line)
                i += 1
                continue
            
            # Check for table start
            if re.match(r'^\s*\|', line):
                # Collect entire table
                table_lines = []
                while i < len(lines) and re.match(r'^\s*\|', lines[i]):
                    table_lines.append(lines[i])
                    i += 1
                
                # If we have content before table, save it first
                if current_chunk_lines:
                    chunk = self._create_chunk(
                        '\n'.join(current_chunk_lines),
                        filepath,
                        source_type,
                        [t for _, t in heading_stack],
                        current_start_line,
                        i - len(table_lines) - 1,
                        page_markers,
                        entity_registry
                    )
                    if chunk:
                        chunks.append(chunk)
                    current_chunk_lines = []
                
                # Create table chunk(s)
                table_text = '\n'.join(table_lines)
                table_chunks = self._split_large_table(table_text)
                
                for tc_idx, tc_text in enumerate(table_chunks):
                    chunk = self._create_chunk(
                        tc_text,
                        filepath,
                        source_type,
                        [t for _, t in heading_stack],
                        i - len(table_lines),
                        i - 1,
                        page_markers,
                        entity_registry,
                        is_table=True,
                        part_suffix=f"_part{tc_idx + 1}" if len(table_chunks) > 1 else ""
                    )
                    if chunk:
                        chunks.append(chunk)
                
                current_start_line = i
                continue
            
            # Check if current chunk is getting too large
            current_text = '\n'.join(current_chunk_lines + [line])
            if self._count_words(current_text) > self.max_words:
                # Split at paragraph boundary
                if not line.strip():
                    chunk = self._create_chunk(
                        '\n'.join(current_chunk_lines),
                        filepath,
                        source_type,
                        [t for _, t in heading_stack],
                        current_start_line,
                        i - 1,
                        page_markers,
                        entity_registry
                    )
                    if chunk:
                        chunks.append(chunk)
                    current_chunk_lines = []
                    current_start_line = i + 1
                    i += 1
                    continue
            
            current_chunk_lines.append(line)
            i += 1
        
        # Create final chunk
        if current_chunk_lines:
            chunk = self._create_chunk(
                '\n'.join(current_chunk_lines),
                filepath,
                source_type,
                [t for _, t in heading_stack],
                current_start_line,
                len(lines) - 1,
                page_markers,
                entity_registry
            )
            if chunk:
                chunks.append(chunk)
        
        return chunks

    def _create_chunk(
        self,
        text: str,
        filepath: Path,
        source_type: str,
        section_hierarchy: List[str],
        start_line: int,
        end_line: int,
        page_markers: Dict[int, int],
        entity_registry: Optional[Dict],
        is_table: bool = False,
        part_suffix: str = ""
    ) -> Optional[Chunk]:
        """Create a Chunk object from raw data."""
        # Clean text
        text = text.strip()
        if not text:
            return None
        
        word_count = self._count_words(text)
        
        # Check minimum size (except for tables which can be smaller)
        if word_count < self.min_words and not is_table:
            logger.debug(f"Chunk too small ({word_count} words): {text[:50]}...")
            return None
        
        # Compute hash
        text_hash = compute_hash(text)
        
        # Generate chunk ID
        chunk_id = generate_chunk_id(
            source_type,
            filepath.stem,
            section_hierarchy,
            text_hash
        ) + part_suffix
        
        # Get page range
        page_range = get_page_range(start_line, end_line, page_markers)
        
        # Find entity references
        entity_refs = []
        if entity_registry:
            entity_refs = find_entity_refs(text, entity_registry)
        
        # Classify content type
        content_type = classify_content(text, str(filepath), section_hierarchy)
        if is_table:
            content_type = "table"
        
        # Get relative source file path
        try:
            source_file = str(filepath.relative_to(config.PROJECT_ROOT))
        except ValueError:
            source_file = str(filepath)
        
        return Chunk(
            chunk_id=chunk_id,
            text=text,
            source_type=source_type,
            source_file=source_file,
            section_hierarchy=section_hierarchy,
            content_type=content_type,
            entity_refs=entity_refs,
            page_range=page_range,
            word_count=word_count,
            hash=text_hash
        )


# ============================================================
# Main Pipeline
# ============================================================

class SemanticChunkingPipeline:
    """
    Main chunking pipeline that processes all Markdown files.
    """

    def __init__(self):
        self.entity_registry: Optional[Dict] = None
        self.seen_hashes: Dict[str, str] = {}  # hash -> chunk_id
        self.chunks: List[Chunk] = []
        self.report = ChunkReport()
        self.errors: List[str] = []

    def _log_error(self, message: str):
        """Log error to both logger and errors list."""
        logger.error(message)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.errors.append(f"[{timestamp}] {message}")

    def _log_chunk_event(self, event: str, chunk_id: str = "", **kwargs):
        """Log structured chunk event."""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "event": event,
            "chunk_id": chunk_id,
            **kwargs
        }
        
        # Write to structured log file
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(config.CHUNKER_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_data) + '\n')

    def load_entities(self):
        """Load entity registry."""
        logger.info("Loading entity registry...")
        self.entity_registry = load_entity_registry()
        logger.info(f"Loaded {len(self.entity_registry)} entity entries")

    def find_markdown_files(self) -> List[Path]:
        """Find all Markdown files to process."""
        files = []
        
        # HTML-derived files
        if config.MARKDOWN_PAGES_DIR.exists():
            files.extend(config.MARKDOWN_PAGES_DIR.glob("**/*.md"))
        
        # PDF-derived files
        if config.MARKDOWN_PDFS_DIR.exists():
            files.extend(config.MARKDOWN_PDFS_DIR.glob("**/*.md"))
        
        return sorted(files)

    def process_file(self, filepath: Path) -> List[Chunk]:
        """Process a single Markdown file."""
        logger.info(f"Processing: {filepath}")
        
        chunker = MarkdownChunker()
        try:
            chunks = chunker.chunk_file(filepath, self.entity_registry)
        except Exception as e:
            self._log_error(f"ERROR: Failed to process {filepath}: {e}")
            return []
        
        # Filter out duplicates
        unique_chunks = []
        for chunk in chunks:
            if chunk.hash in self.seen_hashes:
                original_id = self.seen_hashes[chunk.hash]
                self._log_error(
                    f"DUPLICATE: hash={chunk.hash[:16]}..., "
                    f"original={original_id}, source={chunk.source_file}"
                )
                self.report.duplicates_skipped += 1
            else:
                self.seen_hashes[chunk.hash] = chunk.chunk_id
                unique_chunks.append(chunk)
                self._log_chunk_event(
                    "chunk_created",
                    chunk.chunk_id,
                    source=chunk.source_file
                )
        
        return unique_chunks

    def run(self) -> Tuple[List[Chunk], ChunkReport]:
        """
        Run the complete chunking pipeline.
        
        Returns:
            Tuple of (chunks list, report)
        """
        logger.info("Starting semantic chunking pipeline...")
        
        # Clear structured log
        if config.CHUNKER_LOG_FILE.exists():
            config.CHUNKER_LOG_FILE.unlink()
        
        # Load entities
        self.load_entities()
        
        # Find files
        files = self.find_markdown_files()
        logger.info(f"Found {len(files)} Markdown files to process")
        
        if not files:
            logger.warning("No Markdown files found. Run scraping first.")
            return [], self.report
        
        # Process each file
        for filepath in files:
            chunks = self.process_file(filepath)
            self.chunks.extend(chunks)
            
            # Update report
            for chunk in chunks:
                # By type
                self.report.chunks_by_type[chunk.content_type] = \
                    self.report.chunks_by_type.get(chunk.content_type, 0) + 1
                
                # By source
                self.report.chunks_by_source[chunk.source_type] = \
                    self.report.chunks_by_source.get(chunk.source_type, 0) + 1
        
        self.report.total_chunks = len(self.chunks)
        
        logger.info(f"Chunking complete! Total chunks: {self.report.total_chunks}")
        
        return self.chunks, self.report

    def save_outputs(self):
        """Save all output files."""
        config.CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save chunks
        save_chunks(self.chunks, str(config.CHUNKS_FILE))
        logger.info(f"Saved chunks to {config.CHUNKS_FILE}")
        
        # Save report
        with open(config.CHUNK_REPORT_FILE, 'w', encoding='utf-8') as f:
            f.write(self.report.to_json())
        logger.info(f"Saved report to {config.CHUNK_REPORT_FILE}")
        
        # Save errors
        with open(config.ERRORS_LOG_FILE, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.errors))
        logger.info(f"Saved errors to {config.ERRORS_LOG_FILE}")


# ============================================================
# Public Functions
# ============================================================

def chunk_markdown_file(path: str, entity_registry: Dict) -> List[Chunk]:
    """
    Parse a single Markdown file and return list of chunks.
    
    Args:
        path: Absolute path to .md file
        entity_registry: Pre-loaded entity lookup dictionary
        
    Returns:
        List of Chunk objects
    """
    chunker = MarkdownChunker()
    return chunker.chunk_file(Path(path), entity_registry)


def run_chunking_pipeline():
    """
    Main entry point. Processes all Markdown files and produces outputs.
    
    Steps:
    1. Load entity registry
    2. Find all .md files in data/markdown/pages and data/markdown/pdfs
    3. Process each file
    4. Deduplicate globally
    5. Write outputs (chunks.json, chunk_report.json, errors.log)
    6. Print summary to console
    """
    pipeline = SemanticChunkingPipeline()
    chunks, report = pipeline.run()
    pipeline.save_outputs()
    
    # Print summary
    print("\n" + "=" * 50)
    print("✅ Chunking complete!")
    print("=" * 50)
    print(f"Total chunks: {report.total_chunks}")
    print(f"Duplicates skipped: {report.duplicates_skipped}")
    print(f"Too small skipped: {report.too_small_skipped}")
    print(f"\nChunks by type:")
    for content_type, count in report.chunks_by_type.items():
        print(f"  - {content_type}: {count}")
    print(f"\nChunks by source:")
    for source_type, count in report.chunks_by_source.items():
        print(f"  - {source_type}: {count}")
    print(f"\nOutput: {config.CHUNKS_FILE}")
    print("=" * 50)
    
    return chunks, report


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    run_chunking_pipeline()
