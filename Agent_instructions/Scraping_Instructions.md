# 🔧 REFINED PRODUCTION-GRADE PROMPT FOR CODING AGENT

## Semantic Chunking Module for MBCET CSE Scraping Pipeline

You are a **senior Python data engineer** building a **semantic chunking subsystem** for an academic web-scraping + RAG + Knowledge Graph pipeline.

Your task is to **design and implement semantic chunking logic** that operates on **Markdown files** generated from both **HTML pages and PDFs**, producing **high-quality, meaning-preserving chunks** suitable for vector embedding, entity grounding, and knowledge graph linkage.

This is **offline preprocessing**, not runtime inference.

---

## 🎯 OVERALL GOAL

Convert Markdown documents into **semantic chunks** such that:

* Each chunk represents **one coherent unit of meaning**
* Chunks align with **document structure** (headings, tables, lists)
* Chunks can be linked to **entities (faculty, courses, programs)**
* Chunks are **stable, deterministic, and deduplicatable** across runs
* Chunks are optimized for **RAG retrieval quality**

**DO NOT** use fixed-size token chunking or sentence-level splitting.

---

## 🧱 INPUTS

### 1. Markdown Files

**Location:**
```
data/markdown/pages/**/*.md
data/markdown/pdfs/**/*.md
```

**Structure:**
* YAML frontmatter (optional)
* Structured headings (`#`, `##`, `###`)
* Paragraphs
* Lists (ordered and unordered)
* Tables (Markdown format)
* Optional page markers: `<!-- page: 12 -->`

### 2. Entity Registry Files

**Location:**
```
data/entities/faculty.json
data/entities/courses.json
data/entities/programs.json
```

**Expected Structure:**
```json
[
  {
    "id": "faculty_dr_jisha_john",
    "name": "Dr. Jisha John",
    "aliases": ["Jisha John", "Dr.Jisha John", "Dr Jisha John"],
    "type": "faculty"
  },
  {
    "id": "course_cs101",
    "name": "Data Structures",
    "aliases": ["CS101", "Data Structures and Algorithms"],
    "type": "course"
  }
]
```

---

## 📤 OUTPUTS

### 1. Chunk Registry

**File:** `data/chunks/chunks.json`

**Schema:**
```json
{
  "chunk_id": "string (deterministic, format: {source_type}_{file_stem}_{section_slug}_{hash[:8]})",
  "text": "string (cleaned chunk text, no YAML frontmatter)",
  "source_type": "html | pdf",
  "source_file": "string (relative path from project root)",
  "section_hierarchy": ["H1 title", "H2 title", "H3 title"],
  "content_type": "section | table | list | profile | regulation",
  "entity_refs": ["entity_id_1", "entity_id_2"],
  "page_range": [start_page, end_page] | null,
  "word_count": integer,
  "hash": "string (sha256 of normalized text)"
}
```

**Example:**
```json
{
  "chunk_id": "pdf_cse_btech_2023_sem3_course_table_a94f3c21",
  "text": "| Course Code | Course Name | Credits |\n|---|---|---|\n| CS301 | Database Systems | 4 |",
  "source_type": "pdf",
  "source_file": "data/markdown/pdfs/cse_btech_2023_syllabus.md",
  "section_hierarchy": ["Academics", "B.Tech CSE", "Semester 3"],
  "content_type": "table",
  "entity_refs": ["course_cs301"],
  "page_range": [5, 5],
  "word_count": 120,
  "hash": "a94f3c21e8d9..."
}
```

### 2. Summary Report

**File:** `data/chunks/chunk_report.json`

```json
{
  "total_chunks": 412,
  "duplicates_skipped": 27,
  "too_small_skipped": 9,
  "tables_split": 6,
  "chunks_by_type": {
    "section": 250,
    "table": 80,
    "profile": 45,
    "regulation": 30,
    "list": 7
  },
  "chunks_by_source": {
    "html": 180,
    "pdf": 232
  }
}
```

### 3. Error Log

**File:** `data/chunks/errors.log`

Plain text log of all errors, warnings, and skipped chunks with reasons.

### 4. Structured Activity Log

**File:** `logs/chunker.log`

JSON-structured logs for each chunk creation event:
```json
{"level": "INFO", "event": "chunk_created", "chunk_id": "html_faculty_001_abc123", "source": "data/markdown/pages/faculty.md"}
{"level": "WARN", "event": "chunk_skipped", "reason": "too_small", "source": "...", "word_count": 45}
```

---

## 🧠 SEMANTIC CHUNKING RULES (CRITICAL)

### 1️⃣ Chunk Boundary Detection (Priority Order)

**Process boundaries in this exact order:**

1. **Headings** (structural boundaries)
   - `#` (H1) → New top-level context (always start new chunk)
   - `##` (H2) → New semantic section (always start new chunk)
   - `###` (H3) → Sub-section (start new chunk if previous chunk would exceed 450 words)

2. **Tables** (always atomic)
   - Each Markdown table = **one chunk** (unless it exceeds 700 words)
   - Preserve header row
   - See "Large Table Handling" for split logic

3. **Entity Blocks** (semantic units)
   - Faculty profile blocks (name + designation + bio)
   - Course descriptions (code + name + details)
   - Each entity block = one chunk

4. **Lists** (contextual grouping)
   - Keep bullet/numbered lists grouped if they share the same heading context
   - Split only if list exceeds 600 words

5. **Paragraphs** (last resort splitting)
   - Only split paragraphs if chunk would exceed 700 words
   - Split at paragraph boundaries, never mid-paragraph

---

### 2️⃣ HTML-Derived Markdown Rules

**For files in `data/markdown/pages/`:**

* **Use headings to define semantic scope**
* **Combine paragraphs** under the same heading into a single chunk **UNLESS**:
  - Combined word count exceeds 600 words
  - Content switches entity type (faculty → course)

* **Faculty sections:**
  - Each individual faculty profile = one chunk
  - Detection logic:
    1. **Primary:** Section context (inside headings containing "Faculty", "The People", "Our Team", "Staff")
    2. **Fallback:** Regex pattern:
       ```regex
       (Dr\.?\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)+).*(Professor|Associate Professor|Assistant Professor|Head of Department|Lecturer)
       ```

**Example:**
```markdown
## Faculty
### Dr. Jisha John
Professor & Head
Department of Computer Science

Specialization: Machine Learning, Data Mining
```

**Output:** One chunk with `content_type: "profile"`, `entity_refs: ["faculty_dr_jisha_john"]`

---

### 3️⃣ PDF-Derived Markdown Rules (CRITICAL)

**For files in `data/markdown/pdfs/`:**

#### Syllabus PDFs

* **One course = one chunk** (course code + name + objectives + outcomes + syllabus)
* **Tables listing courses:**
  - Each table = one chunk (unless exceeds 700 words → see split logic)
  - Do NOT split individual table rows now (entity extraction happens later)
* **Semester headers define chunk scope**
  - Example: "Semester 3" heading starts a new chunk context

#### Regulations / Curriculum PDFs

* **One regulation section = one chunk**
* **Preserve regulation identifiers** (R2019, R2023, etc.) in `section_hierarchy`
* **Respect numbering** (e.g., "R2023.4.5 Grading System" is one chunk)

#### Timetables / Notifications

* **One timetable = one chunk**
* **One notice/announcement = one chunk**
* Detect via heading patterns: "Timetable", "Notification", "Announcement", "Notice"

---

## 📐 SIZE CONSTRAINTS

| Constraint         | Value         | Action if Violated                          |
|--------------------|---------------|---------------------------------------------|
| Minimum chunk size | 80 words      | Skip chunk, log to errors.log               |
| Preferred size     | 150–450 words | Target range (no action needed)             |
| Soft limit         | 600 words     | Consider splitting at next boundary         |
| Hard maximum       | 700 words     | **MUST split** at paragraph/semantic boundary|

**Splitting Rules When Exceeding 700 Words:**
1. Try splitting at next `###` heading
2. If no subheading, split at paragraph boundary
3. **Never split:**
   - Mid-paragraph
   - Inside tables (instead, split table by rows)
   - Inside code blocks

---

## 🔍 LARGE TABLE HANDLING

**Problem:** Some tables (e.g., full course catalog) may exceed 700 words.

**Solution: Hierarchical Splitting**

**Priority Order:**
1. **Split by semantic headers** within the table
   - Example: Tables with "Semester 1", "Semester 2" subsections
   - Each semester = one chunk
   
2. **Split by row count** if no semantic headers
   - Max **50 rows per chunk**
   - Preserve header row in every split chunk
   
3. **Append suffix to chunk IDs:**
   ```
   course_table_sem3_part1
   course_table_sem3_part2
   ```

**Example:**
```markdown
| Semester | Course Code | Course Name | Credits |
|----------|-------------|-------------|---------|
| 3        | CS301       | DBMS        | 4       |
| 3        | CS302       | OS          | 4       |
... (100 more rows)
```

**Output:** Multiple chunks, each with header row + subset of data rows.

---

## 🧬 METADATA EXTRACTION

### Section Hierarchy

**Maintain a heading stack as you parse:**

```python
# Example state while parsing
heading_stack = ["Academics", "B.Tech CSE", "Semester 3"]
```

**Store in chunk as:**
```json
"section_hierarchy": ["Academics", "B.Tech CSE", "Semester 3"]
```

**Rules:**
- When you encounter `#` → reset stack to `[H1_title]`
- When you encounter `##` → update stack to `[H1_title, H2_title]`
- When you encounter `###` → update stack to `[H1_title, H2_title, H3_title]`

---

### Entity References

**Use deterministic string matching + regex (NO ML):**

#### 1. Pre-Load Entity Registry at Startup

Load all entities from `data/entities/*.json` into memory:

```python
entity_registry = {
    "dr. jisha john": "faculty_dr_jisha_john",
    "jisha john": "faculty_dr_jisha_john",
    "dr.jisha john": "faculty_dr_jisha_john",
    "cs101": "course_cs101",
    "data structures": "course_cs101",
    # ... all entities with normalized keys
}
```

**Normalization:**
- Lowercase
- Remove extra whitespace
- Strip punctuation (except necessary ones like ".")

#### 2. Matching Priority

For each chunk text, find entity references:

1. **Exact normalized match** ✅ (highest confidence)
   - Example: "Dr. Jisha John" → normalize → "dr jisha john" → lookup

2. **Alias match** ✅
   - Example: "Jisha John" → normalize → "jisha john" → lookup

3. **Very light fuzzy match** ⚠️ (only for faculty names, edit distance ≤ 1)
   - Example: "Dr Jisha Jon" → edit distance 1 from "Dr Jisha John" → match
   - **Only apply to faculty**, not courses/programs

#### 3. Store Entity IDs

```json
"entity_refs": ["faculty_dr_jisha_john", "course_cs101"]
```

**Deduplication:** If same entity appears multiple times in chunk, store ID only once.

---

### Page Range Extraction

**Logic:**

1. **Check for page markers** in Markdown:
   ```markdown
   <!-- page: 12 -->
   Some content here
   <!-- page: 13 -->
   More content
   ```

2. **If markers exist:**
   - Compute `page_range: [start_page, end_page]`
   - Example: Chunk spans pages 12-13 → `[12, 13]`

3. **If markers do NOT exist:**
   - Set `page_range: null`

**Do NOT** couple chunker tightly to PDF internals. This is optional metadata.

---

## 🔁 DEDUPLICATION

**Implement hash-based global deduplication:**

### Normalization for Hashing

Before hashing, normalize chunk text:
1. **Remove YAML frontmatter** (lines between `---` markers)
2. **Lowercase** all text
3. **Remove extra whitespace** (collapse multiple spaces/newlines to single space)
4. **Trim** leading/trailing whitespace

### Hashing

```python
import hashlib

normalized_text = normalize(chunk_text)
chunk_hash = hashlib.sha256(normalized_text.encode('utf-8')).hexdigest()
```

### Deduplication Logic

**Maintain global registry:**
```python
seen_hashes = {}  # hash -> first_chunk_id
```

**When processing each chunk:**
1. Compute hash
2. If hash exists in `seen_hashes`:
   - **Skip** creating duplicate chunk
   - **Log** to `errors.log`:
     ```
     DUPLICATE: hash={hash}, original={first_chunk_id}, duplicate_source={current_file}
     ```
   - **Optionally** append `source_file` to metadata of first occurrence (not required)
3. If hash is new:
   - Create chunk
   - Add to `seen_hashes`

**Why global deduplication?**
- Vision/Mission statements repeat across pages
- Same regulation text in multiple PDFs
- You want **one canonical chunk** in vector DB

---

## 📊 CONTENT TYPE CLASSIFICATION

**Heuristic-based auto-classification (in priority order):**

1. **If chunk contains Markdown table** → `content_type: "table"`
   - Detect via `|---|---|` pattern or AST table nodes

2. **If entity type = faculty** (detected via context or regex) → `content_type: "profile"`

3. **If source is regulation PDF** AND section contains regulation identifiers → `content_type: "regulation"`
   - Patterns: `R2019`, `R2023`, `Regulation 4.5`

4. **If chunk is mostly list items** (>50% of lines start with `-`, `*`, or `1.`) → `content_type: "list"`

5. **Else** → `content_type: "section"`

**This is deterministic, transparent, and easy to explain.**

---

## 🔧 CHUNK ID GENERATION

**Format:**
```
{source_type}_{file_stem}_{section_slug}_{hash[:8]}
```

**Components:**
- `source_type`: `html` or `pdf`
- `file_stem`: Filename without extension (e.g., `faculty`, `cse_btech_2023_syllabus`)
- `section_slug`: Slugified section hierarchy (e.g., `sem3_course_table`)
- `hash[:8]`: First 8 characters of SHA-256 hash

**Example:**
```
pdf_cse_btech_2023_sem3_course_table_a94f3c21
```

**Slugification Rules:**
- Lowercase
- Replace spaces with `_`
- Remove special characters
- Max 50 characters

**Why this format?**
- ✅ Deterministic (same input → same ID)
- ✅ Human-readable
- ✅ Collision-safe (hash suffix)
- ✅ Stable across re-runs

---

## 🚨 ERROR HANDLING & LOGGING

### General Principles

- **NEVER crash the pipeline** on malformed Markdown
- **Always continue** processing remaining files
- **Log all errors** with context

### Error Categories

1. **Malformed Markdown**
   - Action: Skip problematic section, log error, continue with rest of file
   - Log: `ERROR: Malformed markdown in {file} at line {line_num}`

2. **Empty Chunks**
   - Action: Skip, log warning
   - Log: `WARN: Empty chunk skipped in {file}, section {section}`

3. **Too Small Chunks** (<80 words)
   - Action: Skip, log to `errors.log`
   - Log: `SKIP: Chunk too small ({word_count} words) in {file}`

4. **Duplicate Chunks**
   - Action: Skip, log to `errors.log`
   - Log: `DUPLICATE: hash={hash}, original={chunk_id}, source={file}`

5. **Entity Resolution Failures**
   - Action: Create chunk anyway, log warning, leave `entity_refs` empty
   - Log: `WARN: Entity not found: "{entity_mention}" in {file}`

### Logging Outputs

**1. Structured Activity Log** (`logs/chunker.log`)
```json
{"timestamp": "2025-02-05T10:30:45", "level": "INFO", "event": "chunk_created", "chunk_id": "html_faculty_001_abc123"}
{"timestamp": "2025-02-05T10:30:46", "level": "WARN", "event": "chunk_skipped", "reason": "too_small", "word_count": 45}
{"timestamp": "2025-02-05T10:30:47", "level": "ERROR", "event": "parse_error", "file": "data/markdown/pdfs/broken.md", "error": "..."}
```

**2. Error Log** (`data/chunks/errors.log`)
```
[2025-02-05 10:30:46] SKIP: Chunk too small (45 words) in data/markdown/pages/news.md, section: ["News", "Brief Update"]
[2025-02-05 10:30:50] DUPLICATE: hash=a94f3c21..., original=html_vision_001_xyz789, source=data/markdown/pages/about.md
```

**3. Summary Report** (`data/chunks/chunk_report.json`)
- See "OUTPUTS" section above

---

## 🗂️ ARCHITECTURE REQUIREMENTS

### Module Structure

**Create:** `semantic_chunker.py`

**Must expose these functions:**

```python
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_type: str
    source_file: str
    section_hierarchy: List[str]
    content_type: str
    entity_refs: List[str]
    page_range: List[int] | None
    word_count: int
    hash: str

def chunk_markdown_file(path: str, entity_registry: Dict) -> List[Chunk]:
    """
    Parse a single Markdown file and return list of chunks.
    
    Args:
        path: Absolute path to .md file
        entity_registry: Pre-loaded entity lookup dictionary
    
    Returns:
        List of Chunk objects
    """
    pass

def load_entity_registry() -> Dict[str, str]:
    """
    Load all entities from data/entities/*.json into memory.
    
    Returns:
        Dictionary mapping normalized entity names/aliases to entity IDs
    """
    pass

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
    pass
```

### Integration Point

**Must be callable from:**
```bash
python main.py --stage chunk
```

**Expected behavior:**
1. Load entity registry
2. Process all Markdown files
3. Write outputs
4. Print summary:
   ```
   ✅ Chunking complete!
   Total chunks: 412
   Duplicates skipped: 27
   Too small skipped: 9
   Output: data/chunks/chunks.json
   ```

---

## 📚 MARKDOWN PARSING LIBRARY

**Recommended: `markdown-it-py`**

**Why:**
- ✅ CommonMark compliant
- ✅ Provides AST (critical for tables & headings)
- ✅ More stable than regex-only parsing
- ✅ Handles edge cases better

**Installation:**
```bash
pip install markdown-it-py
```

**Usage Example:**
```python
from markdown_it import MarkdownIt

md = MarkdownIt()
tokens = md.parse(markdown_text)

# Iterate through AST
for token in tokens:
    if token.type == 'heading_open':
        level = int(token.tag[1])  # h1 → 1, h2 → 2
    elif token.type == 'table_open':
        # Handle table
```

**Fallback:** `mistune` (acceptable if you prefer)

**❌ DO NOT** use regex-only parsing for this task (too fragile)

---

## 🚫 EXPLICIT NON-GOALS (DO NOT IMPLEMENT)

- ❌ No embeddings generation
- ❌ No vector database integration
- ❌ No transformers / LLMs for chunking
- ❌ No sentence-level chunking
- ❌ No runtime inference or API endpoints
- ❌ No ML-based entity extraction
- ❌ No graph construction (that's a separate stage)

**Focus:** Deterministic, rule-based chunking only.

---

## ✅ SUCCESS CRITERIA

**The solution is correct if:**

1. **Semantic integrity:**
   - Chunks align with document meaning
   - Faculty, course, and program info are never mixed inappropriately
   - Tables remain intact (or split intelligently)

2. **Entity linkage:**
   - Chunks reference correct entity IDs
   - Entity references are deterministic

3. **KG-ready:**
   - Chunks are suitable for graph traversal queries
   - Section hierarchy enables context reconstruction

4. **Determinism:**
   - Output is **identical** across multiple runs
   - Chunk IDs are stable

5. **Quality:**
   - No chunks < 80 words (except tables/profiles that are naturally small)
   - No chunks > 700 words
   - Deduplication works correctly

6. **Robustness:**
   - Pipeline doesn't crash on malformed Markdown
   - All errors are logged
   - Summary report is accurate

---

## 🧠 IMPLEMENTATION MINDSET

**You are building ground truth context blocks for:**
- RAG answer generation
- Graph traversal queries
- Academic QA systems
- Faculty/course search

**Optimize for:**
- ✅ Clarity (chunks should be self-contained and understandable)
- ✅ Correctness (entity references must be accurate)
- ✅ Explainability (decisions should be traceable via logs)

**NOT for:**
- ❌ Speed (this is offline preprocessing)
- ❌ Minimum chunk count (quality > quantity)

---

## 📋 DEVELOPMENT CHECKLIST

Before considering the module complete, verify:

- [ ] Entity registry loads successfully from JSON files
- [ ] Markdown parsing handles all edge cases (tables, lists, code blocks)
- [ ] Chunk IDs are deterministic and collision-free
- [ ] Section hierarchy correctly tracks heading nesting
- [ ] Content type classification works for all categories
- [ ] Entity references match correctly (test with known entities)
- [ ] Page range extraction works when markers exist
- [ ] Deduplication catches identical content across files
- [ ] Large tables split intelligently
- [ ] Size constraints are enforced
- [ ] All three log outputs are generated
- [ ] Summary report matches actual chunk counts
- [ ] Pipeline continues on errors
- [ ] No crashes on malformed Markdown
- [ ] Integration with `main.py --stage chunk` works

---

## 🧪 TESTING RECOMMENDATIONS

**Create test cases for:**

1. **Faculty profile detection**
   - Standard format: "Dr. Name\nProfessor"
   - Edge cases: "Dr.Name" (no space), "Prof. Name"

2. **Table handling**
   - Small tables (< 700 words)
   - Large tables requiring splitting
   - Tables with semantic headers

3. **Deduplication**
   - Identical text with different whitespace
   - Same content in HTML and PDF sources

4. **Section hierarchy**
   - Nested headings (H1 → H2 → H3)
   - Missing heading levels (H1 → H3)

5. **Edge cases**
   - Empty files
   - Files with only frontmatter
   - Malformed tables
   - Very long paragraphs

---

## 📝 FINAL NOTES

- This is a **deterministic, rule-based system** — not AI-based chunking
- Prioritize **correctness** over cleverness
- When in doubt, **log it** — observability is critical
- Keep chunks **human-readable** — you may need to inspect them

**Good luck building! 🚀**