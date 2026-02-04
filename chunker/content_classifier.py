"""
Content type classifier for semantic chunks.
Determines chunk type based on heuristics.
"""

import re
from typing import Optional

import config


def classify_content(
    text: str,
    source_file: str = "",
    section_hierarchy: Optional[list] = None
) -> str:
    """
    Classify content type based on heuristics.
    
    Priority order:
    1. Table (if contains Markdown table)
    2. Profile (if contains faculty info)
    3. Regulation (if source is regulation PDF and contains regulation identifiers)
    4. List (if >50% of lines are list items)
    5. Section (default)
    
    Args:
        text: Chunk text content
        source_file: Path to source file
        section_hierarchy: List of heading titles
        
    Returns:
        Content type: "table", "profile", "regulation", "list", or "section"
    """
    if not text:
        return "section"
    
    section_hierarchy = section_hierarchy or []
    
    # 1. Check for tables
    if _is_table_content(text):
        return "table"
    
    # 2. Check for faculty profile
    if _is_faculty_profile(text, section_hierarchy):
        return "profile"
    
    # 3. Check for regulations
    if _is_regulation_content(text, source_file, section_hierarchy):
        return "regulation"
    
    # 4. Check for lists
    if _is_list_content(text):
        return "list"
    
    # 5. Default to section
    return "section"


def _is_table_content(text: str) -> bool:
    """
    Check if text contains a Markdown table.
    
    Detects:
    - Table separator rows: |---|---|
    - Multiple pipe characters in lines
    """
    # Check for table separator pattern
    if re.search(r'\|[\s\-:]+\|', text):
        return True
    
    # Check for multiple lines with pipe characters
    lines = text.split('\n')
    pipe_lines = sum(1 for line in lines if '|' in line and line.count('|') >= 2)
    
    return pipe_lines >= 2


def _is_faculty_profile(text: str, section_hierarchy: list) -> bool:
    """
    Check if text is a faculty profile.
    
    Detection via:
    1. Section context (Faculty, The People, Our Team, Staff)
    2. Pattern matching (name + designation)
    """
    text_lower = text.lower()
    
    # Check section context
    section_text = ' '.join(section_hierarchy).lower()
    for keyword in config.FACULTY_SECTION_KEYWORDS:
        if keyword in section_text:
            # If in faculty section, check for profile markers
            if _has_faculty_markers(text_lower):
                return True
    
    # Check for faculty pattern even outside faculty section
    return _matches_faculty_pattern(text)


def _has_faculty_markers(text_lower: str) -> bool:
    """Check for common faculty profile markers."""
    markers = [
        'professor',
        'assistant professor',
        'associate professor',
        'head of department',
        'qualification',
        'email',
        'aicte id',
        'specialization',
    ]
    
    return sum(1 for marker in markers if marker in text_lower) >= 2


def _matches_faculty_pattern(text: str) -> bool:
    """
    Check if text matches faculty name + designation pattern.
    
    Pattern: (Dr./Prof./Mr./Ms.) Name + (Professor/Lecturer/etc.)
    """
    pattern = re.compile(
        r'(Dr\.?\s*|Prof\.?\s*|Mr\.?\s*|Ms\.?\s*|Mrs\.?\s*)'
        r'[A-Z][a-z]+(\s+[A-Z]\.?\s*)*(\s+[A-Z][a-z]+)+'
        r'.*'
        r'(Professor|Associate Professor|Assistant Professor|'
        r'Head of Department|Lecturer|HOD)',
        re.IGNORECASE | re.DOTALL
    )
    
    return bool(pattern.search(text))


def _is_regulation_content(
    text: str,
    source_file: str,
    section_hierarchy: list
) -> bool:
    """
    Check if text is regulation content.
    
    Detection via:
    1. Source file name contains "regulation"
    2. Contains regulation identifiers (R2019, R2023, etc.)
    3. Section hierarchy contains "regulation"
    """
    text_lower = text.lower()
    source_lower = source_file.lower()
    section_text = ' '.join(section_hierarchy).lower()
    
    # Check source file
    if 'regulation' in source_lower or 'curriculum' in source_lower:
        return True
    
    # Check section hierarchy
    if 'regulation' in section_text:
        return True
    
    # Check for regulation identifiers
    regulation_patterns = [
        r'R\d{4}',  # R2019, R2023
        r'Regulation\s*\d+',
        r'Section\s*\d+\.\d+',
        r'Clause\s*\d+',
    ]
    
    for pattern in regulation_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    return False


def _is_list_content(text: str) -> bool:
    """
    Check if text is primarily list content (>50% list items).
    """
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    if not lines:
        return False
    
    # Count list item lines
    list_patterns = [
        r'^[-*•]\s+',  # Unordered list
        r'^\d+\.\s+',  # Ordered list
        r'^[a-zA-Z]\)\s+',  # Letter list (a), b), etc.)
    ]
    
    list_lines = 0
    for line in lines:
        for pattern in list_patterns:
            if re.match(pattern, line):
                list_lines += 1
                break
    
    return list_lines / len(lines) > 0.5


if __name__ == "__main__":
    # Test content classification
    test_cases = [
        (
            "| Name | Designation |\n|---|---|\n| Dr. John | Professor |",
            "table"
        ),
        (
            "Dr. Jisha John\nProfessor & Head\nQualification: Ph.D\nEmail: jisha@mbcet.ac.in",
            "profile"
        ),
        (
            "R2023 Regulation 4.5\nThe grading system follows...",
            "regulation"
        ),
        (
            "- Item 1\n- Item 2\n- Item 3\n- Item 4",
            "list"
        ),
        (
            "The department was established in 2000. It offers various programs.",
            "section"
        ),
    ]
    
    for text, expected in test_cases:
        result = classify_content(text)
        status = "✓" if result == expected else "✗"
        print(f"{status} Expected: {expected}, Got: {result}")
        print(f"   Text: {text[:50]}...")
        print()
