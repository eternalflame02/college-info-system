#!/usr/bin/env python3
"""
Extract Entities from Markdown Files.

Parses syllabus markdown files to extract:
1. Courses: Course Code + Course Name + Credits + Category
2. Programs: Standard list of programs

Outputs to:
- data/entities/courses.json
- data/entities/programs.json
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Set

import config

logger = logging.getLogger(__name__)

# Basic Logging setup
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def extract_courses(markdown_dir: Path) -> List[Dict]:
    """
    Extract course entities from all markdown files in the directory.
    
    Looks for syllabus table rows matching:
    | Course Code | Course Name | Category | L | T | P | Credit |
    or specific heading patterns.
    """
    courses_dict: Dict[str, Dict] = {}
    
    # Regex to match course codes like "23CSL30A", "23CSL2MA", "MAT101", "CST202", etc.
    # Usually alphanumeric, 6-8 chars. 
    # Let's match typical format: 2+ digits, 2+ letters, 1+ digits, optional letters.
    # Or just generic: mostly alphanumeric.
    
    # We will primarily rely on the explicit syllabus definition structure.
    # Two common formats:
    # Format A: | 23CSL30A | COMPUTER NETWORKS | PCC | 3 | 1 | 0 | 4 |
    # Format B: | 23CSL30A |  | COMPUTER NETWORKS | PCC | ...
    # Format C (Heading): | Course Code | Course Name |
    #                     | 23CSL30A | COMPUTER NETWORKS...
    
    files = list(markdown_dir.glob("**/*.md"))
    logger.info(f"Scanning {len(files)} markdown files for courses...")
    
    for filepath in files:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        lines = content.split('\n')
        
        # 1. Look for syllabus header blocks:
        # | Course Code | Course Name |
        # | 23CSL30A | COMPUTER NETWORKS |
        
        for i, line in enumerate(lines):
            # Check if line looks like it contains a course code and name in a table row
            # A typical course code is 6-10 characters, upper case letters and numbers.
            # E.g. 23CSL30A, HUT200, EST120, MAT101
            
            # Simple heuristic: Split by pipe
            parts = [p.strip() for p in line.split('|') if p.strip()]
            
            if len(parts) >= 2:
                # Find a part that looks like a course code
                code_matcher = re.compile(r'^[A-Z0-9]{3,10}$')
                
                # Check the first 3 columns for a course code
                code = None
                code_idx = -1
                for idx, part in enumerate(parts[:3]):
                    # Must have at least one digit and one letter
                    if code_matcher.match(part) and any(c.isdigit() for c in part) and any(c.isalpha() for c in part):
                        code = part
                        code_idx = idx
                        break
                
                if code:
                    # Look for the name in the next 1-2 columns
                    # Name is usually all caps or title case, > 4 chars, no digits (usually)
                    name = None
                    for part in parts[code_idx+1:code_idx+3]:
                        if len(part) > 4 and not re.match(r'^(PCC|ESC|BSC|HSC|PEC|OEC|VAC|IEC|PWS|MNC|MSA)$', part, re.IGNORECASE):
                            name = part
                            break
                    
                    # Try to find credits
                    credits = ""
                    for part in parts[code_idx+2:]:
                        if re.match(r'^[0-9](\.[0-9])?$', part):
                            credits = part
                            # keep searching, maybe the last single digit is the credit
                    
                    if code and name and "Course Name" not in name:
                        # Normalize name
                        name = re.sub(r'\s+', ' ', name).strip()
                        
                        # Add to dictionary
                        if code not in courses_dict:
                            courses_dict[code] = {
                                "id": f"course_{code.lower()}",
                                "type": "course",
                                "name": name,
                                "code": code,
                                "aliases": [code, name, f"{code} {name}", f"{name} ({code})"],
                                "credits": credits
                            }
    
    result = list(courses_dict.values())
    logger.info(f"Found {len(result)} unique courses.")
    return result

def get_base_programs() -> List[Dict]:
    """Return static list of known programs."""
    return [
        {
            "id": "prog_btech_cse",
            "type": "program",
            "name": "B.Tech Computer Science and Engineering",
            "aliases": ["B.Tech CSE", "Computer Science and Engineering", "CSE", "BTech CSE", "B.Tech in Computer Science and Engineering"]
        },
        {
            "id": "prog_btech_ai",
            "type": "program",
            "name": "B.Tech Artificial Intelligence",
            "aliases": ["B.Tech AI", "Artificial Intelligence", "AI", "BTech AI", "B.Tech in Artificial Intelligence"]
        },
        {
            "id": "prog_mtech_cse",
            "type": "program",
            "name": "M.Tech Computer Science and Engineering",
            "aliases": ["M.Tech CSE", "MTech CSE", "M.Tech in Computer Science and Engineering"]
        }
    ]

def main():
    config.ensure_directories()
    
    # 1. Extract Courses
    pdf_md_dir = config.MARKDOWN_PDFS_DIR
    courses = extract_courses(pdf_md_dir)
    
    with open(config.COURSES_FILE, 'w', encoding='utf-8') as f:
        json.dump(courses, f, indent=2)
    logger.info(f"Written {len(courses)} courses to {config.COURSES_FILE}")
    
    # 2. Extract Programs
    programs = get_base_programs()
    with open(config.PROGRAMS_FILE, 'w', encoding='utf-8') as f:
        json.dump(programs, f, indent=2)
    logger.info(f"Written {len(programs)} programs to {config.PROGRAMS_FILE}")

if __name__ == "__main__":
    main()
