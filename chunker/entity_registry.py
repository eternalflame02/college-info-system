"""
Entity Registry for loading and matching entities.
Supports faculty, courses, and programs.
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import config

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """
    Normalize text for entity matching.
    
    - Lowercase
    - Remove extra whitespace
    - Strip punctuation (except period for Dr.)
    
    Args:
        text: Raw text to normalize
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Normalize title variations while retaining token boundaries
    text = re.sub(r'\bdr\.?\s*', 'dr ', text)
    text = re.sub(r'\bprof\.?\s*', 'prof ', text)
    text = re.sub(r'\bmr\.?\s*', 'mr ', text)
    text = re.sub(r'\bms\.?\s*', 'ms ', text)
    text = re.sub(r'\bmrs\.?\s*', 'mrs ', text)
    
    # Remove punctuation but preserve hyphenated terms as single tokens
    text = re.sub(r'[^\w\s-]', '', text)
    
    # Collapse whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def _edit_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein edit distance between two strings.
    Used for fuzzy faculty name matching.
    """
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


class EntityRegistry:
    """
    Manages entity loading and matching for faculty, courses, and programs.
    """

    def __init__(self):
        self.entities: Dict[str, dict] = {}  # id -> entity data
        self.lookup: Dict[str, str] = {}  # normalized name/alias -> id
        self.faculty_names: Set[str] = set()  # For fuzzy matching

    def _register_alias(self, alias: str, entity_id: str, entity_type: str):
        normalized = normalize_text(alias)
        if not normalized:
            return

        existing = self.lookup.get(normalized)
        if existing and existing != entity_id:
            # Keep first-seen deterministic mapping to avoid alias drift.
            logger.debug(
                "Alias collision for '%s': keeping %s, skipping %s",
                normalized,
                existing,
                entity_id,
            )
            return

        self.lookup[normalized] = entity_id
        if entity_type == 'faculty':
            self.faculty_names.add(normalized)

    def load_file(self, filepath: Path) -> int:
        """
        Load entities from a JSON file.
        
        Args:
            filepath: Path to entity JSON file
            
        Returns:
            Number of entities loaded
        """
        if not filepath.exists():
            logger.warning(f"Entity file not found: {filepath}")
            return 0
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            count = 0
            for entity in data:
                entity_id = entity.get('id')
                if not entity_id:
                    continue
                
                # Store full entity data
                self.entities[entity_id] = entity
                
                # Add primary name to lookup
                name = entity.get('name', '')
                if name:
                    self._register_alias(name, entity_id, entity.get('type', ''))
                
                # Add aliases to lookup
                for alias in entity.get('aliases', []):
                    self._register_alias(alias, entity_id, entity.get('type', ''))
                
                count += 1
            
            logger.info(f"Loaded {count} entities from {filepath}")
            return count
            
        except Exception as e:
            logger.error(f"Failed to load entities from {filepath}: {e}")
            return 0

    def load_all(self) -> int:
        """
        Load all entity files from the entities directory.
        
        Returns:
            Total number of entities loaded
        """
        total = 0
        
        # Load faculty
        total += self.load_file(config.FACULTY_FILE)
        
        # Load courses
        total += self.load_file(config.COURSES_FILE)
        
        # Load programs
        total += self.load_file(config.PROGRAMS_FILE)
        
        logger.info(f"Total entities loaded: {total}")
        return total

    def find_exact_match(self, text: str) -> Optional[str]:
        """
        Find exact entity match for normalized text.
        
        Args:
            text: Text to match
            
        Returns:
            Entity ID or None
        """
        normalized = normalize_text(text)
        return self.lookup.get(normalized)

    def find_fuzzy_match(
        self,
        text: str,
        max_distance: int = 1,
        entity_type: str = 'faculty',
    ) -> Optional[str]:
        """
        Find fuzzy match for faculty names (only).
        
        Args:
            text: Text to match
            max_distance: Maximum edit distance (default 1)
            
        Returns:
            Entity ID or None
        """
        normalized = normalize_text(text)
        if not normalized:
            return None

        candidates: Set[str] = set()
        if entity_type == 'faculty':
            candidates = self.faculty_names
        else:
            candidates = set(self.lookup.keys())

        best_id = None
        best_dist = max_distance + 1
        for candidate in candidates:
            distance = _edit_distance(normalized, candidate)
            if distance <= max_distance and distance < best_dist:
                best_id = self.lookup.get(candidate)
                best_dist = distance

        return best_id

    def find_entities_in_text(self, text: str) -> List[str]:
        """
        Find all entity references in a text chunk.
        
        Args:
            text: Text to search
            
        Returns:
            List of unique entity IDs found
        """
        found_ids: Set[str] = set()
        normalized_text = normalize_text(text)
        
        # Check for each known entity name/alias
        for name, entity_id in self.lookup.items():
            # Use boundaries to avoid substring-only false positives.
            if re.search(rf'\b{re.escape(name)}\b', normalized_text):
                found_ids.add(entity_id)
        
        # Fuzzy match for potentially misspelled faculty names
        # Extract potential names using regex
        potential_names = re.findall(
            r'(?:dr|prof|mr|ms|mrs)\s+[a-z]{2,}(?:\s+[a-z]{2,}){0,2}',
            normalized_text,
        )
        
        for potential in potential_names:
            # Only fuzzy match if no exact match found
            if potential not in self.lookup:
                fuzzy_id = self.find_fuzzy_match(potential)
                if fuzzy_id:
                    found_ids.add(fuzzy_id)
                    logger.debug(f"Fuzzy matched '{potential}' to {fuzzy_id}")

        # Explicitly check for Semester entities using Regex
        # Matches: "semester 1", "semester i", "s1", "s 1", "semester v", "s5", "semester 5"
        sem_pattern = re.compile(r'\b(semester|s)\s*([1-8ivx]+)\b', re.IGNORECASE)
        for match in sem_pattern.finditer(text):
            val = match.group(2).lower()
            
            # Map roman/number to 1-8
            mapping = {
                '1': '1', 'i': '1',
                '2': '2', 'ii': '2',
                '3': '3', 'iii': '3',
                '4': '4', 'iv': '4',
                '5': '5', 'v': '5',
                '6': '6', 'vi': '6',
                '7': '7', 'vii': '7',
                '8': '8', 'viii': '8'
            }
            if val in mapping:
                found_ids.add(f"semester_{mapping[val]}")
        
        return list(found_ids)

    def get_entity(self, entity_id: str) -> Optional[dict]:
        """Get entity data by ID."""
        return self.entities.get(entity_id)


# Global registry instance
_registry: Optional[EntityRegistry] = None


def load_entity_registry() -> Dict[str, str]:
    """
    Load all entities and return lookup dictionary.
    
    Returns:
        Dictionary mapping normalized names/aliases to entity IDs
    """
    global _registry
    _registry = EntityRegistry()
    _registry.load_all()
    return _registry.lookup


def find_entity_refs(text: str, registry: Optional[Dict] = None) -> List[str]:
    """
    Find entity references in text.
    
    Args:
        text: Text to search
        registry: Optional lookup dictionary (uses global if None)
        
    Returns:
        List of entity IDs found
    """
    global _registry
    
    if _registry is None:
        load_entity_registry()
    
    return _registry.find_entities_in_text(text)


def get_registry() -> Optional[EntityRegistry]:
    """Get the global entity registry instance."""
    global _registry
    return _registry


if __name__ == "__main__":
    # Test entity registry
    logging.basicConfig(level=logging.DEBUG)
    
    # Create sample entity files for testing
    config.ensure_directories()
    
    sample_faculty = [
        {
            "id": "faculty_dr_jisha_john",
            "name": "Dr. Jisha John",
            "aliases": ["Jisha John", "Dr.Jisha John", "Dr Jisha John"],
            "type": "faculty",
            "designation": "Professor"
        },
        {
            "id": "faculty_prof_raju_k_gopal",
            "name": "Prof. Raju K Gopal",
            "aliases": ["Raju K Gopal", "Prof Raju K Gopal"],
            "type": "faculty",
            "designation": "Professor & Head ITMS"
        }
    ]
    
    with open(config.FACULTY_FILE, 'w') as f:
        json.dump(sample_faculty, f, indent=2)
    
    # Test loading
    registry = EntityRegistry()
    registry.load_all()
    
    # Test matching
    test_texts = [
        "Dr. Jisha John is the professor.",
        "Prof. Raju K Gopal teaches DBMS.",
        "Dr Jisha Jon presented.",  # Fuzzy match test
    ]
    
    for text in test_texts:
        refs = registry.find_entities_in_text(text)
        print(f"Text: {text}")
        print(f"Found: {refs}")
        print()
