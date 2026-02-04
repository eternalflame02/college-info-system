"""
Tests for entity registry module.
"""

import pytest
import json
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from chunker.entity_registry import (
    normalize_text,
    EntityRegistry,
    _edit_distance,
)


class TestNormalizeText:
    """Test text normalization for entity matching."""

    def test_lowercase(self):
        assert normalize_text("Dr. JOHN DOE") == "dr john doe"

    def test_normalize_dr_variations(self):
        assert normalize_text("Dr.John") == "dr john"
        assert normalize_text("Dr John") == "dr john"
        assert normalize_text("Dr. John") == "dr john"

    def test_normalize_prof_variations(self):
        assert normalize_text("Prof.Smith") == "prof smith"
        assert normalize_text("Prof Smith") == "prof smith"

    def test_collapse_whitespace(self):
        assert normalize_text("Dr.  John   Doe") == "dr john doe"

    def test_remove_special_chars(self):
        result = normalize_text("Dr. John's (PhD)")
        assert "'" not in result
        assert "(" not in result


class TestEditDistance:
    """Test edit distance calculation."""

    def test_identical_strings(self):
        assert _edit_distance("hello", "hello") == 0

    def test_one_char_difference(self):
        assert _edit_distance("hello", "hallo") == 1

    def test_one_char_missing(self):
        assert _edit_distance("hello", "helo") == 1

    def test_completely_different(self):
        assert _edit_distance("abc", "xyz") == 3


class TestEntityRegistry:
    """Test EntityRegistry class."""

    @pytest.fixture
    def sample_entities_file(self):
        """Create a temporary entity file."""
        entities = [
            {
                "id": "faculty_dr_john_doe",
                "name": "Dr. John Doe",
                "aliases": ["John Doe", "Dr John Doe"],
                "type": "faculty"
            },
            {
                "id": "faculty_prof_jane_smith",
                "name": "Prof. Jane Smith",
                "aliases": ["Jane Smith"],
                "type": "faculty"
            },
            {
                "id": "course_cs101",
                "name": "Data Structures",
                "aliases": ["CS101", "DS"],
                "type": "course"
            }
        ]
        
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False
        ) as f:
            json.dump(entities, f)
            return Path(f.name)

    def test_load_file(self, sample_entities_file):
        registry = EntityRegistry()
        count = registry.load_file(sample_entities_file)
        assert count == 3
        sample_entities_file.unlink()

    def test_exact_match(self, sample_entities_file):
        registry = EntityRegistry()
        registry.load_file(sample_entities_file)
        
        result = registry.find_exact_match("Dr. John Doe")
        assert result == "faculty_dr_john_doe"
        sample_entities_file.unlink()

    def test_alias_match(self, sample_entities_file):
        registry = EntityRegistry()
        registry.load_file(sample_entities_file)
        
        result = registry.find_exact_match("John Doe")
        assert result == "faculty_dr_john_doe"
        sample_entities_file.unlink()

    def test_fuzzy_match(self, sample_entities_file):
        registry = EntityRegistry()
        registry.load_file(sample_entities_file)
        
        # Fuzzy match with 1 char difference
        result = registry.find_fuzzy_match("Dr. John Dor")  # 'Dor' instead of 'Doe'
        assert result == "faculty_dr_john_doe"
        sample_entities_file.unlink()

    def test_find_entities_in_text(self, sample_entities_file):
        registry = EntityRegistry()
        registry.load_file(sample_entities_file)
        
        text = "Dr. John Doe teaches Data Structures (CS101)"
        refs = registry.find_entities_in_text(text)
        
        assert "faculty_dr_john_doe" in refs
        assert "course_cs101" in refs
        sample_entities_file.unlink()

    def test_no_duplicate_refs(self, sample_entities_file):
        registry = EntityRegistry()
        registry.load_file(sample_entities_file)
        
        text = "Dr. John Doe and John Doe are the same person"
        refs = registry.find_entities_in_text(text)
        
        # Should only have one reference to the faculty
        faculty_refs = [r for r in refs if "john_doe" in r]
        assert len(faculty_refs) == 1
        sample_entities_file.unlink()


class TestEntityRegistryEdgeCases:
    """Test edge cases for entity registry."""

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False
        ) as f:
            json.dump([], f)
            temp_path = Path(f.name)
        
        registry = EntityRegistry()
        count = registry.load_file(temp_path)
        assert count == 0
        temp_path.unlink()

    def test_missing_file(self):
        registry = EntityRegistry()
        count = registry.load_file(Path("/nonexistent/path.json"))
        assert count == 0

    def test_empty_text_search(self):
        registry = EntityRegistry()
        refs = registry.find_entities_in_text("")
        assert refs == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
