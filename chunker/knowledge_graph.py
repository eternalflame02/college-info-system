import json
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _oxford_join(parts: List[str]) -> str:
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def generate_knowledge_graph_documents(data_dir: Path) -> List[Dict]:
    """
    Generate synthetic documents describing explicit knowledge graph relationships
    defined by the user, such as which faculty teaches which course.
    """
    faculty_path = data_dir / "entities" / "faculty.json"
    courses_path = data_dir / "entities" / "courses.json"
    assignments_path = data_dir / "entities" / "teaching_assignments.json"

    if not assignments_path.exists():
        logger.info(f"Registry not found at {assignments_path}. Skipping Knowledge Graph.")
        return []
        
    try:
        faculty_list = _load_json(faculty_path)
        faculty_map = {item.get("id"): item for item in faculty_list}
            
        courses_list = _load_json(courses_path)
        courses_map = {item.get("id"): item for item in courses_list}
            
        assignments = _load_json(assignments_path)
    except Exception as e:
        logger.warning(f"Error loading entity registries for Knowledge Graph: {e}")
        return []
        
    documents = []

    for fac_id, course_ids in assignments.items():
        if fac_id not in faculty_map:
            logger.warning(f"Faculty ID '{fac_id}' in teaching_assignments not found in faculty.json")
            continue

        faculty = faculty_map[fac_id]
        fac_name = faculty.get("name", "Unknown")
        fac_email = faculty.get("email")
        fac_designation = faculty.get("designation")

        valid_courses = []
        for course_id in course_ids:
            if course_id not in courses_map:
                logger.warning(f"Course ID '{course_id}' in teaching_assignments not found in courses.json")
                continue

            course_entity = courses_map[course_id]
            valid_courses.append(
                {
                    "course_id": course_id,
                    "course_name": course_entity.get("name", "Unknown"),
                    "course_code": course_entity.get("code", course_id),
                }
            )

        if not valid_courses:
            continue

        course_text_parts = [
            f"{course['course_code']} ({course['course_name']})"
            for course in valid_courses
        ]
        course_text = _oxford_join(course_text_parts)

        role_text = fac_designation if fac_designation else "faculty member"
        email_text = f" Contact email: {fac_email}." if fac_email else ""

        text = (
            f"[Context: Knowledge Graph] {fac_name} is a {role_text} in the CSE department."
            f" {fac_name} teaches {course_text}."
            f" The instructor for {course_text} is {fac_name}.{email_text}"
        )

        doc = {
            "id": f"kg_{fac_id}",
            "text": text,
            "metadata": {
                "source_file": "data/entities/teaching_assignments.json",
                "content_type": "knowledge_graph",
                "main_topic": "Teaching Assignment",
                "faculty_id": fac_id,
                "faculty_name": fac_name,
                "faculty_designation": fac_designation or "",
                "faculty_email": fac_email or "",
                "course_ids": [course["course_id"] for course in valid_courses],
                "course_codes": [course["course_code"] for course in valid_courses],
                "course_names": [course["course_name"] for course in valid_courses],
            },
        }
        documents.append(doc)

    logger.info(f"Generated {len(documents)} synthetic Knowledge Graph documents.")
    return documents
