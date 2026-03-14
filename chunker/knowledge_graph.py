import json
import logging
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)

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
        with open(faculty_path, 'r', encoding='utf-8') as f:
            faculty_list = json.load(f)
            faculty_map = {f.get('id'): f for f in faculty_list}
            
        with open(courses_path, 'r', encoding='utf-8') as f:
            courses_list = json.load(f)
            courses_map = {c.get('id'): c for c in courses_list}
            
        with open(assignments_path, 'r', encoding='utf-8') as f:
            assignments = json.load(f)
    except Exception as e:
        logger.warning(f"Error loading entity registries for Knowledge Graph: {e}")
        return []
        
    documents = []
    
    for fac_id, course_ids in assignments.items():
        if fac_id not in faculty_map:
            logger.warning(f"Faculty ID '{fac_id}' in teaching_assignments not found in faculty.json")
            continue
            
        fac_name = faculty_map[fac_id].get('name', 'Unknown')
        
        for course_id in course_ids:
            if course_id not in courses_map:
                logger.warning(f"Course ID '{course_id}' in teaching_assignments not found in courses.json")
                continue
                
            course_name = courses_map[course_id].get('name', 'Unknown')
            course_code = courses_map[course_id].get('code', course_id)
            
            # Synthetic sentence mapping everything together securely and implicitly.
            # E.g. Dr. Jisha John teaches course CS0U20A (Artificial Intelligence).
            text = f"[Context: Knowledge Graph] {fac_name} teaches {course_name} ({course_code}). The instructor for {course_code} {course_name} is {fac_name}. The subject {course_name} is taught by {fac_name}."
            
            doc = {
                "id": f"kg_{fac_id}_{course_id}",
                "text": text,
                "metadata": {
                    "source_file": "teaching_assignments.json",
                    "content_type": "knowledge_graph",
                    "main_topic": "Teaching Assignment",
                    "faculty_id": fac_id,
                    "course_id": course_id,
                    "faculty_name": fac_name,
                    "course_name": course_name
                }
            }
            documents.append(doc)
            
    logger.info(f"Generated {len(documents)} synthetic Knowledge Graph documents.")
    return documents
