# Implementation Strategy: Implicit Fact Linking

## Goal Description
The core problem preventing associative RAG queries (like "Who teaches Data Structures?") is that vector embeddings mathematically rely on textual overlap. If a faculty profile markdown file never explicitly lists the word "Data Structures", the query vector will naturally ignore them and retrieve the Syllabus page instead. 

To teach the Vector Model the relational links between entities, we need to ingest a **Knowledge Graph** (synthetic sentences explicitly stating facts) and introduce **Relational Metadata Mapping** to chunks.

## Proposed Changes

### 1. Generating a Synthetic "Knowledge Graph" (New Script)
Since the scraped website HTML does not contain explicit timetable mapping (i.e. Faculty Profiles do not list the subjects they teach), we need a system for the user to state these facts, which we then embed.
- **[NEW] `data/entities/teaching_assignments.json`**: A user-managed registry that explicitly maps `Faculty_Name` to an array of `Course_Codes`.
- **[NEW] `chunker/knowledge_graph.py`**: A new script that reads `faculties.json`, [courses.json](file:///e:/Rohith/COLLEGE/Mini%20Project/data/entities/courses.json), and `teaching_assignments.json`. For every assignment, it automatically generates a synthetic text document like:
  > *"Dr. Jisha John is a Professor in the CSE department. Her email is jisha@mbcet.ac.in. She teaches CS100 (Computer Networks) and CS200 (Artificial Intelligence)."*
- These synthetic "Knowledge Concept" documents will be embedded into ChromaDB with a special `content_type: knowledge_graph` tag. Because these synthetic sentences compress all the relational keywords into one tight vector, they naturally become the #1 closest L2 distance match for any associative query!

### 2. Expanding Metadata Tagging ([chunker/semantic_chunker.py](file:///e:/Rohith/COLLEGE/Mini%20Project/chunker/semantic_chunker.py))
- The semantic chunker will cross-reference `teaching_assignments.json`. 
- When chunking a **Faculty Profile**, the array of `Course_Codes` they teach will be injected into the ChromaDB [metadata](file:///e:/Rohith/COLLEGE/Mini%20Project/rag_ingestion.py#332-390).
- When chunking a **Course Syllabus**, the array of `Faculty_Names` who teach it will be injected into the ChromaDB [metadata](file:///e:/Rohith/COLLEGE/Mini%20Project/rag_ingestion.py#332-390).
- **Why?** If the user later adds an LLM on top of this Retrieval system, the RAG API will instantly pass the exact Course Codes and Teacher Names directly into the LLM context through the metadata!

## Verification Plan
### Automated Tests
- Run `query_rag` with "Who teaches Artificial Intelligence?". 
- Verify that a `content_type: knowledge_graph` chunk is retrieved at Rank 1.
