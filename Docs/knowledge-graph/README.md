# Knowledge Graph Documentation (Phase 1)

This folder defines the implementation scope for the phase-1 knowledge graph work.

## Decisions Fixed for Phase 1

1. **JSON only for now**  
   Graph data is stored in JSON files only. No graph database integration is part of this phase.

2. **High-confidence deterministic edges only**  
   Only edges derivable via deterministic rules with high confidence are included in phase 1.

3. **Graph construction and documentation first**  
   The immediate priority is building the graph artifacts and documenting the design/constraints.

4. **Deterministic reproducibility required**  
   Graph output must be reproducible with stable node/edge IDs and deterministic ordering.

## Implementation Entry Points

- CLI stage: `python main.py --stage graph`
- Module entry: `knowledge_graph.builder.run_knowledge_graph_pipeline()`

## Data Inputs and Outputs

### Inputs

- `data/entities/faculty.json`
- `data/entities/courses.json`
- `data/entities/programs.json`
- `data/chunks/chunks.json`

### Outputs

- `data/knowledge_graph/graph.json`
- `data/knowledge_graph/graph_report.json`

## Validation Checklist (Phase 1)

- Graph JSON has top-level keys: `version`, `generated_at`, `nodes`, `edges`
- Every edge has:
  - `confidence == 1.0`
  - `deterministic == true`
  - non-empty `evidence`
- Every edge endpoint references an existing node
- No duplicate node IDs
- No duplicate edge IDs

## Phase-1 Document Index

- `phase1-schema.md` — JSON shape for nodes/edges and deterministic edge rules.
