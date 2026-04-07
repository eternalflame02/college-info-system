# Knowledge Graph Documentation (Phase 1)

This folder defines the implementation scope for the phase-1 knowledge graph work.

## Decisions Fixed for Phase 1

1. **JSON only for now**  
   Graph data is stored in JSON files only. No graph database integration is part of this phase.

2. **High-confidence deterministic edges only**  
   Only edges derivable via deterministic rules with high confidence are included in phase 1.

3. **Graph construction and documentation first**  
   The immediate priority is building the graph artifacts and documenting the design/constraints.

4. **Technical-docs-only update scope (for this cycle)**  
   Updates are currently limited to:
   - `README.md`
   - `CONTRIBUTING.md`
   - `Docs/knowledge-graph/*`

## Phase-1 Document Index

- `phase1-schema.md` — JSON shape for nodes/edges and deterministic edge rules.
