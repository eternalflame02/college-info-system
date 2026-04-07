# Phase-1 Knowledge Graph JSON Schema

## Scope

This schema applies only to phase 1:

- JSON storage only
- Deterministic high-confidence edges only

## Graph File Shape

```json
{
  "version": "1.0",
  "generated_at": "ISO-8601 timestamp",
  "nodes": [],
  "edges": []
}
```

## Node Shape

```json
{
  "id": "string",
  "type": "string",
  "name": "string",
  "aliases": ["string"],
  "source_refs": ["string"]
}
```

## Edge Shape

```json
{
  "id": "string",
  "type": "string",
  "source": "node_id",
  "target": "node_id",
  "confidence": 1.0,
  "deterministic": true,
  "evidence": ["string"]
}
```

## Phase-1 Edge Constraints

- `confidence` must be high confidence (recommended: `1.0` for deterministic rule-based edges).
- `deterministic` must be `true`.
- Edges inferred by probabilistic or model-dependent logic are out of scope for phase 1.

## Deterministic Edge Rules (Phase 1)

Examples of acceptable deterministic rules:

- `faculty -> teaches -> course` when explicit assignment is present in source content.
- `course -> part_of -> program` when course-program mapping is explicitly listed.
- `course -> has_prerequisite -> course` only when prerequisites are explicitly stated.

If evidence is ambiguous, omit the edge in phase 1.
