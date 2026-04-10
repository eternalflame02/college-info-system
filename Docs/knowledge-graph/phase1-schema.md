# Phase-1 Knowledge Graph JSON Schema

## Scope

This schema applies only to phase 1:

- JSON storage only
- Deterministic high-confidence edges only
- Deterministic serialization (stable IDs and sorted nodes/edges)

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

## Accepted Evidence Patterns (Current Implementation)

### `course -> part_of -> program`

Accepted only when:
- exactly one explicit program entity appears in the chunk evidence, or
- source file naming deterministically maps to one known program.

Rejected when:
- multiple program mappings are present in same chunk,
- no deterministic program mapping exists.

### `course -> has_prerequisite -> course`

Accepted only when:
- text contains explicit prerequisite marker (for example `Prerequisite:` or `Pre-requisite:`),
- referenced prerequisite course code can be mapped to a known course node.

Rejected when:
- source course is ambiguous in the chunk,
- prerequisite code cannot be mapped to a known course node.

### `faculty -> teaches -> course`

Accepted only when:
- faculty and course appear in the same row/span,
- row/span has assignment signal (assignment cue words or table-row structure).

Rejected when:
- faculty and course only co-occur without assignment signal.
