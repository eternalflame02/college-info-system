# Wide Quality Evaluation Summary

Generated at (UTC): 2026-04-14T20:05:06.870000+00:00

## Chunk Quality
- Total chunks: 2207
- Word count mean / median: 179.07 / 120.00
- Metadata completeness ratio: 1.000
- Exact duplicate ratio: 0.000
- Near-duplicate proxy ratio: 0.191

## Embedding Quality
- Embeddings: 2261 x 768
- Norm mean/std: 1.0000 / 0.0000
- Same-type neighbor ratio (k=5): 0.952
- Outlier count: 0

## Retrieval Quality
- Query count: 30
- Hit@5: 0.767
- MRR: 0.740
- NDCG@5: 0.841
- Precision@5 proxy: 0.600
- Recall@5 proxy: 0.733
- Latency mean/min/max (ms): 25898.19 / 13365.32 / 37817.61

## Knowledge Graph Quality
- Nodes / Edges: 480 / 1700
- Orphan node ratio: 0.017
- Connected components: 9
- Largest component ratio: 0.983
- Prerequisite edges: 0

## Improvement Backlog
- [P1] retrieval: Hit@k below target
  Change: Retune route-specific thresholds and add fallback broad retrieval when filtered results are empty.
  Success metric: Hit@k >= 0.85 and MRR >= 0.65 on 75-query benchmark
- [P1] retrieval/perf: High average retrieval latency
  Change: Persist loaded embedding model during multi-query evaluation and avoid per-query model initialization in runtime query path.
  Success metric: Mean query latency < 900ms on same hardware
- [P1] knowledge_graph: No prerequisite edges extracted
  Change: Constrain prerequisite extraction to nearest course heading context and resolve ambiguous source course with local window rules.
  Success metric: Non-zero prerequisite edges with manual precision >= 0.85 on sampled edges
