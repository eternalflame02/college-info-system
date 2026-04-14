# Improvement Backlog

## P1 - retrieval
- Issue: Hit@k below target
- Recommended change: Retune route-specific thresholds and add fallback broad retrieval when filtered results are empty.
- Success metric: Hit@k >= 0.85 and MRR >= 0.65 on 75-query benchmark

## P1 - retrieval/perf
- Issue: High average retrieval latency
- Recommended change: Persist loaded embedding model during multi-query evaluation and avoid per-query model initialization in runtime query path.
- Success metric: Mean query latency < 900ms on same hardware

## P1 - knowledge_graph
- Issue: No prerequisite edges extracted
- Recommended change: Constrain prerequisite extraction to nearest course heading context and resolve ambiguous source course with local window rules.
- Success metric: Non-zero prerequisite edges with manual precision >= 0.85 on sampled edges
