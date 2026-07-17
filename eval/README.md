# Retrieval eval

Small golden-set harness that measures retrieval **quality** (recall@k / MRR), as
opposed to `scripts/smoke.py` which checks the MCP transport + basic sanity.

```bash
uv run python eval/run_eval.py
```

Runs the real tool functions in-process against `golden.jsonl`, prints a per-case
rank table and aggregate metrics, and **exits non-zero if recall@6 < 0.85** — so it
doubles as a retrieval-regression gate.

## Results (bge-large-en-v1.5, 20 cases)

| Mode | recall@1 | recall@3 | recall@6 | MRR |
|------|---------|---------|---------|-----|
| dense-only (`GODOT_MCP_RERANK=0`) | 0.75 | 0.85 | 0.95 | 0.832 |
| + reranker (default, MiniLM-L-6)  | 0.85 | 0.95 | 1.00 | 0.902 |

The dense-only misses were semantic-phrase queries where the literal terms point
elsewhere — e.g. *"run cleanup code before the game quits"* (rank 7; "cleanup" pulls
editor methods) and *"connect a signal to a function"* (rank 4; buried under the
`Signal` class API). These are **reranker** cases, not lexical/BM25 cases — dense
already nails exact API-name queries (`NOTIFICATION_WM_CLOSE_REQUEST` → rank 1). The
cross-encoder reranker (`Xenova/ms-marco-MiniLM-L-6-v2`, 80 MB, torch-free ONNX) fixed
both and saturated recall@6. Toggle with `GODOT_MCP_RERANK=1` to compare modes.

## Adding a case

One JSON object per line in `golden.jsonl`:

```json
{"tool": "search_docs", "query": "…", "expect_url": "substring_of_expected_url"}
{"tool": "lookup_method", "args": {"class_name": "Node", "method": "add_child"}, "expect_url": "add-child"}
{"tool": "find_examples", "query": "…", "expect_url": "…", "expect_code": true}
```

- `expect_url` — a substring that must appear in a returned hit's `url` (page slug or
  anchor). Choose something stable and specific.
- `args` — kwargs for structured tools (or `kind` for `search_symbols`).
- `expect_code` — for `find_examples`, also require a code fence in the top hits.

## Metrics

- **recall@k** — fraction of cases whose `expect_url` appears in the top-k hits.
- **MRR** — mean of `1/rank` of the first matching hit (0 if missed).
- Structured tools return a single dict → effectively rank 1 or miss.
