"""Retrieval eval harness — recall@k / MRR over a small golden set.

Runs the real MCP tool functions in-process (import, no stdio) against
``eval/golden.jsonl`` and reports how often the expected page lands in the top-k,
plus mean reciprocal rank. Doubles as a retrieval-regression gate: exits non-zero if
recall@6 falls below RECALL6_FLOOR.

    uv run python eval/run_eval.py

``scripts/smoke.py`` covers the MCP transport + a few assertions; this measures
retrieval *quality* so M3 (hybrid/reranker) can target real misses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# import the actual tool functions so we exercise the same code an agent hits
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from godot_mcp import mcp_server as M  # noqa: E402
from godot_mcp import retrieval as R  # noqa: E402

# Windows consoles default to cp1252; the report uses ✓ / ✗ / ≥.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

GOLDEN = Path(__file__).resolve().parent / "golden.jsonl"
K = 10                 # fetch depth
RECALL6_FLOOR = 0.85   # gate: fail the run if recall@6 drops below this

# tools whose results are ranked lists (vs a single structured dict)
_LIST_TOOLS = {"search_docs", "find_examples", "related_docs", "search_symbols"}
_QUERY_TOOLS = {"search_docs", "find_examples", "related_docs", "search_symbols"}


def _call(case: dict) -> list[dict]:
    """Invoke the tool and normalize to a ranked list of hits."""
    tool = case["tool"]
    kwargs = dict(case.get("args", {}))
    if tool in _QUERY_TOOLS:
        kwargs["query"] = case["query"]
        if tool in ("search_docs", "find_examples", "related_docs"):
            kwargs.setdefault("k", K)
        # related_docs uses 'topic', not 'query'
        if tool == "related_docs":
            kwargs["topic"] = kwargs.pop("query")
    result = getattr(M, tool)(**kwargs)
    return result if isinstance(result, list) else [result]


def _rank_of(hits: list[dict], expect_url: str) -> int | None:
    """1-based rank of the first hit whose url contains expect_url, else None."""
    for i, h in enumerate(hits, 1):
        if expect_url in (h.get("url") or ""):
            return i
    return None


def main() -> int:
    cases = [json.loads(line) for line in GOLDEN.open(encoding="utf-8") if line.strip()]
    ranks: list[int | None] = []
    rows: list[tuple] = []

    for c in cases:
        hits = _call(c)
        rank = _rank_of(hits, c["expect_url"])
        ranks.append(rank)

        code_ok = True
        if c.get("expect_code"):
            code_ok = any("```" in (h.get("text") or "") for h in hits[: rank or K])

        label = c.get("query") or c["args"]
        rows.append((c["tool"], str(label)[:42], rank, code_ok))

    # aggregate over all cases
    n = len(cases)
    def recall_at(k: int) -> float:
        return sum(1 for r in ranks if r is not None and r <= k) / n
    mrr = sum((1.0 / r) for r in ranks if r) / n
    r1, r3, r6 = recall_at(1), recall_at(3), recall_at(6)

    mode = f"rerank ON ({R.RERANK_MODEL})" if R.RERANK_DEFAULT else "dense-only (no rerank)"
    print(f"mode: {mode}\n")
    print(f"{'tool':<16}{'case':<44}{'rank':>6}  code")
    print("-" * 74)
    for tool, label, rank, code_ok in rows:
        mark = f"✓ {rank}" if rank else "✗ MISS"
        code = "" if not any(c.get("expect_code") for c in cases if c["tool"] == tool) \
            else (" ✓" if code_ok else " ✗")
        print(f"{tool:<16}{label:<44}{mark:>6}{code}")

    print("-" * 74)
    print(f"cases={n}  recall@1={r1:.2f}  recall@3={r3:.2f}  recall@6={r6:.2f}  MRR={mrr:.3f}")

    misses = [rows[i][1] for i, r in enumerate(ranks) if r is None]
    if misses:
        print("MISSES:", ", ".join(misses))

    if r6 < RECALL6_FLOOR:
        print(f"\nFAIL: recall@6 {r6:.2f} < floor {RECALL6_FLOOR}")
        return 1
    print(f"\nPASS: recall@6 {r6:.2f} ≥ floor {RECALL6_FLOOR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
