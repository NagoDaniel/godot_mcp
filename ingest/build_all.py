"""Run the full offline ingestion pipeline in order.

    uv run python ingest/build_all.py

Produces everything the MCP server needs in ``store/godot.sqlite`` (plus the
intermediate jsonl files). Requires the ingest extra:  uv sync --extra ingest
"""

from __future__ import annotations

import time

import parse_classes
import parse_guides
import chunk
import inventory
import load_db
import embed_index


def _step(name: str, fn) -> None:
    t = time.time()
    print(f"\n=== {name} ===")
    fn()
    print(f"    done in {time.time() - t:.1f}s")


def main() -> None:
    _step("parse class reference -> class_records.jsonl", parse_classes.main)
    _step("build symbol index (objects.inv) -> godot.sqlite", inventory.build)
    _step("load class records -> godot.sqlite:classes", load_db.build)
    _step("parse guides -> guide_docs.jsonl", parse_guides.main)
    _step("chunk docs -> chunks.jsonl", chunk.main)
    _step("embed chunks -> godot.sqlite:vec_chunks (BGE, fastembed)", embed_index.build)
    print("\nAll stores built. Run the MCP server:")
    print("    uv run godot-mcp")


if __name__ == "__main__":
    main()
