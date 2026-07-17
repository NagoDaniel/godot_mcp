#!/usr/bin/env python
"""Maintainer helper: prep a GitHub release for the prebuilt index.

The wheel ships without the ~160 MB `store/godot.sqlite`; end users download it once
from a GitHub release (see `src/godot_mcp/data.py`). Run this after (re)building the
index to get the checksum to pin and the exact commands to cut the release.

    python scripts/publish_release.py [--tag v0.1.0]

It prints the SHA-256 and, if the pinned value in data.py is stale, offers to update
it in place. It does not upload anything (no network, no gh dependency).
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DB = REPO / "store" / "godot.sqlite"
DATA_PY = REPO / "src" / "godot_mcp" / "data.py"
REPO_SLUG = "NagoDaniel/godot_mcp"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v0.1.0", help="release tag (default v0.1.0)")
    args = ap.parse_args()

    if not DB.exists():
        print(f"ERROR: {DB} not found — build it first (uv run python ingest/build_all.py).")
        return 1

    size_mb = DB.stat().st_size / (1 << 20)
    digest = _sha256(DB)
    print(f"index:   {DB}  ({size_mb:.0f} MB)")
    print(f"sha256:  {digest}\n")

    # keep the pinned checksum in data.py in sync
    text = DATA_PY.read_text(encoding="utf-8")
    m = re.search(r'"GODOT_MCP_DB_SHA256",\s*\n\s*"([0-9a-f]{64})"', text)
    pinned = m.group(1) if m else None
    if pinned == digest:
        print("data.py checksum: up to date [OK]")
    else:
        print(f"data.py checksum: STALE (pinned {pinned})")
        if m and input("update data.py in place? [y/N] ").strip().lower() == "y":
            DATA_PY.write_text(text.replace(pinned, digest), encoding="utf-8")
            print("  updated [OK] — commit this change with the release.")

    print("\nNext (needs the `gh` CLI, or use the GitHub web UI):")
    print(f"  git push origin master")
    print(f"  gh release create {args.tag} \"{DB}\" \\")
    print(f"     --repo {REPO_SLUG} --title {args.tag} --notes \"Prebuilt Godot docs index\"")
    print("\nWeb UI alternative: Releases -> Draft a new release -> "
          f"tag {args.tag} -> attach store/godot.sqlite -> Publish.")
    print(f"\nThe asset URL must match data.py:_DEFAULT_DB_URL "
          f"(.../releases/download/{args.tag}/godot.sqlite).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
