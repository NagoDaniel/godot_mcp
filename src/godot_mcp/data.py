"""Resolve (and if needed download) the prebuilt Godot docs index.

The whole product is a single SQLite file (structured lookups + symbol index +
vector index). Resolution order:

1. ``$GODOT_MCP_DB`` — explicit path override.
2. A repo-local ``store/godot.sqlite`` — the dev build (when running from source).
3. A user cache dir (``platformdirs``). If absent there, download it once from the
   pinned release URL (``$GODOT_MCP_DB_URL`` or ``_DEFAULT_DB_URL``) and cache it.

This keeps the shipped wheel small: end users fetch the ~100 MB index on first run.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

from platformdirs import user_cache_dir

APP = "godot-mcp"
DB_FILENAME = "godot.sqlite"

# Pinned GitHub release asset for the prebuilt index. Override at runtime with
# $GODOT_MCP_DB_URL. Bump the tag + checksum below whenever the index is rebuilt.
_DEFAULT_DB_URL = os.environ.get(
    "GODOT_MCP_DB_URL",
    "https://github.com/NagoDaniel/godot_mcp/releases/download/v0.1.0/godot.sqlite",
)

# SHA-256 of the release asset, verified after download. Regenerate with
# `python scripts/publish_release.py` after re-indexing; override via env for testing.
_DB_SHA256 = os.environ.get(
    "GODOT_MCP_DB_SHA256",
    "f3d0bed54b94fc756e8289eb413c3f39cc0a808910610cbc9a8b0941a5812e77",
)

_REPO_DB = Path(__file__).resolve().parents[2] / "store" / DB_FILENAME


def _cache_db() -> Path:
    return Path(user_cache_dir(APP)) / DB_FILENAME


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"[godot-mcp] downloading index (~160 MB) from {url} ...")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 (trusted release URL)
    if _DB_SHA256:
        got = _sha256(tmp)
        if got != _DB_SHA256:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"index checksum mismatch: expected {_DB_SHA256}, got {got}. "
                "The download may be corrupt or the pinned checksum is stale."
            )
    tmp.replace(dest)
    print(f"[godot-mcp] index cached at {dest}")


def get_db_path() -> Path:
    """Return a path to the index db, downloading to cache if necessary."""
    env = os.environ.get("GODOT_MCP_DB")
    if env:
        p = Path(env).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"GODOT_MCP_DB points to a missing file: {p}")
        return p

    if _REPO_DB.exists():
        return _REPO_DB

    cache = _cache_db()
    if not cache.exists():
        _download(_DEFAULT_DB_URL, cache)
    return cache
