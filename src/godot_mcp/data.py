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
import sys
import threading
import urllib.request
from pathlib import Path

from platformdirs import user_cache_dir

APP = "godot-mcp"
DB_FILENAME = "godot.sqlite"

# Both the retrieval and lookup layers resolve the db independently; serialize the
# resolve-and-maybe-download so two callers can't race on the same cache file.
_resolve_lock = threading.Lock()

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


def _cache_marker(dest: Path) -> Path:
    # Records which checksum was verified when `dest` was cached, so later runs can
    # detect a re-indexed release (new pinned checksum) without re-hashing the whole
    # ~160 MB file on every startup -- just compare two short strings.
    return dest.with_suffix(".sha256")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    # stdout is the stdio MCP transport's JSON-RPC channel; log progress to stderr
    # only, or plain text here corrupts the protocol stream.
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"[godot-mcp] downloading index (~160 MB) from {url} ...", file=sys.stderr)
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 (trusted release URL)
    if _DB_SHA256:
        got = _sha256(tmp)
        if got != _DB_SHA256:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"index checksum mismatch: expected {_DB_SHA256}, got {got}. "
                "The download may be corrupt or the pinned checksum is stale."
            )
        _cache_marker(dest).write_text(got, encoding="utf-8")
    tmp.replace(dest)
    print(f"[godot-mcp] index cached at {dest}", file=sys.stderr)


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

    # Serialize: the warmup thread and any concurrent tool call both land here.
    with _resolve_lock:
        cache = _cache_db()
        if not cache.exists():
            _download(_DEFAULT_DB_URL, cache)
            return cache

        # A previously cached index only stays valid if it matches the checksum this
        # version of the code is pinned to -- otherwise a re-indexed release (different
        # embedding model/dimensions) would silently keep serving the stale file and
        # produce a dimension mismatch at query time instead of a clear re-download.
        if _DB_SHA256:
            marker = _cache_marker(cache)
            cached_sha = (
                marker.read_text(encoding="utf-8").strip() if marker.exists() else None
            )
            if cached_sha != _DB_SHA256:
                print(
                    "[godot-mcp] cached index is stale (pinned checksum changed), "
                    "re-downloading...",
                    file=sys.stderr,
                )
                _download(_DEFAULT_DB_URL, cache)
        return cache
