"""Structured lookup logic over the Godot docs SQLite store.

Pure query functions with no MCP dependency, so they can be unit-tested and
reused. The MCP server (``mcp_server.py``) wraps these as tools.

Two backing tables:
- ``classes``  : one JSON blob per class (from the class-reference parser)
- ``symbols``  : typed symbol index (from objects.inv) for fast fuzzy search
"""

from __future__ import annotations

import json
import sqlite3
import threading

from . import data
from .textutil import resolve_links

DOCS_BASE = "https://docs.godotengine.org/en/stable/"

_MEMBER_FIELDS = {
    "method": "methods",
    "property": "properties",
    "signal": "signals",
    "constant": "constants",
    "enum": "enums",
    "constructor": "constructors",
    "operator": "operators",
    "annotation": "annotations",
    "theme_item": "theme_items",
}


# Lock-guarded singleton (not @lru_cache): tool handlers run in worker threads, so
# two structured calls can hit this concurrently on a cold start.
_con_lock = threading.Lock()
_con_inst: sqlite3.Connection | None = None


def _con() -> sqlite3.Connection:
    global _con_inst
    if _con_inst is None:
        with _con_lock:
            if _con_inst is None:
                con = sqlite3.connect(
                    f"file:{data.get_db_path()}?mode=ro",
                    uri=True,
                    check_same_thread=False,
                )
                con.row_factory = sqlite3.Row
                _con_inst = con
    return _con_inst


def _doc_url(uri: str, anchor: str = "") -> str:
    url = DOCS_BASE + uri
    return f"{url}#{anchor}" if anchor else url


class NotFound(Exception):
    """Raised when a requested class or member does not exist."""


# --- class-level -----------------------------------------------------------

def get_class(name: str) -> dict:
    row = _con().execute(
        "SELECT json FROM classes WHERE name_lower = ?", (name.lower(),)
    ).fetchone()
    if row is None:
        raise NotFound(f"class {name!r} not found")
    return json.loads(row["json"])


# Values considered "empty" and dropped from lean member payloads.
_EMPTY = (None, "", [], {})


def lookup_class(name: str) -> dict:
    """Summary of a class: inheritance, brief, and member name lists."""
    rec = get_class(name)
    base = _doc_url(rec["url"])
    return {
        "name": rec["name"],
        "inherits": rec["inherits"],
        "inherited_by": rec["inherited_by"],
        "brief": resolve_links(rec["brief"] or "", base),
        "description_md": resolve_links(rec["description_md"] or "", base),
        "url": base,
        "members": {
            field: [m["name"] for m in rec[field]]
            for field in _MEMBER_FIELDS.values()
            if rec.get(field)
        },
        "tutorials": rec["tutorial_links"],
    }


def _find_member(cls: str, kind: str, member: str) -> dict:
    rec = get_class(cls)
    base = _doc_url(rec["url"])
    field = _MEMBER_FIELDS[kind]
    target = member.lower().lstrip("_")
    for m in rec.get(field, []):
        if m["name"].lower().lstrip("_") == target:
            # resolve the member's relative links against the class page, then drop
            # empty columns so a method doesn't carry value:null, a constant no args, etc.
            out = {k: v for k, v in m.items() if v not in _EMPTY}
            if out.get("description_md"):
                out["description_md"] = resolve_links(out["description_md"], base)
            out["class"] = rec["name"]
            out["url"] = _doc_url(rec["url"], m["anchor"])
            return out
    raise NotFound(f"{kind} {member!r} not found on {rec['name']}")


def lookup_method(cls: str, method: str) -> dict:
    return _find_member(cls, "method", method)


def lookup_property(cls: str, prop: str) -> dict:
    return _find_member(cls, "property", prop)


def lookup_signal(cls: str, signal: str) -> dict:
    return _find_member(cls, "signal", signal)


def lookup_enum(cls: str, enum: str) -> dict:
    return _find_member(cls, "enum", enum)


def lookup_constant(cls: str, constant: str) -> dict:
    return _find_member(cls, "constant", constant)


# --- inheritance -----------------------------------------------------------

def show_inheritance(name: str) -> dict:
    """Ancestors (bottom-up chain) and known direct descendants of a class."""
    rec = get_class(name)
    return {
        "name": rec["name"],
        "inherits": rec["inherits"],
        "inherited_by": rec["inherited_by"],
        "url": _doc_url(rec["url"]),
    }


# --- symbol search ---------------------------------------------------------

def search_symbols(query: str, kind: str | None = None, limit: int = 25) -> list[dict]:
    """Fuzzy search over every documented symbol (classes, members, pages).

    Ranks exact name matches first, then prefix, then substring.
    """
    q = query.strip().lower()
    kind_clause = "AND kind = ?" if kind else ""
    sql = f"""
        SELECT member_name, kind, owner_class, uri, anchor,
               CASE
                   WHEN lower(member_name) = ? THEN 0
                   WHEN lower(member_name) LIKE ? THEN 1
                   ELSE 2
               END AS rank
        FROM symbols
        WHERE (lower(member_name) LIKE ? OR lower(member_name) LIKE ?) {kind_clause}
        ORDER BY rank, length(member_name), member_name
        LIMIT ?
    """
    params = [q, f"{q}%", f"{q}%", f"%{q}%"] + ([kind] if kind else []) + [limit]
    rows = _con().execute(sql, params).fetchall()
    return [
        {
            "name": r["member_name"],
            "kind": r["kind"],
            "class": r["owner_class"],
            "url": _doc_url(r["uri"], r["anchor"]),
        }
        for r in rows
    ]
