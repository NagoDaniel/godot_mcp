"""Component 2 — build the symbol index from Sphinx's ``objects.inv``.

``objects.inv`` maps every documented symbol (class, method, property, signal,
constant, enum, and tutorial page) to its ``file#anchor``. Parsing it gives us an
exact, exhaustive symbol table for free — no HTML scraping and no embeddings —
which powers ``search_symbols`` and precise citation links.

Output: a SQLite table ``symbols`` in ``store/godot.sqlite``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from sphobjinv import Inventory

DB_PATH = Path("store/godot.sqlite")
INV_PATH = Path("godot-docs-html-stable/objects.inv")


_MEMBER_KINDS = {
    "method", "property", "constant", "signal", "operator",
    "constructor", "annotation",
}


def _decode(name: str, role: str, dispname: str) -> tuple[str, str, str | None]:
    """Return ``(kind, member_name, owner_class)`` for a symbol.

    Godot's class reference is published as RST labels, not a typed Sphinx
    domain, following the convention ``class_<class>_<kind>_<member>`` where the
    class token is underscore-free (e.g. ``class_node_method_add_child``,
    ``class_acceptdialog_theme_constant_buttons_min_height``). Non-class labels
    (guide section anchors) and ``doc`` page entries fall through to a coarse
    kind derived from the Sphinx role.
    """
    if role == "doc":
        return "page", dispname, None
    if not name.startswith("class_"):
        return "section", dispname, None

    parts = name[len("class_"):].split("_")
    owner = parts[0]
    rest = parts[1:]
    if not rest:
        return "class", owner, None
    head = rest[0]
    if head == "theme":
        return "theme_item", "_".join(rest[2:]) or "_".join(rest[1:]), owner
    if head == "private":  # virtual/private method, leading underscore stripped
        return "method", "_" + "_".join(rest[2:]), owner
    if head in _MEMBER_KINDS:
        return head, "_".join(rest[1:]), owner
    return "member", "_".join(rest), owner


def build(db_path: Path = DB_PATH, inv_path: Path = INV_PATH) -> int:
    inv = Inventory(str(inv_path))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute("DROP TABLE IF EXISTS symbols")
    con.execute(
        """
        CREATE TABLE symbols (
            label         TEXT NOT NULL,   -- raw objects.inv name
            member_name   TEXT,            -- decoded member/class/section name
            kind          TEXT,            -- class|method|property|signal|...|page|section
            owner_class   TEXT,            -- squashed class token, or NULL
            uri           TEXT,
            anchor        TEXT,
            role          TEXT
        )
        """
    )
    rows = []
    for obj in inv.objects:
        uri = obj.uri_expanded  # '$' placeholder expanded to the object name
        file, _, anchor = uri.partition("#")
        display = obj.dispname if obj.dispname and obj.dispname != "-" else obj.name
        kind, member, owner = _decode(obj.name, obj.role, display)
        rows.append((obj.name, member, kind, owner, file, anchor, obj.role))
    con.executemany(
        "INSERT INTO symbols VALUES (?,?,?,?,?,?,?)", rows
    )
    con.execute("CREATE INDEX idx_symbols_member ON symbols(member_name)")
    con.execute("CREATE INDEX idx_symbols_owner ON symbols(owner_class)")
    con.execute("CREATE INDEX idx_symbols_kind ON symbols(kind)")
    con.commit()
    n = con.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
    con.close()
    return n


if __name__ == "__main__":
    n = build()
    print(f"indexed {n} symbols -> {DB_PATH}")
