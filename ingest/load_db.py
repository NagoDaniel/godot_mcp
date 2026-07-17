"""Load parsed class records (JSONL) into ``store/godot.sqlite``.

Stores one row per class with the full record as a JSON blob, keyed by a
normalized (lower-cased) class name for case-insensitive lookups. This keeps all
structured data in a single SQLite file alongside the symbol index, so the MCP
server opens exactly one store.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DB_PATH = Path("store/godot.sqlite")
RECORDS = Path("store/class_records.jsonl")


def build(db_path: Path = DB_PATH, records: Path = RECORDS) -> int:
    con = sqlite3.connect(db_path)
    con.execute("DROP TABLE IF EXISTS classes")
    con.execute(
        """
        CREATE TABLE classes (
            name_lower TEXT PRIMARY KEY,
            name       TEXT NOT NULL,
            json       TEXT NOT NULL
        )
        """
    )
    n = 0
    with records.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            con.execute(
                "INSERT OR REPLACE INTO classes VALUES (?,?,?)",
                (rec["name"].lower(), rec["name"], line.strip()),
            )
            n += 1
    con.commit()
    con.close()
    return n


if __name__ == "__main__":
    n = build()
    print(f"loaded {n} class records -> {DB_PATH}:classes")
