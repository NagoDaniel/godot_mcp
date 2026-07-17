#!/usr/bin/env python
"""Quick SQLite query script. Edit the queries below and run."""

import sqlite3
from pathlib import Path
import json

DB = Path(__file__).parent.parent / "store" / "godot.sqlite"
con = sqlite3.connect(DB)
con.row_factory = sqlite3.Row

# Tables:
# - vec_chunks: sqlite-vec virtual table (id, embedding)
# - chunk_meta: chunk metadata (id, chunk_key, text, source_type, title, class, symbol, kind, section_path, url, anchor)
# - fts_chunks: FTS5 virtual table (rowid, text from chunk_meta)
# - classes: class reference records (name, inherits, inherited_by, brief, description_md, ...)
# - symbols: symbol index (member_name, kind, owner_class, uri, anchor, ...)

# Edit queries here:
queries = [
    "SELECT * FROM chunk_meta WHERE id = 21876; ",
    "SELECT COUNT(*) as cnt FROM chunk_meta;",
    "SELECT COUNT(*) as cnt FROM classes;",
    "SELECT COUNT(*) as cnt FROM symbols;",
    "SELECT source_type, COUNT(*) FROM chunk_meta GROUP BY source_type;",
    "SELECT * FROM chunk_meta WHERE class='CharacterBody2D' LIMIT 2;",
]

# for q in queries:
#     print(f"\n>>> {q}")
#     try:
#         for row in con.execute(q).fetchall():
#             print(dict(row))
#     except Exception as e:
#         print(f"ERROR: {e}")
result = con.execute(queries[0]).fetchall()
print(f"\n>>> {queries[0]}")
result = [dict(row) for row in result]
for row in result:
    print(json.dumps(row, indent=2))



con.close()
