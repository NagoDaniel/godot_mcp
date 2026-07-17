"""Component 4 — embed chunks and build the vector index inside godot.sqlite.

Uses fastembed (ONNX, no torch) so the same lightweight runtime that indexes here
can also embed the user's query in the shipped ``uvx`` package. The dense model is
``BAAI/bge-large-en-v1.5`` (1024-d, ~1.3 GB, strong English retrieval) — a
fastembed-supported stand-in for BGE-M3, which fastembed does not offer and whose
multilingual strength is moot for the English-only Godot docs.

Everything lands in the existing ``store/godot.sqlite`` so the whole product ships
as one file:
  - ``vec_chunks``  : sqlite-vec virtual table, cosine KNN over dense vectors
  - ``chunk_meta``  : chunk text + metadata, keyed by the same integer id
  - ``fts_chunks``  : FTS5 over chunk text (BM25), built now to enable hybrid later

Idempotent: the three tables are dropped and rebuilt.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import sqlite_vec
from fastembed import TextEmbedding

# fastembed-supported dense English model. Change = full re-index (model-locked).
MODEL_NAME = "BAAI/bge-large-en-v1.5"
DIM = 1024
BATCH = 256

DB_PATH = Path("store/godot.sqlite")
CHUNKS = Path("store/chunks.jsonl")


def _connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    return con


def _create_tables(con: sqlite3.Connection) -> None:
    con.execute("DROP TABLE IF EXISTS vec_chunks")
    con.execute("DROP TABLE IF EXISTS chunk_meta")
    con.execute("DROP TABLE IF EXISTS fts_chunks")
    con.execute(
        f"CREATE VIRTUAL TABLE vec_chunks USING vec0("
        f"id INTEGER PRIMARY KEY, embedding float[{DIM}] distance_metric=cosine)"
    )
    con.execute(
        """
        CREATE TABLE chunk_meta (
            id           INTEGER PRIMARY KEY,
            chunk_key    TEXT,
            text         TEXT,
            source_type  TEXT,
            class        TEXT,
            symbol       TEXT,
            kind         TEXT,
            title        TEXT,
            section_path TEXT,
            url          TEXT,
            anchor       TEXT
        )
        """
    )
    con.execute(
        "CREATE VIRTUAL TABLE fts_chunks USING fts5(text, content='chunk_meta', content_rowid='id')"
    )


def _load_chunks() -> list[dict]:
    return [json.loads(line) for line in CHUNKS.open(encoding="utf-8")]


def build(db_path: Path = DB_PATH) -> int:
    chunks = _load_chunks()
    con = _connect(db_path)
    _create_tables(con)

    model = TextEmbedding(model_name=MODEL_NAME)
    texts = (c["text"] for c in chunks)

    con.execute("BEGIN")
    for i, (chunk, vec) in enumerate(zip(chunks, model.embed(texts, batch_size=BATCH))):
        con.execute(
            "INSERT INTO vec_chunks(id, embedding) VALUES (?, ?)",
            (i, sqlite_vec.serialize_float32(vec.tolist())),
        )
        con.execute(
            """INSERT INTO chunk_meta
               (id, chunk_key, text, source_type, class, symbol, kind, title,
                section_path, url, anchor)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                i,
                chunk["id"],
                chunk["text"],
                chunk["source_type"],
                chunk.get("class"),
                chunk.get("symbol"),
                chunk["kind"],
                chunk.get("title"),
                " > ".join(chunk.get("section_path") or []),
                chunk["url"],
                chunk.get("anchor", ""),
            ),
        )
        if (i + 1) % 2000 == 0:
            print(f"  embedded {i + 1}/{len(chunks)}")
    con.commit()

    # populate FTS from the metadata table
    con.execute("INSERT INTO fts_chunks(rowid, text) SELECT id, text FROM chunk_meta")
    con.commit()

    n = con.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0]
    con.close()
    return n


if __name__ == "__main__":
    n = build()
    print(f"indexed {n} chunk vectors ({MODEL_NAME}, {DIM}-d) -> {DB_PATH}")
