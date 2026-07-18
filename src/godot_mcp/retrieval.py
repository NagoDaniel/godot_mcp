"""Semantic retrieval over the sqlite-vec index in the Godot docs store.

Opens the store read-only, loads the sqlite-vec extension, lazily loads the
fastembed model once, and exposes ``search()`` — embed the query, run cosine KNN,
join chunk metadata, return chunks with ``url#anchor`` citations and scores.

Kept behind a thin function so BM25/hybrid (FTS5 is already in the file) and a
reranker can slot in later without changing the MCP tools. Emits a one-line trace
per query per ``RAG_info.md``.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import sys
from functools import lru_cache

import sqlite_vec
from fastembed import TextEmbedding

from . import data
from .textutil import resolve_links, strip_images

# Must match ingest/embed_index.py (embeddings are model-locked).
MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Cross-encoder reranker. Dense KNN gets the right page into the top-N but not always
# rank 1; the reranker reorders a candidate pool to fix precision@1. Measured on the
# eval set (eval/run_eval.py): recall@1 0.75->0.85, recall@6 0.95->1.00, MRR
# 0.83->0.90 — a clear win at 80 MB, torch-free ONNX via fastembed. On by default;
# set GODOT_MCP_RERANK=0 to disable (dense-only), or override the model via env.
RERANK_MODEL = os.environ.get("GODOT_MCP_RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
RERANK_POOL = int(os.environ.get("GODOT_MCP_RERANK_POOL", "40"))
RERANK_DEFAULT = os.environ.get("GODOT_MCP_RERANK", "1").lower() in ("1", "true", "yes")

log = logging.getLogger("godot_mcp.retrieval")


@lru_cache(maxsize=1)
def _con() -> sqlite3.Connection:
    con = sqlite3.connect(
        f"file:{data.get_db_path()}?mode=ro", uri=True, check_same_thread=False
    )
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    con.row_factory = sqlite3.Row
    return con


@lru_cache(maxsize=1)
def _model() -> TextEmbedding:
    return TextEmbedding(model_name=MODEL_NAME)


@lru_cache(maxsize=1)
def _reranker():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=RERANK_MODEL)


def warmup() -> None:
    """Force the index/embedder/reranker to load now, rather than on the first
    tool call. Downloading + loading these (index ~160 MB, embedder ~210 MB,
    reranker ~80 MB on first run) can take longer than an MCP client's per-call
    timeout; doing it during server startup avoids the first real query timing out.

    Logs to stderr only — stdout is the stdio MCP transport's JSON-RPC channel,
    and writing plain text there corrupts the protocol stream.
    """
    print("[godot-mcp] warming up index + models (first run downloads ~450 MB)...",
          file=sys.stderr, flush=True)
    _con()
    next(iter(_model().embed(["warmup"])))
    if RERANK_DEFAULT:
        list(_reranker().rerank("warmup", ["warmup"]))
    print("[godot-mcp] ready.", file=sys.stderr, flush=True)


def _embed_query(query: str) -> bytes:
    vec = next(iter(_model().query_embed([query])))
    return sqlite_vec.serialize_float32(vec.tolist())


def _row_to_hit(
    r: sqlite3.Row, distance: float | None, rerank_score: float | None = None
) -> dict:
    # Lean payload: only what the LLM acts on. The row still carries source_type/
    # kind/class for internal filtering — those just aren't emitted.
    # class-ref rows already store url with the anchor baked in; only guides carry a
    # bare url + separate anchor. Append the anchor only when it isn't already there.
    url = r["url"]
    if r["anchor"] and "#" not in url:
        url += f"#{r['anchor']}"
    # relative links -> absolute citations; drop unusable image markup.
    text = strip_images(resolve_links(r["text"], r["url"]))
    # reranker emits its own relevance score; else report cosine similarity.
    score = (
        round(rerank_score, 4)
        if rerank_score is not None
        else round(1.0 - (distance or 0.0), 4)
    )
    return {
        "text": text,
        "title": r["title"],
        "url": url,
        "score": score,
    }


# GDScript/C# tabs are collapsed into adjacent fenced blocks in one chunk; a
# single-language user only wants one. Post-filter at query time (no re-index).
_LANG_ALIASES = {
    "gdscript": "gdscript", "gd": "gdscript",
    "csharp": "csharp", "cs": "csharp", "c#": "csharp",
}
_FENCE_RE = re.compile(r"```([^\n`]*)\n.*?\n```\n?", re.DOTALL)


def _filter_lang(text: str, lang: str | None) -> str:
    """Strip fenced code blocks whose info-string names a language other than
    ``lang``. Unknown/None ``lang`` is a no-op. Fences with no/other info-string
    (e.g. shell, cpp, plain) are left untouched."""
    keep = _LANG_ALIASES.get((lang or "").strip().lower())
    if not keep:
        return text
    drop = {"gdscript", "csharp"} - {keep}

    def _sub(m: re.Match) -> str:
        info = m.group(1).strip().lower()
        return "" if info in drop else m.group(0)

    return _FENCE_RE.sub(_sub, text).strip()


def search(
    query: str,
    k: int = 6,
    source_type: str | None = None,
    kinds: list[str] | None = None,
    pool: int | None = None,
    lang: str | None = None,
    rerank: bool | None = None,
) -> list[dict]:
    """Semantic search. Optionally restrict by ``source_type`` or ``kinds``, and
    optionally keep only ``lang`` ('gdscript'|'csharp') code blocks in the text.

    When filtering, we over-fetch a candidate ``pool`` from the vector index and
    filter in Python (sqlite-vec KNN can't be combined with arbitrary WHEREs). When
    ``rerank`` is on (default ``GODOT_MCP_RERANK``), a cross-encoder rescences a larger
    pool and reorders it before taking the top ``k``.
    """
    use_rerank = RERANK_DEFAULT if rerank is None else rerank
    con = _con()
    qvec = _embed_query(query)
    # rerank needs a wide candidate pool to reorder; otherwise fetch just what filters need
    if use_rerank:
        fetch = pool or max(RERANK_POOL, k)
    else:
        fetch = pool or (k if not (source_type or kinds) else max(k * 6, 50))
    rows = con.execute(
        """
        SELECT m.*, v.distance
        FROM vec_chunks v
        JOIN chunk_meta m ON m.id = v.id
        WHERE v.embedding MATCH ? AND k = ?
        ORDER BY v.distance
        """,
        (qvec, fetch),
    ).fetchall()

    # apply metadata filters first (cheap), keeping distance order
    cand = [
        r for r in rows
        if (not source_type or r["source_type"] == source_type)
        and (not kinds or r["kind"] in kinds)
    ]

    if use_rerank and cand:
        scores = list(_reranker().rerank(query, [r["text"] for r in cand]))
        order = sorted(range(len(cand)), key=lambda i: scores[i], reverse=True)
        top = [(cand[i], scores[i]) for i in order[:k]]
        hits = [_row_to_hit(r, None, rerank_score=s) for r, s in top]
    else:
        hits = [_row_to_hit(r, r["distance"]) for r in cand[:k]]

    if lang:
        for h in hits:
            h["text"] = _filter_lang(h["text"], lang)

    log.info(
        "search q=%r k=%d rerank=%s filters=%s -> %s",
        query, k, use_rerank, {"source_type": source_type, "kinds": kinds},
        [(h["url"].rsplit("/", 1)[-1], h["score"]) for h in hits],
    )
    return hits


def chunks_for_class(cls: str, limit: int = 4) -> list[dict]:
    """Structured fetch of a class's own chunks (overview first). For related_docs."""
    con = _con()
    rows = con.execute(
        """
        SELECT *, 0.0 AS distance FROM chunk_meta
        WHERE lower(class) = lower(?)
        ORDER BY CASE kind WHEN 'class' THEN 0 ELSE 1 END, id
        LIMIT ?
        """,
        (cls, limit),
    ).fetchall()
    return [_row_to_hit(r, 0.0) for r in rows]


def read_page(url: str, max_chars: int = 8000) -> dict:
    """Read the full page behind a search hit's ``url``.

    Guide/tutorial pages are reconstructed from their chunks (all share one url),
    de-duplicating the overlap carried between consecutive chunks. Class-reference
    urls are *not* expanded member-by-member (a class page can be huge); instead the
    class overview is returned with a hint to use the structured ``lookup_*`` tools.
    """
    con = _con()
    base = url.split("#", 1)[0].replace("\\", "/")
    rows = con.execute(
        """
        SELECT * FROM chunk_meta
        WHERE url = ? OR url LIKE ? || '#%'
        ORDER BY id
        """,
        (base, base),
    ).fetchall()
    
    if not rows:
        return {"error": f"no indexed page for url: {base}"}

    if rows[0]["source_type"] != "tutorial":
        cls = rows[0]["class"]
        overview = chunks_for_class(cls, limit=1) if cls else []
        return {
            "url": base,
            "class": cls,
            "text": overview[0]["text"] if overview else "",
            "hint": (
                f"'{cls}' is a class reference page. Use lookup_class('{cls}') for "
                f"its full member list, or lookup_method/lookup_property for a "
                f"specific member — richer than the doc prose."
            ),
        }

    # Guide: join chunks in order, dropping the breadcrumb prefix and de-duping the
    # whole-block overlap carried between consecutive chunks.
    out_blocks: list[str] = []
    last_section: str | None = None
    for r in rows:
        section = r["section_path"]
        body = r["text"]
        if section and body.startswith(section):
            body = body[len(section):].lstrip("\n")
        if section != last_section:
            out_blocks.append(f"## {section}" if section else "")
            last_section = section
        for blk in body.split("\n\n"):
            blk = blk.strip()
            if blk and (not out_blocks or blk != out_blocks[-1]):
                out_blocks.append(blk)

    text = "\n\n".join(b for b in out_blocks if b)
    truncated = len(text) > max_chars
    return {
        "title": rows[0]["title"],
        "url": base,
        "text": text[:max_chars],
        "truncated": truncated,
    }
