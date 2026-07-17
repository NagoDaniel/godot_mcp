"""Component 3 — chunk parsed docs into retrieval units.

Two sources, two strategies (per the plan and ``RAG_info.md``):

- **Guides** (``guide_docs.jsonl``): split Markdown header-aware so a chunk never
  crosses a section boundary, then pack paragraph/code blocks up to a ~512-token
  budget with overlap. Fenced code blocks are kept intact (never split mid-code).
- **Class reference** (``class_records.jsonl``): the natural unit is one member
  description (method/property/signal/…), with the owning class as parent. Plus a
  class-overview chunk from the brief + description.

Every chunk carries metadata for later filtering: ``source_type``, ``class``,
``symbol``, ``kind``, ``url``, ``anchor``, ``section_path``, ``title``.

Token counts are approximated as ``len/4`` here; the embedding step (milestone 2)
re-tokenizes with the real BGE-M3 tokenizer. Output: ``store/chunks.jsonl``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

DOCS_BASE = "https://docs.godotengine.org/en/stable/"
CHARS_PER_TOKEN = 4
MAX_CHARS = 512 * CHARS_PER_TOKEN       # ~512 tokens
OVERLAP_CHARS = 80 * CHARS_PER_TOKEN    # ~80 tokens

RECORDS = Path("store/class_records.jsonl")
GUIDES = Path("store/guide_docs.jsonl")
OUT = Path("store/chunks.jsonl")

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _doc_url(uri: str, anchor: str = "") -> str:
    return f"{DOCS_BASE}{uri}#{anchor}" if anchor else f"{DOCS_BASE}{uri}"


def _split_blocks(md: str) -> list[str]:
    """Split Markdown into atomic blocks: fenced code stays whole, prose splits
    on blank lines."""
    blocks: list[str] = []
    lines = md.split("\n")
    buf: list[str] = []

    def flush_prose() -> None:
        text = "\n".join(buf).strip()
        if text:
            for para in re.split(r"\n\s*\n", text):
                if para.strip():
                    blocks.append(para.strip())
        buf.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            flush_prose()
            fence = [line]
            i += 1
            while i < len(lines):
                fence.append(lines[i])
                if lines[i].strip().startswith("```"):
                    i += 1
                    break
                i += 1
            blocks.append("\n".join(fence))
            continue
        buf.append(line)
        i += 1
    flush_prose()
    return blocks


def _pack(blocks: list[str]) -> list[str]:
    """Greedily pack blocks up to MAX_CHARS, carrying overlap between chunks."""
    chunks: list[str] = []
    cur: list[str] = []
    size = 0
    for blk in blocks:
        blen = len(blk)
        if blen > MAX_CHARS:  # oversized single block (long code): flush + emit whole
            if cur:
                chunks.append("\n\n".join(cur))
                cur, size = [], 0
            chunks.append(blk)
            continue
        if size + blen > MAX_CHARS and cur:
            chunks.append("\n\n".join(cur))
            # start next chunk with a trailing-overlap block for continuity
            overlap = cur[-1] if len(cur[-1]) <= OVERLAP_CHARS else ""
            cur = [overlap] if overlap else []
            size = len(overlap)
        cur.append(blk)
        size += blen + 2
    if cur:
        chunks.append("\n\n".join(cur).strip())
    return [c for c in chunks if c.strip()]


def _sections(md: str) -> list[tuple[list[str], str]]:
    """Split Markdown into (heading_path, body) segments at every heading."""
    segments: list[tuple[list[str], str]] = []
    path: list[str] = []
    body: list[str] = []
    in_fence = False

    def flush() -> None:
        text = "\n".join(body).strip()
        if text:
            segments.append((list(path), text))
        body.clear()

    for line in md.split("\n"):
        if line.strip().startswith("```"):
            in_fence = not in_fence
        m = _HEADER_RE.match(line) if not in_fence else None
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            path = path[: level - 1] + [title]
        else:
            body.append(line)
    flush()
    return segments


def chunk_guide(doc: dict) -> list[dict]:
    out: list[dict] = []
    for idx, (path, body) in enumerate(_sections(doc["markdown"])):
        for j, text in enumerate(_pack(_split_blocks(body))):
            section_path = path or [doc["title"]]
            out.append(
                {
                    "id": f"guide::{doc['url']}::{idx}::{j}",
                    "text": f"{' > '.join(section_path)}\n\n{text}",
                    "source_type": "tutorial",
                    "title": doc["title"],
                    "class": None,
                    "symbol": None,
                    "kind": "guide",
                    "section_path": section_path,
                    "url": _doc_url(doc["url"]),
                    "anchor": "",
                }
            )
    return out


_CLASS_MEMBER_FIELDS = [
    ("methods", "method"), ("properties", "property"), ("signals", "signal"),
    ("constants", "constant"), ("enums", "enum"), ("constructors", "constructor"),
    ("operators", "operator"), ("annotations", "annotation"),
    ("theme_items", "theme_item"),
]


def chunk_class(rec: dict) -> list[dict]:
    out: list[dict] = []
    cls = rec["name"]
    # class-overview chunk
    overview = f"# {cls}\n\nInherits: {' < '.join(rec['inherits'])}\n\n{rec['description_md']}".strip()
    for j, text in enumerate(_pack(_split_blocks(overview))):
        out.append(
            {
                "id": f"class::{cls}::overview::{j}",
                "text": text,
                "source_type": "class_ref",
                "title": cls,
                "class": cls,
                "symbol": cls,
                "kind": "class",
                "section_path": [cls, "Description"],
                "url": _doc_url(rec["url"]),
                "anchor": "",
            }
        )
    # one chunk per member
    for field, kind in _CLASS_MEMBER_FIELDS:
        for m in rec.get(field, []):
            desc = m.get("description_md", "")
            text = f"{cls}.{m['name']} — {m['signature']}\n\n{desc}".strip()
            for j, part in enumerate(_pack(_split_blocks(text))):
                out.append(
                    {
                        "id": f"class::{cls}::{kind}::{m['name']}::{j}",
                        "text": part,
                        "source_type": "class_ref",
                        "title": f"{cls}.{m['name']}",
                        "class": cls,
                        "symbol": f"{cls}.{m['name']}",
                        "kind": kind,
                        "section_path": [cls, kind, m["name"]],
                        "url": _doc_url(rec["url"], m.get("anchor", "")),
                        "anchor": m.get("anchor", ""),
                    }
                )
    return out


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with OUT.open("w", encoding="utf-8") as fh:
        def emit(ch: dict) -> int:
            if len(ch["text"].strip()) < 20:  # drop near-empty stubs
                return 0
            fh.write(json.dumps(ch, ensure_ascii=False) + "\n")
            return 1

        for line in RECORDS.open(encoding="utf-8"):
            for ch in chunk_class(json.loads(line)):
                n += emit(ch)
        for line in GUIDES.open(encoding="utf-8"):
            for ch in chunk_guide(json.loads(line)):
                n += emit(ch)
    print(f"wrote {n} chunks -> {OUT}")


if __name__ == "__main__":
    main()
