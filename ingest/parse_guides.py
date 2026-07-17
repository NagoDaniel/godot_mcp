"""Pipeline B — parse Godot prose guides/tutorials into clean Markdown docs.

Everything outside ``classes/`` (getting_started, tutorials, about, community,
engine_details, plus the root index) is prose. We isolate the main content,
convert it to Markdown (preserving headers, lists, code fences, tables), and
record the page title. The Markdown keeps its heading structure so the chunker
can split header-aware.

Output: ``store/guide_docs.jsonl`` — one JSON object per page
``{url, title, source_type: 'tutorial', markdown}``.
"""

from __future__ import annotations

import json
from pathlib import Path

from clean import load_main, to_markdown

DOCS_ROOT = Path("godot-docs-html-stable")
OUT = Path("store/guide_docs.jsonl")

# Directories/files that are not prose content.
_SKIP_DIRS = {"classes", "_static", "_sources", "_downloads", "_images"}
_SKIP_FILES = {"genindex.html", "search.html", "404.html"}


def _iter_pages() -> list[Path]:
    pages: list[Path] = []
    for p in DOCS_ROOT.rglob("*.html"):
        rel = p.relative_to(DOCS_ROOT)
        if rel.parts[0] in _SKIP_DIRS or p.name in _SKIP_FILES:
            continue
        pages.append(p)
    return sorted(pages)


def parse_guide_file(path: Path) -> dict | None:
    main = load_main(path)
    h1 = main.find("h1")
    title = h1.get_text(strip=True).replace("¶", "").strip() if h1 else path.stem
    markdown = to_markdown(main)
    if not markdown.strip():
        return None
    return {
        "url": path.relative_to(DOCS_ROOT).as_posix(),
        "title": title,
        "source_type": "tutorial",
        "markdown": markdown,
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pages = _iter_pages()
    n = 0
    with OUT.open("w", encoding="utf-8") as fh:
        for p in pages:
            try:
                doc = parse_guide_file(p)
            except Exception as e:  # noqa: BLE001
                print(f"  !! {p}: {e}")
                continue
            if doc is None:
                continue
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} guide docs -> {OUT}")


if __name__ == "__main__":
    main()
