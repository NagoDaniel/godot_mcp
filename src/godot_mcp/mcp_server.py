"""Godot documentation MCP server.

Exposes structured lookup tools and RAG tools over stdio, backed by a single
SQLite store (class records + symbol index + sqlite-vec vectors). Resolved/
downloaded by ``data.get_db_path()``.

Run:  ``godot-mcp``  (console script)  or  ``python -m godot_mcp.mcp_server``
Register in an MCP client (Cursor / Claude Desktop / VS Code) with that command.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from . import lookups as L
from . import retrieval as R

mcp = FastMCP("godot-docs")


def _safe(fn, *args):
    try:
        return fn(*args)
    except L.NotFound as e:
        return {"error": str(e)}


# --- structured tools ------------------------------------------------------

@mcp.tool()
def lookup_class(name: str) -> dict:
    """Get a Godot class summary: inheritance chain, description, and the names
    of its methods, properties, signals, constants, enums, and operators."""
    return _safe(L.lookup_class, name)


@mcp.tool()
def lookup_method(class_name: str, method: str) -> dict:
    """Get a method's full signature, return type, arguments, and description."""
    return _safe(L.lookup_method, class_name, method)


@mcp.tool()
def lookup_property(class_name: str, property: str) -> dict:
    """Get a property's type, default value, and description."""
    return _safe(L.lookup_property, class_name, property)


@mcp.tool()
def lookup_signal(class_name: str, signal: str) -> dict:
    """Get a signal's arguments and description."""
    return _safe(L.lookup_signal, class_name, signal)


@mcp.tool()
def lookup_enum(class_name: str, enum: str) -> dict:
    """Get an enum's values and description."""
    return _safe(L.lookup_enum, class_name, enum)


@mcp.tool()
def lookup_constant(class_name: str, constant: str) -> dict:
    """Get a constant's value and description."""
    return _safe(L.lookup_constant, class_name, constant)


@mcp.tool()
def show_inheritance(class_name: str) -> dict:
    """Show a class's ancestor chain and its known direct descendants."""
    return _safe(L.show_inheritance, class_name)


@mcp.tool()
def search_symbols(query: str, kind: str | None = None, limit: int = 25) -> list[dict]:
    """Fuzzy-search every documented Godot symbol (classes, methods, properties,
    signals, constants, pages). Optionally filter by ``kind`` (e.g. 'method',
    'property', 'class', 'signal')."""
    return L.search_symbols(query, kind=kind, limit=limit)


# --- RAG tools -------------------------------------------------------------

@mcp.tool()
def search_docs(
    query: str,
    k: int = 6,
    source_type: str | None = None,
    lang: str | None = None,
) -> list[dict]:
    """Semantic search across all Godot documentation (class reference + guides).
    Returns the most relevant passages with a source URL each. Optionally restrict
    to 'tutorial' (prose guides) or 'class_ref' (API reference) via source_type.
    Pass lang='gdscript' or 'csharp' to keep only that language's code blocks."""
    return R.search(query, k=k, source_type=source_type, lang=lang)


@mcp.tool()
def find_examples(query: str, k: int = 6, lang: str | None = None) -> list[dict]:
    """Find Godot code examples relevant to a task. Returns passages that contain
    GDScript/C# code (tutorial snippets and class-member usage). Pass lang='gdscript'
    or 'csharp' to keep only that language's code blocks."""
    hits = R.search(query, k=k * 4, pool=80, lang=lang)
    coded = [h for h in hits if "```" in h["text"]]
    return (coded or hits)[:k]


@mcp.tool()
def read_page(url: str, max_chars: int = 8000) -> dict:
    """Read the full page behind a search hit's url when a passage is relevant but
    too short. Pass a url from a prior search_docs/find_examples/related_docs hit.
    Tutorial pages are reconstructed in full; class-reference urls return the class
    overview plus a pointer to the structured lookup_* tools."""
    return R.read_page(url, max_chars=max_chars)


@mcp.tool()
def related_docs(topic: str, k: int = 6) -> list[dict]:
    """Find documentation related to a topic or class. If the topic names a class,
    its overview is included alongside semantically related passages."""
    results: list[dict] = []
    seen: set[str] = set()

    match = L.search_symbols(topic, kind="class", limit=1)
    if match:
        # class-kind symbols carry the squashed class token in "name" (e.g.
        # "sprite2d"); chunk_meta.class matches case-insensitively.
        for h in R.chunks_for_class(match[0]["name"], limit=2):
            if h["url"] not in seen:
                seen.add(h["url"])
                results.append(h)

    for h in R.search(topic, k=k):
        if h["url"] not in seen:
            seen.add(h["url"])
            results.append(h)
    return results[:k]


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
