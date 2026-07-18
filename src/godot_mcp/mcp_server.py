"""Godot documentation MCP server.

Exposes structured lookup tools and RAG tools over stdio, backed by a single
SQLite store (class records + symbol index + sqlite-vec vectors). Resolved/
downloaded by ``data.get_db_path()``.

Run:  ``godot-mcp``  (console script)  or  ``python -m godot_mcp.mcp_server``
Register in an MCP client (Cursor / Claude Desktop / VS Code) with that command.
"""

from __future__ import annotations

from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from . import lookups as L
from . import retrieval as R

mcp = FastMCP("godot-docs")


def _safe(fn, *args):
    try:
        return fn(*args)
    except L.NotFound as e:
        return {"error": str(e)}


ClassName = Annotated[
    str,
    Field(description="Exact Godot class name, case-insensitive (e.g. "
                       "'CharacterBody2D' or 'characterbody2d'). If you don't know "
                       "the exact name, call search_symbols first."),
]

# --- structured tools --------------------------------------------------------
# Use these when you already know (or can guess) the exact class/member name and
# want an authoritative, zero-hallucination answer straight from the parsed class
# reference. Prefer search_symbols first if the exact name is unknown, and prefer
# search_docs/related_docs for conceptual "how do I..." questions instead of a
# specific member lookup.

@mcp.tool()
def lookup_class(name: ClassName) -> dict:
    """Get a full class summary: inheritance chain, description, and the names of
    its methods, properties, signals, constants, enums, and operators. This is the
    default starting point for "what is/what can X do" questions about a class --
    it already includes the inheritance chain, so you don't need show_inheritance
    unless you specifically want descendants without the rest of the summary."""
    return _safe(L.lookup_class, name)


@mcp.tool()
def lookup_method(
    class_name: ClassName,
    method: Annotated[
        str,
        Field(description="Method name, case-insensitive, leading underscore "
                           "optional (e.g. 'move_and_slide' or '_physics_process')."),
    ],
) -> dict:
    """Get one method's full signature, return type, arguments, and description.
    Use this instead of lookup_class when you need the precise signature of a
    single method rather than the whole class summary."""
    return _safe(L.lookup_method, class_name, method)


@mcp.tool()
def lookup_property(
    class_name: ClassName,
    property: Annotated[
        str,
        Field(description="Property name, case-insensitive, leading underscore "
                           "optional (e.g. 'velocity')."),
    ],
) -> dict:
    """Get one property's type, default value, and description."""
    return _safe(L.lookup_property, class_name, property)


@mcp.tool()
def lookup_signal(
    class_name: ClassName,
    signal: Annotated[
        str,
        Field(description="Signal name, case-insensitive (e.g. 'body_entered')."),
    ],
) -> dict:
    """Get one signal's arguments and description."""
    return _safe(L.lookup_signal, class_name, signal)


@mcp.tool()
def lookup_enum(
    class_name: ClassName,
    enum: Annotated[
        str,
        Field(description="Enum type name, case-insensitive (e.g. 'MotionMode')."),
    ],
) -> dict:
    """Get one enum's values and description."""
    return _safe(L.lookup_enum, class_name, enum)


@mcp.tool()
def lookup_constant(
    class_name: ClassName,
    constant: Annotated[
        str,
        Field(description="Constant name, case-insensitive "
                           "(e.g. 'NOTIFICATION_WM_CLOSE_REQUEST')."),
    ],
) -> dict:
    """Get one constant's value and description."""
    return _safe(L.lookup_constant, class_name, constant)


@mcp.tool()
def show_inheritance(class_name: ClassName) -> dict:
    """Get just a class's ancestor chain and known direct descendants -- nothing
    else. lookup_class already returns the same inherits/inherited_by fields
    plus a full summary, so prefer lookup_class unless you're walking a class
    hierarchy tree and specifically don't want the rest of the payload."""
    return _safe(L.show_inheritance, class_name)


@mcp.tool()
def search_symbols(
    query: Annotated[
        str,
        Field(description="Full or partial symbol name to match, e.g. "
                           "'body_entered' or 'move_and'."),
    ],
    kind: Annotated[
        str | None,
        Field(description="Restrict to one symbol kind: 'class', 'method', "
                           "'property', 'signal', 'constant', 'enum', 'operator', "
                           "'constructor', 'theme_item', 'annotation', 'page', or "
                           "'section'. Omit to search across all kinds."),
    ] = None,
    limit: Annotated[
        int, Field(description="Maximum number of matches to return.")
    ] = 25,
) -> list[dict]:
    """Fuzzy-search every documented Godot symbol and page title by name (exact,
    then prefix, then substring match). Use this first when you don't know the
    exact class or member name to pass to a lookup_* tool -- e.g. to find which
    class defines a signal, or to resolve a name you're not sure how to spell."""
    return L.search_symbols(query, kind=kind, limit=limit)


# --- RAG tools ----------------------------------------------------------------
# Use these for conceptual questions, explanations, or code samples, where no
# single class/member lookup would answer the question. All three return the same
# lean hit shape (text, title, url, score) with an absolute docs.godotengine.org
# citation. Prefer search_docs by default; reach for find_examples only when you
# specifically want a working code sample, and related_docs when you want a broad
# overview of a class/topic rather than an answer to one specific question.

@mcp.tool()
def search_docs(
    query: Annotated[
        str,
        Field(description="A natural-language question or description of what "
                           "you want to know, e.g. 'how do 2D lights and shadows "
                           "work' or 'detect a body entering an Area2D'."),
    ],
    k: Annotated[
        int, Field(description="Number of passages to return.")
    ] = 6,
    source_type: Annotated[
        str | None,
        Field(description="Restrict to 'tutorial' (prose guides) or 'class_ref' "
                           "(API reference text). Omit to search both."),
    ] = None,
    lang: Annotated[
        str | None,
        Field(description="'gdscript' or 'csharp' to keep only that language's "
                           "code blocks in the returned text (Godot docs show "
                           "every snippet in both languages by default)."),
    ] = None,
) -> list[dict]:
    """Semantic search across all Godot documentation (class reference + guides).
    This is the default tool for "how do I..." or "how does X work" questions --
    it returns the most relevant passages with a citation each, ranked by a
    cross-encoder reranker. Prefer this over related_docs unless you want a broad
    overview of a whole class/topic, and over find_examples unless you
    specifically need a code sample rather than an explanation."""
    return R.search(query, k=k, source_type=source_type, lang=lang)


@mcp.tool()
def find_examples(
    query: Annotated[
        str,
        Field(description="What you want a code sample for, e.g. 'move a "
                           "CharacterBody2D with velocity and gravity'."),
    ],
    k: Annotated[
        int, Field(description="Number of passages to return.")
    ] = 6,
    lang: Annotated[
        str | None,
        Field(description="'gdscript' or 'csharp' to keep only that language's "
                           "code in the returned text."),
    ] = None,
) -> list[dict]:
    """Find passages likely to contain a working code snippet for a task (tutorial
    examples and class-member usage). Same ranking as search_docs, but filtered to
    passages containing a fenced code block -- use this instead of search_docs when
    the user specifically wants code, not prose explanation."""
    hits = R.search(query, k=k * 4, pool=80, lang=lang)
    coded = [h for h in hits if "```" in h["text"]]
    return (coded or hits)[:k]


@mcp.tool()
def read_page(
    url: Annotated[
        str,
        Field(description="A url from a prior search_docs/find_examples/"
                           "related_docs hit. Passing an unrelated or hand-typed "
                           "url will not resolve."),
    ],
    max_chars: Annotated[
        int, Field(description="Truncate the returned text to this many "
                                "characters.")
    ] = 8000,
) -> dict:
    """Expand a search hit into its full source page, for when a passage looked
    relevant but was too short to answer the question on its own. Tutorial pages
    come back reconstructed in full; class-reference urls return the class
    overview instead of every member and point you at lookup_class/lookup_method
    for specifics, since a full class page can be very large."""
    return R.read_page(url, max_chars=max_chars)


@mcp.tool()
def related_docs(
    topic: Annotated[
        str,
        Field(description="A class name or general topic to explore broadly, "
                           "e.g. 'Area2D' or 'save system'."),
    ],
    k: Annotated[
        int, Field(description="Number of passages to return.")
    ] = 6,
) -> list[dict]:
    """Get a broad overview of a topic or class: if the topic names a class, its
    own overview and member docs come first, followed by semantically related
    passages from elsewhere in the docs. Use this for open-ended exploration
    ("tell me about Area2D", "what's relevant to save systems") rather than
    search_docs, which is better suited to answering one specific question."""
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
    # Block startup only on the index (needed by every tool, structured and RAG
    # alike) -- not the much larger embedder/reranker, which only search_docs/
    # find_examples/related_docs need. A client's connect timeout can be shorter
    # than the combined download, so keeping the blocking part small matters.
    L._con()
    R.warm_index()
    R.warm_models_async()
    mcp.run()


if __name__ == "__main__":
    main()
