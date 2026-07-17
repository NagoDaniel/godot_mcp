"""Shared HTML cleanup for the Godot Sphinx (sphinx_rtd_theme) docs.

Responsibilities:
- isolate the real page content (``div[role=main]``)
- drop nav / footer / version chrome and header-link glyphs
- turn ``sphinx-tabs`` code widgets into labeled fenced code blocks so a GDScript
  + C# pair does not get emitted as duplicated prose
- convert cleaned content to Markdown, preserving headers/lists/code/tables

The functions here are used by both ingestion pipelines (class reference and
prose guides).
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

from bs4 import BeautifulSoup, Tag
from markdownify import MarkdownConverter

# Chrome that can appear *inside* div[role=main] (the side/top nav lives outside
# it, so isolating role=main already removes most of the theme).
_CHROME_SELECTORS = [
    ".rst-footer-buttons",
    ".rst-versions",
    ".wy-breadcrumbs",
    "footer",
    ".headerlink",  # the 🔗 anchor glyph after every heading/member
]

# sphinx-tabs panel ids end in a base64-encoded language name, e.g.
# ``...-R0RTY3JpcHQ=`` -> "GDScript", ``...-QyM=`` -> "C#".
_LANG_MAP = {"GDScript": "gdscript", "C#": "csharp", "C++": "cpp", "shell": "shell"}


def load_main(path: str | Path) -> Tag:
    """Parse an HTML file and return its ``div[role=main]`` content, cleaned."""
    soup = BeautifulSoup(Path(path).read_text(encoding="utf-8"), "lxml")
    main = soup.find("div", attrs={"role": "main"})
    if main is None:  # a few index pages differ; fall back to <body>
        main = soup.body or soup
    _strip_chrome(main)
    _resolve_sphinx_tabs(main)
    return main


def _strip_chrome(main: Tag) -> None:
    for sel in _CHROME_SELECTORS:
        for el in main.select(sel):
            el.decompose()
    for el in main.find_all(["script", "style"]):
        el.decompose()


def _panel_language(panel: Tag) -> str:
    """Recover a code language label from a sphinx-tabs panel id."""
    pid = panel.get("id", "")
    token = pid.rsplit("-", 1)[-1] if "-" in pid else ""
    try:
        decoded = base64.b64decode(token + "===").decode("utf-8", "ignore")
    except Exception:
        decoded = ""
    return _LANG_MAP.get(decoded, decoded.lower() or "gdscript")


def _resolve_sphinx_tabs(main: Tag) -> None:
    """Replace each sphinx-tabs widget with plain fenced ``<pre>`` code blocks.

    Sphinx renders one tab (and one hidden panel) per language for the *same*
    snippet. We keep every language once, labeled, and drop the tab chrome so the
    text extractor never sees a snippet twice.
    """
    for widget in main.select("div.sphinx-tabs"):
        replacements: list[Tag] = []
        for panel in widget.select("div.sphinx-tabs-panel"):
            code = panel.find(["pre", "code"])
            if code is None:
                continue
            lang = _panel_language(panel)
            fence = BeautifulSoup(
                f'<pre data-lang="{lang}"></pre>', "lxml"
            ).pre
            fence.string = code.get_text()
            replacements.append(fence)
        widget.replace_with(*replacements) if replacements else widget.decompose()


class _GodotMarkdown(MarkdownConverter):
    """Markdown converter that respects our fenced code language hints."""

    def convert_pre(self, el, text, parent_tags):
        lang = el.get("data-lang", "")
        code = el.get_text()
        return f"\n```{lang}\n{code.rstrip()}\n```\n"


# Godot symbols are full of underscores/asterisks (set_process, *args); letting
# markdownify backslash-escape them corrupts identifiers for display and search.
_MD = _GodotMarkdown(
    heading_style="ATX",
    bullets="-",
    escape_underscores=False,
    escape_asterisks=False,
    escape_misc=False,
)


def to_markdown(node: Tag) -> str:
    """Convert a cleaned bs4 node to normalized Markdown."""
    md = _MD.convert_soup(node)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def clean_text(node: Tag) -> str:
    """Plain-text of a node with the header-link glyph and whitespace tidied."""
    text = node.get_text(" ", strip=True)
    return re.sub(r"\s*🔗\s*", " ", text).strip()


# TEST:

if __name__ == "__main__":
    import sys

    main = load_main(sys.argv[1])
    print(to_markdown(main))