"""Small text-cleanup helpers shared by the retrieval and lookup paths.

Deliberately dependency-light (``re`` + ``urllib.parse`` only) so ``lookups.py`` can
use it without pulling in the fastembed/vector stack that ``retrieval.py`` imports.
"""

from __future__ import annotations

import re
from urllib.parse import urljoin

# Markdown image: ![alt](src). Checked before the link regex when stripping.
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)\s]*\)")

# Markdown link: [label](href). Note this also matches the [..](..) tail of an
# image, which is fine — resolve rewrites the href, strip_images removes the whole
# image separately.
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")


def resolve_links(text: str, base_url: str) -> str:
    """Rewrite relative markdown-link hrefs to absolute URLs.

    Source-page links are relative to that page ("../../foo.html", "class_x.html",
    "#anchor"); once a chunk or member description is lifted out of context those are
    dead. Resolve each against ``base_url`` (the page the text came from) so the LLM
    gets a real citation it can open or pass to ``read_page``. Absolute http(s)/mailto
    hrefs pass through untouched.
    """
    base = base_url.split("#", 1)[0]

    def _sub(m: re.Match) -> str:
        label, href = m.group(1), m.group(2)
        if href.startswith(("http://", "https://", "mailto:")):
            return m.group(0)
        return f"[{label}]({urljoin(base, href)})"

    return _MD_LINK_RE.sub(_sub, text)


def strip_images(text: str) -> str:
    """Remove markdown image markup (``![alt](src)``) — an LLM can't use the pixels,
    and the src is noise. Collapse the blank line an isolated image leaves behind."""
    out = _MD_IMAGE_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", out).strip()
