"""Pipeline A — parse Godot class-reference pages into structured records.

Each ``classes/class_*.html`` page becomes one record holding the class's
inheritance, description, and every member (method, property, signal, constant,
enum, operator, constructor, annotation, theme item). Members are identified by
their ``<p class="classref-*" id=...>`` signature blocks, which carry both a
stable anchor id and a one-line signature; the following sibling nodes up to the
next member block are the member's description.

The resulting records power the structured lookup tools directly and also feed
clean per-member text into the RAG chunker.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bs4 import Tag

from clean import clean_text, load_main, to_markdown

# member <p> class -> record bucket
_MEMBER_KINDS = {
    "classref-constructor": "constructors",
    "classref-method": "methods",
    "classref-operator": "operators",
    "classref-property": "properties",
    "classref-signal": "signals",
    "classref-constant": "constants",
    "classref-enumeration": "enums",
    "classref-annotation": "annotations",
    "classref-themeproperty": "theme_items",
}

_HREF_BASE = "classes/{stem}.html"


@dataclass
class Member:
    name: str
    signature: str
    anchor: str
    description_md: str
    return_type: str | None = None
    args: str | None = None
    qualifiers: str | None = None
    default: str | None = None
    value: str | None = None
    enum_values: list[str] = field(default_factory=list)


@dataclass
class ClassRecord:
    name: str
    url: str
    inherits: list[str] = field(default_factory=list)
    inherited_by: list[str] = field(default_factory=list)
    brief: str = ""
    description_md: str = ""
    tutorial_links: list[dict] = field(default_factory=list)
    constructors: list[Member] = field(default_factory=list)
    methods: list[Member] = field(default_factory=list)
    operators: list[Member] = field(default_factory=list)
    properties: list[Member] = field(default_factory=list)
    signals: list[Member] = field(default_factory=list)
    constants: list[Member] = field(default_factory=list)
    enums: list[Member] = field(default_factory=list)
    annotations: list[Member] = field(default_factory=list)
    theme_items: list[Member] = field(default_factory=list)


def _chain(text: str) -> list[str]:
    """'Node2D < CanvasItem < Node < Object' -> ['Node2D', ...]."""
    body = text.split(":", 1)[1] if ":" in text else text
    return [p.strip() for p in re.split(r"[<,]", body) if p.strip()]


def _member_description(sig_p: Tag) -> str:
    """Markdown of everything from a member signature block up to the next one."""
    parts: list[str] = []
    for sib in sig_p.find_next_siblings():
        if isinstance(sib, Tag) and _member_kind(sib):
            break
        parts.append(to_markdown(sib))
    return "\n\n".join(p for p in parts if p).strip()


def _member_kind(el: Tag) -> str | None:
    classes = el.get("class") or []
    for c in classes:
        if c in _MEMBER_KINDS:
            return _MEMBER_KINDS[c]
    return None


# signature parsers ---------------------------------------------------------

_CALL_RE = re.compile(r"^\s*(?:(?P<ret>[\w\[\]\.]+)\s+)?(?P<name>[\w\+\-\*/%<>=!&|~^\[\]]+|operator\s*\S+)\s*\((?P<args>.*)\)\s*(?P<qual>.*)$")
_ASSIGN_RE = re.compile(r"^\s*(?:(?P<type>[\w\[\]\.]+)\s+)?(?P<name>[\w]+)\s*=\s*(?P<val>.+)$")


def _parse_member(sig_p: Tag, kind: str) -> Member:
    sig = clean_text(sig_p)
    anchor = sig_p.get("id", "")
    desc = _member_description(sig_p)
    m = Member(name=sig, signature=sig, anchor=anchor, description_md=desc)

    if kind in ("methods", "constructors", "operators", "signals", "annotations"):
        cm = _CALL_RE.match(sig)
        if cm:
            m.name = cm.group("name").strip()
            m.return_type = (cm.group("ret") or "").strip() or None
            m.args = cm.group("args").strip()
            m.qualifiers = (cm.group("qual") or "").strip() or None
    elif kind in ("properties", "theme_items"):
        am = _ASSIGN_RE.match(sig)
        if am:
            m.name = am.group("name")
            m.return_type = (am.group("type") or "").strip() or None
            m.default = am.group("val").strip()
        else:  # some properties have no default
            m.name = sig.split()[-1] if sig.split() else sig
    elif kind == "constants":
        am = _ASSIGN_RE.match(sig)
        if am:
            m.name = am.group("name")
            m.value = am.group("val").strip()
    elif kind == "enums":
        em = re.match(r"(?:enum|flags)\s+(\w+)", sig)
        if em:
            m.name = em.group(1)
        # enum values are classref-enumeration-constant blocks that follow the
        # header up to the next enum / member block.
        value_texts: list[str] = []
        for sib in sig_p.find_next_siblings():
            if isinstance(sib, Tag) and _member_kind(sib):
                break
            classes = sib.get("class") or [] if isinstance(sib, Tag) else []
            if "classref-enumeration-constant" in classes:
                vt = clean_text(sib)
                value_texts.append(vt)
                vm = re.search(r"\b([A-Z][A-Z0-9_]+)\s*=", vt)
                if vm:
                    m.enum_values.append(vm.group(1))
        if not desc and value_texts:
            m.description_md = "\n\n".join(value_texts)
    return m


def parse_class_file(path: Path) -> ClassRecord:
    main = load_main(path)
    stem = path.stem
    name = main.find(["h1"]).get_text(strip=True).replace("¶", "").strip() if main.find("h1") else stem
    rec = ClassRecord(name=name, url=_HREF_BASE.format(stem=stem))

    for para in main.find_all("p", recursive=True):
        t = para.get_text(" ", strip=True)
        if t.startswith("Inherits:") and not rec.inherits:
            rec.inherits = _chain(t)
        elif t.startswith("Inherited By:") and not rec.inherited_by:
            rec.inherited_by = _chain(t)

    desc_sec = main.find("section", id="description")
    if desc_sec:
        rec.description_md = to_markdown(desc_sec)
        first_p = desc_sec.find("p")
        if first_p:
            rec.brief = clean_text(first_p)[:400]

    tut = main.find("section", id="tutorials")
    if tut:
        for a in tut.find_all("a", href=True):
            rec.tutorial_links.append({"title": a.get_text(strip=True), "url": a["href"]})

    seen: set[str] = set()
    for sig_p in main.find_all("p"):
        kind = _member_kind(sig_p)
        if not kind:
            continue
        key = f"{kind}:{sig_p.get('id','')}"
        if key in seen:
            continue
        seen.add(key)
        getattr(rec, kind).append(_parse_member(sig_p, kind))
    return rec


def record_to_dict(rec: ClassRecord) -> dict:
    d = asdict(rec)
    return d


def main() -> None:
    root = Path("godot-docs-html-stable/classes")
    out = Path("store/class_records.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in root.glob("class_*.html") if p.name != "class_index.html")
    n = 0
    with out.open("w", encoding="utf-8") as fh:
        for p in files:
            try:
                rec = parse_class_file(p)
            except Exception as e:  # noqa: BLE001
                print(f"  !! {p.name}: {e}")
                continue
            fh.write(json.dumps(record_to_dict(rec), ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} class records -> {out}")

def test_one_file() -> None:
    rec = parse_class_file(Path("godot-docs-html-stable/classes/class_vector2.html"))
    print(json.dumps(record_to_dict(rec), ensure_ascii=False, indent=2))
    assert rec.name == "Node2D"
    assert rec.inherits == ["CanvasItem"]
    assert len(rec.methods) > 10
    assert len(rec.properties) > 10
    assert len(rec.signals) > 0

if __name__ == "__main__":
    main()
    #test_one_file()


