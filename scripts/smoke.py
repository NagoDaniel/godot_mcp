"""Smoke test / driver for the Godot docs MCP server.

Launches the real server over stdio (the same way an MCP client does), lists its
tools, then calls each with representative inputs and asserts sane, *cited*
results. Exits non-zero on any failure.

    uv run python scripts/smoke.py

Doubles as the driver for a `/run-godot-mcp` skill.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = Path(__file__).resolve().parent.parent

# Windows consoles default to cp1252; our check names use → and — .
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PASS, FAIL = "PASS", "FAIL"
failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = PASS if cond else FAIL
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def _payload(result) -> object:
    """Extract a tool result as Python data (structured content or JSON text)."""
    if getattr(result, "structuredContent", None):
        sc = result.structuredContent
        return sc.get("result", sc) if isinstance(sc, dict) else sc
    text = result.content[0].text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


async def main() -> int:
    params = StdioServerParameters(
        command="uv", args=["run", "python", "-m", "godot_mcp.mcp_server"], cwd=str(REPO)
    )
    async with stdio_client(params) as (r, w):  
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = {t.name for t in (await s.list_tools()).tools}
            print("Tools:", ", ".join(sorted(tools)))

            expected = {
                "lookup_class", "lookup_method", "lookup_property", "lookup_signal",
                "lookup_enum", "lookup_constant", "show_inheritance",
                "search_symbols", "search_docs", "find_examples", "related_docs",
                "read_page",
            }
            check("all tools registered", expected <= tools, str(expected - tools))

            async def call(name, args):
                return _payload(await s.call_tool(name, args))

            # --- structured ---
            m = await call("lookup_method", {"class_name": "Node", "method": "add_child"})
            check("lookup_method Node.add_child", isinstance(m, dict) and "add-child" in m.get("url", ""))

            sym = await call("search_symbols", {"query": "body_entered", "kind": "signal"})
            check("search_symbols body_entered→Area2D",
                  any(h.get("class") == "area2d" for h in sym))

            inh = await call("show_inheritance", {"class_name": "Area2D"})
            check("show_inheritance Area2D⊃Node", "Node" in inh.get("inherits", []))

            # --- RAG ---
            sd = await call("search_docs", {"query": "how do 2D lights and shadows work"})
            check("search_docs 2d lights", bool(sd) and isinstance(sd, list))
            check("search_docs top hit is on-topic",
                  bool(sd) and any("2d_lights_and_shadows" in h.get("url", "") for h in sd[:5]),
                  detail=str([h.get("url", "").rsplit("/", 1)[-1] for h in sd[:5]]))
            check("search_docs hits are lean",
                  bool(sd) and set(sd[0]) == {"text", "title", "url", "score"},
                  detail=str(sorted(sd[0].keys()) if sd else []))

            fe = await call("find_examples", {"query": "move a CharacterBody2D with velocity"})
            check("find_examples returns code", bool(fe) and any("```" in h.get("text", "") for h in fe))

            # lang post-filter: gdscript request must drop csharp fences
            gd = await call("find_examples",
                            {"query": "move a CharacterBody2D with velocity", "lang": "gdscript"})
            check("find_examples lang=gdscript strips csharp",
                  bool(gd) and not any("```csharp" in h.get("text", "") for h in gd),
                  detail="a hit still contains a ```csharp block")

            rd = await call("related_docs", {"topic": "Area2D"})
            check("related_docs Area2D",
                  bool(rd) and any("area2d" in h.get("url", "").lower() for h in rd))

            # read_page: reconstruct a full guide page from a search hit's url
            page_url = next((h["url"] for h in sd if "2d_lights_and_shadows" in h.get("url", "")),
                            sd[0]["url"] if sd else "")
            pg = await call("read_page", {"url": page_url})
            check("read_page returns page text",
                  isinstance(pg, dict) and len(pg.get("text", "")) > len(sd[0]["text"]),
                  detail=str({k: (v if k != "text" else f"<{len(v)} chars>") for k, v in pg.items()})
                  if isinstance(pg, dict) else str(pg))

    print()
    if failures:
        print(f"FAILED ({len(failures)}): {', '.join(failures)}")
        return 1
    print("ALL SMOKE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
