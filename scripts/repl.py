#!/usr/bin/env python
"""Interactive REPL for testing godot-mcp tools.

Usage:
  search_docs how to move a character
  lookup_class CharacterBody2D
  lookup_method CharacterBody2D move_and_slide
  search_symbols Node
  find_examples collision detection
  related_docs physics
  show_inheritance Node2D
  lookup_property CharacterBody2D velocity
  lookup_signal Node tree_entered
  lookup_enum Node ProcessMode
  lookup_constant Node NOTIFICATION_ENTER_TREE
  read_page physics/2d/physics_body_2d

Type 'help' or 'quit' to exit.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from godot_mcp import mcp_server

TOOLS = {
    "search_docs": mcp_server.search_docs,
    "lookup_class": mcp_server.lookup_class,
    "lookup_method": mcp_server.lookup_method,
    "lookup_property": mcp_server.lookup_property,
    "lookup_signal": mcp_server.lookup_signal,
    "lookup_enum": mcp_server.lookup_enum,
    "lookup_constant": mcp_server.lookup_constant,
    "show_inheritance": mcp_server.show_inheritance,
    "search_symbols": mcp_server.search_symbols,
    "find_examples": mcp_server.find_examples,
    "related_docs": mcp_server.related_docs,
    "read_page": mcp_server.read_page,
}


def parse_input(line: str) -> tuple[str, list[str], dict[str, str]]:
    """Parse 'tool_name query string key=value' into (tool, args, kwargs).

    Everything before first key=value is the query (joined with spaces).
    """
    parts = line.strip().split()
    if not parts:
        return "", [], {}

    tool = parts[0]
    remaining = " ".join(parts[1:])

    args = []
    kwargs = {}

    # Find first key=value pattern
    import re
    match = re.search(r'\s+(\w+)=', remaining)
    if match:
        # Split into query part and kwargs part
        query_part = remaining[:match.start()].strip()
        kwargs_part = remaining[match.start():].strip()

        if query_part:
            args = [query_part]

        # Parse all key=value pairs (coerce digit-only values to int)
        for kv in kwargs_part.split():
            if "=" in kv:
                k, v = kv.split("=", 1)
                kwargs[k] = int(v) if v.isdigit() else v
    else:
        # No kwargs, whole remaining is query
        if remaining:
            args = [remaining]

    return tool, args, kwargs


def format_result(result: dict | list) -> str:
    """Pretty-print result."""
    if isinstance(result, dict) and "error" in result:
        return f"ERROR: {result['error']}"
    return json.dumps(result, indent=2)


def main() -> None:
    print("godot-mcp REPL")
    print("Type 'help' for usage, 'quit' to exit\n")

    while True:
        try:
            line = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nquit")
            break

        if not line or line.startswith("#"):
            continue

        if line.lower() in ("help", "?"):
            print(__doc__)
            continue

        if line.lower() in ("quit", "exit"):
            break

        tool, args, kwargs = parse_input(line)

        if tool not in TOOLS:
            print(f"unknown tool: {tool}")
            print(f"available: {', '.join(sorted(TOOLS.keys()))}")
            continue

        try:
            fn = TOOLS[tool]
            result = fn(*args, **kwargs)
            # tool handlers are async coroutines; drive to completion
            if inspect.iscoroutine(result):
                result = asyncio.run(result)
            print(format_result(result))
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}")

        print()


if __name__ == "__main__":
    main()
