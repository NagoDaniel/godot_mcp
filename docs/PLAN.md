# Godot Documentation MCP Server — Plan

## Context

The goal (`init_plan.md`) is an MCP server that gives coding agents accurate Godot
knowledge through two complementary surfaces: **RAG** (`search_docs`, `find_examples`,
`related_docs`) over prose, and **deterministic structured lookups**
(`lookup_class/method/property/signal/enum/constant`, `search_symbols`,
`show_inheritance`) over the class reference.

The immediate blocker is *how to parse the downloaded docs*
(`godot-docs-html-stable/`, ~1591 HTML files, 2.2 GB incl. images). The user's
instinct — strip navigation, keep main content — is correct, but the investigation
below shows we can do much better than a single "HTML → text" pipeline.

### Key findings from investigating the actual files

- Docs are **Sphinx / `sphinx_rtd_theme`** output. Real content lives in
  `<div role="main" class="document">`. Chrome to drop: `wy-nav-side`, `wy-nav-top`,
  `wy-side-nav-search`, `wy-breadcrumbs`, `rst-versions`, `<footer>`.
- **Class reference (1079 files under `classes/`)** is *semantically structured*.
  Sections carry stable ids: `#description`, `#properties`, `#methods`, `#signals`,
  `#enumerations`, `#constants`, plus `classref-descriptions-group` blocks with
  per-member detail anchors. → parse into **structured records**, not just text.
- **Tutorials/guides (392 files under `tutorials/`, plus `getting_started/`,
  `about/`, `community/`)** are prose with clean hierarchical `<section id=...>`.
  → header-aware chunking.
- **`objects.inv`** = Sphinx v2 inventory (Godot 4.7, zlib). Maps every symbol
  (class, method, property, signal, constant, tutorial page) → `file#anchor`.
  Parse with `sphobjinv`. This is the backbone for `search_symbols`, exact
  citations, and cross-linking.
- Code samples use `sphinx-tabs` — every snippet is duplicated as GDScript **and**
  C#. Must dedup / label by language during extraction, or chunks double up.

### Decisions locked with the user
- **Python** for ingestion and the MCP server.
- **BGE-M3** embeddings, run via **fastembed (ONNX)** rather than
  FlagEmbedding/torch — same model, but a lean CPU runtime (no 2 GB torch), which
  matters because the shipped `uvx` package must embed the user's *query* at
  runtime. fastembed also provides BGE-M3 sparse + BM25 for hybrid later.
- **Vector store: `sqlite-vec`** (see Milestone 2). Chosen over Chroma/Qdrant
  because the whole index folds into the existing single `store/godot.sqlite`
  file — trivial to ship as one release asset, no server to run.
- **Distribution: PyPI / `uvx godot-mcp`.** The index is a GitHub release asset the
  package fetches into a cache dir on first run (too big to bundle in the wheel).
- **MVP-first**: ship structured tools + dense semantic search, then layer on
  hybrid / reranker / query rewriting when traces expose failures.

---

## MILESTONE 2 (current) — Dense RAG + shareable `uvx` package

> Milestone 1 (extraction + structured MCP server) is **complete and verified**:
> `store/` holds `class_records.jsonl`, `guide_docs.jsonl`, `chunks.jsonl`
> (26,748 chunks), and `godot.sqlite` (29,586 symbols + 1,078 class records). The
> structured server (`server/mcp_server.py`, `server/lookups.py`) works over stdio.
> Milestone 2 adds semantic retrieval and makes it installable by others.

### Why sqlite-vec (the DB recommendation)
Everything already lives in one `store/godot.sqlite`. `sqlite-vec` adds vector
KNN *inside that same file* as a virtual table, so the entire product — structured
lookups, symbol search, and RAG — ships as **one portable file**. No server
process (unlike Qdrant), no second store directory (unlike Chroma), and FTS5 lives
in the same file to give BM25 for hybrid later without re-indexing or new infra.
The tradeoff vs Qdrant is native BGE-M3 *sparse* vectors; we cover the lexical
half of hybrid with FTS5/BM25 instead, which is enough for this corpus.

### New / changed files
- **`ingest/embed_index.py`** (new): load `chunks.jsonl`, embed each chunk's
  `text` with fastembed `BAAI/bge-m3` (dense, 1024-d, normalized), write into a
  `sqlite-vec` virtual table `vec_chunks(embedding float[1024])` plus a plain
  `chunk_meta` table (id, text, source_type, class, symbol, kind, url, anchor,
  title, section_path). Also build an FTS5 `fts_chunks` over `text` now (free,
  no model) to future-proof hybrid. Idempotent: drop+rebuild the three tables.
- **`server/retrieval.py`** (new): open `godot.sqlite` read-only + load sqlite-vec;
  lazily load the fastembed model once; `search(query, k, filters)` embeds the
  query, runs KNN, joins `chunk_meta`, returns chunks with `url#anchor` citations
  and scores. Thin interface so BM25/hybrid + reranker slot in later. Emits a
  one-line trace per query (query → top-k ids + distances) per `RAG_info.md`.
- **`server/mcp_server.py`** (edit): add the three RAG tools —
  - `search_docs(query, k=6, source_type=None)` → semantic chunks + citations.
  - `find_examples(query, k=6)` → same, filtered to code-bearing chunks
    (`kind='guide'` chunks containing a fenced block, or class member chunks).
  - `related_docs(topic, k=6)` → resolve `topic` via the symbol index
    (`server/lookups.py:search_symbols`) to a class/page, then return that
    page's chunks + nearest vector neighbors.
- **Packaging** — restructure the server into an installable package:
  `src/godot_mcp/` (move `mcp_server.py`, `lookups.py`, `retrieval.py`, package
  relative imports, drop the `sys.path` hacks); `pyproject.toml` gains
  `[project.scripts] godot-mcp = "godot_mcp.mcp_server:main"` and runtime deps
  (`mcp`, `sqlite-vec`, `fastembed`). Ingestion deps (`beautifulsoup4`, `lxml`,
  `markdownify`, `sphobjinv`) move to an optional `[ingest]` extra — end users
  don't need them. `ingest/` stays a dev-only script dir, not shipped.
- **`src/godot_mcp/data.py`** (new): resolve the index path — env override
  `GODOT_MCP_DB`, else a user cache dir (`platformdirs`); if absent, download the
  prebuilt `godot.sqlite` from a pinned GitHub release URL (with a checksum) and
  cache it. Keeps the wheel small; first run bootstraps the index.

### Embedding & runtime notes
- One-time offline embed of 26,748 chunks with fastembed BGE-M3 on the M4 CPU
  (ONNX, arm64) — minutes, not hours. Produces a `godot.sqlite` of ~130–150 MB
  (≈110 MB of float32 vectors). That file is the shipped release asset.
- At runtime the server loads the fastembed model once (first query pays a few
  seconds; fastembed caches the ONNX model in `~/.cache`). Query and document
  embeddings **must** use the same model — BGE-M3 is now locked.
- Confirmed: this uv-managed Python supports `sqlite3.enable_load_extension`, so
  the `sqlite-vec` loadable extension works here. (Gotcha to document: stock
  system Pythons sometimes disable it.)

### Testing — smoke driver (per user choice)
- **`scripts/smoke.py`** (new): the single runnable harness. It launches the real
  server over stdio (via `mcp` `stdio_client`), lists tools, then calls each tool
  with representative inputs and asserts sane, *cited* results:
  - `lookup_method("Node","add_child")` → signature present, url has the anchor.
  - `search_symbols("body_entered", kind="signal")` → Area2D hit.
  - `show_inheritance("Area2D")` → chain includes `Node`.
  - `search_docs("how do 2D lights and shadows work")` → a top hit whose url
    contains `2d_lights_and_shadows`.
  - `find_examples("move a CharacterBody2D")` → a chunk containing a code fence.
  - `related_docs("Area2D")` → chunks from `class_area2d`.
  Exit non-zero on any failure. Doubles as the driver for a `/run-godot-mcp`
  skill (the earlier `/run-skill-generator` invocation).
- A short **`eval/golden_questions.md`** (~15 Q→expected-url-substring) lives
  alongside as documentation; `smoke.py` runs a handful inline. A full
  recall@k/MRR harness is deferred (not needed for MVP smoke).

### Milestone 2 verification (end-to-end)
1. `uv run python ingest/embed_index.py` → prints chunk/vector counts; `godot.sqlite`
   grows to ~130–150 MB with `vec_chunks`, `chunk_meta`, `fts_chunks` populated.
2. `uv run python scripts/smoke.py` → all tool assertions pass, exit 0.
3. `uvx --from . godot-mcp` (or `uv run godot-mcp`) starts the server via the
   console entry point; register in Claude Desktop / Cursor and confirm a
   `search_docs` call returns cited Godot answers.

## Architecture

```
godot-docs-html-stable/
  ├─ objects.inv ──────────────► symbol_index (sqlite)  ── search_symbols, citations
  ├─ classes/*.html ── parse ──► class_records (json/sqlite) ── lookup_* , show_inheritance
  │                       └────► clean text ─┐
  └─ tutorials/**/*.html ─ parse ─ main ─────┤─► chunker ─► BGE-M3 ─► Chroma (vectors)
                                             │                         └─ search_docs, find_examples, related_docs
                                    (metadata: source_type, class, symbol, url#anchor, section, lang)
```

Two ingestion pipelines feeding three stores (vector, structured, symbol), all
queried by one MCP server.

## Component 1 — HTML extraction (answers the core question)

Yes, strip everything except main content — but branch by page type. Use
`BeautifulSoup` (`lxml` parser).

**Shared cleanup helper:** select `div[role=main]`, remove nav/footer/version
chrome (selectors above), strip `<script>`/`<style>`, resolve `sphinx-tabs` into
labeled fenced code blocks (` ```gdscript ` / ` ```csharp `) rather than duplicated
prose, rewrite internal `href`s to canonical `file#anchor` for citations.

**Pipeline A — class reference** (`classes/class_*.html`): walk the structured
sections into one record per class:
```
{ name, inherits[], brief, description(md),
  properties[{name, type, default, description, anchor}],
  methods[{name, return, args[], description, anchor}],
  signals[...], enums[...], constants[...], theme_items[...],
  tutorial_links[], url }
```
This record set *is* the structured index and *also* yields clean per-member text
for RAG. Inheritance chain comes from the `Inherits:` line + `objects.inv`.

**Pipeline B — guides/tutorials** (everything else): keep the cleaned main content,
convert to Markdown (`markdownify`) preserving headers, lists, code blocks, tables,
and admonitions (note/warning). Carry the `<section>` hierarchy as heading metadata.

## Component 2 — Symbol index (from `objects.inv`)

Parse with `sphobjinv` into a small SQLite table:
`(symbol, kind, class, uri, anchor, display_name)`. Powers `search_symbols`
(prefix/fuzzy over ~10k+ symbols), disambiguation, and precise citation links for
every RAG answer. Cheap, deterministic, no embeddings.

## Component 3 — Chunking

Per `RAG_info.md` defaults: **recursive split ~512 tokens, 50–100 overlap**,
header-aware.
- **Tutorials:** `MarkdownHeaderTextSplitter` → recursive, so chunks never cross a
  section boundary; keep code blocks intact (don't split mid-code).
- **Class refs:** natural unit = one member description (method/property/signal) =
  one chunk, with parent = the class (parent/child pattern from `RAG_info.md`:
  retrieve the member, can expand to class context).
- **Metadata on every chunk:** `source_type` (class_ref | tutorial),
  `class`, `symbol`, `url`, `anchor`, `section_path`, `lang` (for code),
  `version` (4.7). Enables metadata filtering later.
  > **Shipped reality (M2.5):** the per-chunk `lang` field was *not* implemented —
  > GDScript/C# stay as adjacent fenced blocks inside one chunk's `text`. Language
  > selection is instead a **query-time post-filter** (`search_docs`/`find_examples`
  > `lang=` param in `retrieval._filter_lang`), which avoids a full re-index. A
  > `read_page(url)` tool reconstructs full guide pages for "read more". `version`
  > is likewise not stamped yet (deferred to M4).
- Tables → keep as whole Markdown chunks (per `RAG_info.md` semi-structured guidance).

## Component 4 — Embeddings + vector store

> **Superseded by the Milestone 2 section above.** Final choice: **BGE-M3 via
> fastembed (ONNX)** into **`sqlite-vec`** (in-file vectors), not
> FlagEmbedding+Chroma — for a lean, single-file, `uvx`-shippable package. FTS5 in
> the same file covers BM25 for hybrid later. Embeddings remain model-locked to
> BGE-M3 (switching = full re-index).

## Component 5 — Structured store

SQLite (or JSON files) from Component 1 records. Directly answers `lookup_class`,
`lookup_method`, `lookup_property`, `lookup_signal`, `lookup_enum`,
`lookup_constant`, and `show_inheritance` (recursive over `inherits`) with zero
model calls — fast, exact, no hallucination.

## Component 6 — MCP server

Python **MCP SDK** (`mcp` package), stdio transport (works in Cursor / Claude /
VS Code). One tool per `init_plan.md` entry:

- **RAG tools** → dense query over Chroma, return chunks with `url#anchor`
  citations. `find_examples` filters `source_type` for code-bearing chunks;
  `related_docs` uses the symbol index + vector neighbors.
- **Structured tools** → direct SQLite reads.

Return compact, agent-friendly payloads (Markdown + a `source` URL each), not raw
HTML. Add lightweight tracing (log query → retrieved chunks + scores) from day one,
per `RAG_info.md`.

## Build order (milestones)

1. **Extraction + indexes (no ML).** Pipelines A/B + `objects.inv` parse →
   `class_records`, `chunks.jsonl`, `symbol_index`. Ship `lookup_*`,
   `show_inheritance`, `search_symbols` as a working structured-only MCP server.
   *This alone is immediately useful to agents.*
2. **Dense RAG + package (current).** Embed chunks with BGE-M3 (fastembed) →
   `sqlite-vec` in `godot.sqlite`. Wire `search_docs`, `find_examples`,
   `related_docs`. Ship as `uvx godot-mcp` with the index as a release asset.
   Smoke driver for testing. MVP complete. *(See the Milestone 2 section above.)*
3. **Layer up (as traces demand):** FTS5/BM25 (already in-file) → hybrid + RRF,
   `bge-reranker-v2-m3` reranker (retrieve 50 → top 5), query rewriting.
4. **Ingestion hygiene:** version stamp, incremental re-index on doc updates,
   dedup.

## Critical files / new layout (proposed)

```
rag_godot/
  ingest/
    clean.py        # shared BeautifulSoup cleanup + sphinx-tabs handling
    parse_classes.py# Pipeline A → class_records
    parse_guides.py # Pipeline B → markdown docs
    inventory.py    # objects.inv → symbol_index (sphobjinv)
    chunk.py        # header-aware + recursive splitting
    embed_index.py  # BGE-M3 → Chroma
  store/            # chroma/, godot.sqlite (class_records + symbol_index)
  server/
    mcp_server.py   # tool definitions
    retrieval.py    # search/rerank behind an interface
    lookups.py      # structured queries
  pyproject.toml
```
Reused libraries: `beautifulsoup4`+`lxml`, `markdownify`, `sphobjinv`,
`FlagEmbedding`, `chromadb`, `mcp`.

## Verification

- **Extraction:** spot-check `class_aabb.html` and `class_node.html` records
  (methods/properties/signals present, no nav text leaked); confirm sphinx-tabs
  produce one gdscript + one csharp block, not duplicated prose. Confirm counts:
  ~1079 class records, symbol_index non-empty.
- **Structured tools:** `lookup_method("AABB","intersects")`,
  `show_inheritance("Sprite2D")` return correct data matching the HTML.
- **RAG:** golden-set of ~15 real Godot questions (e.g. "how do 2D lights and
  shadows work", "how to detect body entering an Area2D"); assert the right
  tutorial/class chunk lands in top-k with a valid `url#anchor`. Log traces.
- **MCP integration:** run the server over stdio; register in Cursor/Claude
  Desktop; confirm tools are callable and return cited answers end-to-end.

## Open / future (not blocking v1)

- Additional sources from `init_plan.md` (demo projects, release notes) — same
  pipeline B, add `source_type`.
- Hybrid + reranker + agentic multi-hop retrieval — deferred to milestone 3 per the
  MVP decision.
