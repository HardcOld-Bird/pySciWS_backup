---
name: literature-research
description: Search, read, evaluate, and manage academic physics literature (acoustics, non-Hermitian / exceptional points, topological, BIC, CPA, phononic). Retrieval runs on a community multi-source search MCP (primary) plus a unified `research` CLI: OpenAlex + arXiv search, paywalled full text via headless browser, PDF → Markdown with LaTeX equations (MinerU), journal-impact and citation enrichment, structured evaluation notes, Zotero sync, and semantic search over the user's own extracted corpus (local PaperQA2 index). Use when the user asks to find or search papers/literature, read or analyze a paper, extract a PDF, run a literature review or survey, assess a paper's novelty / rigor / journal tier, build paper notes, query and manage their reference library, or semantically search / ask questions across papers they have collected.
---

# Literature Research

One CLI drives the whole workflow: **search → read → note → library → index**. Retrieval and library
access each run on two tiers — a community **MCP** is primary, the `research` CLI is fallback plus the
differentiated layer.

The backend in `src/pysci/skills/literature_research/tools/` is a **black box**: drive it through the CLI,
open code only when maintaining it ([maintenance.md](references/maintenance.md)). One caveat — `python -m …`
debug backdoors: [sources.md](references/sources.md#debug-backdoors--the-one-caveat-to-black-box).

## Who does what

| Task | Primary (community MCP) | `research` CLI for |
|---|---|---|
| find papers | `paper-search-mcp` `search_papers` — ~22 sources, deduped | `search` when the MCP is down; **always** for frontmatter normalization, WoS `--enrich`, the arXiv LaTeX source `read` needs |
| OA full text | `paper-search-mcp` `download_with_fallback` (source-native → OpenAIRE/CORE/PMC → Unpaywall) | `browser_fetch` — **paywalled / Cloudflare** (institutional-IP Playwright); no MCP replaces it |
| deep arXiv | `arxiv-mcp-server` — `get_paper_latex`, `read_paper_section`, `semantic_search`, `watch_topic` | `citegraph` — snowballs everything OpenAlex indexes, both ways (the MCP's `citation_graph` is arXiv-only) |
| Zotero | `zotero-mcp` (server `zotero`) — search, metadata / BibTeX / full text / **PDF page images**, add by DOI/URL/ISBN/BibTeX, merge, **Scite retraction alerts** | `add` / `library` / `tex refs` — delegate to `zotero-cli --json`; own frontmatter → BibTeX normalization + `zotero_key`/`zotero_uri` write-back |

Setup and provenance: `scripts/paper_search_mcp/`, `scripts/zotero_mcp/`. `zotero-mcp` needs a
**persistent** `uv tool install "zotero-mcp-server[pdf,scite]"`; without it `add` degrades to a skeleton
note and `library`/`refs` print a notice.

## Two gates before writing anything

- **Citation integrity.** `add` cross-checks each citation against OpenAlex + Crossref + arXiv **before**
  writing; a hard conflict blocks it (`--allow-fail` overrides, `--no-verify` skips). `citecheck` audits
  what is already written, exit 1 on any FAIL. Severity model and **what the gate cannot check**:
  [citecheck.md](references/citecheck.md).
- **Semantic RAG over your own corpus.** `rag index` embeds everything `read`/`ingest` extracted;
  **`rag search '<q>'` is the main tool here** — free, never blocks, returns chunks with citation + path
  for you to synthesize. Needs `SILICONFLOW_API_KEY`: [rag.md](references/rag.md).

## Invocation

From the **project root**: `uv run pysci-research <command> [options]`, below abbreviated to `research …`.
(Not installed? `python -m pysci.skills.literature_research.tools.research`.)

> **PowerShell:** this CLI prints Chinese — apply `.qoder/rules/basic.md` §3 or you read mojibake.

## Commands

| Group | Commands | For |
|---|---|---|
| health | `doctor` · `cache stats｜clean｜prune` | self-check; disk pressure |
| discover | `search "<query>"` · `citegraph <id>` · `journal lookup｜build-scimago｜status` | topic search; snowball both ways; weigh a venue |
| read | `read <doi｜url｜arxiv-id>` · `get <id>` | full text + note skeleton; metadata only |
| library | `add <doi>` · `library ping｜list｜search｜get` · `index [--fix]` | add via the citation gate; query Zotero; rebuild `INDEX.md` |
| verify | `citecheck <doi｜id｜title> [--all｜--bib｜--review]` | ✓PASS / △WARN / ✗FAIL / ?NOT_FOUND |
| corpus | `rag index｜search｜ask｜status` · `ingest [run｜status]` | semantic search of your corpus; bulk local PDFs |
| surveys | `review new｜status｜sync` | scaffold + audit a survey |

Two shapes: **verb + target** (`search`, `read`, `get`, `add`, `citegraph`, `citecheck`, `index`,
`doctor`) and **noun + `action`** (`library`, `rag`, `cache`, `journal`, `review`, `ingest`). `--json`
makes stdout **only** JSON (progress and warnings go to stderr).

## Workflow: topic → reviewed notes

```
- [ ] 1. research search 'non-Hermitian acoustic exceptional point' --save --purpose '<goal>'
- [ ] 2. Screen the shortlist (→ shortlists/); pick the keepers
- [ ] 3. research add <doi>          # per keeper → citation gate → Zotero + note skeleton
- [ ] 4. (optional) research citegraph <seed-id> --save        # snowball → back to step 2
- [ ] 5. research read 10.1103/PhysRevLett.121.124501          # close reads → full text
- [ ] 6. Fill each note's TLDR / Key Claims / Novelty / Rigor / Relevance (see references/read.md)
- [ ] 7. research index              # refresh INDEX.md
- [ ] 8. (optional) research review new '<topic>' --from-shortlist shortlists/<f>.md
- [ ] 9. Write the survey yourself; then research citecheck --review   # audit every reference
```

`read` tries the **open-access direct link first**, falls back to the publisher page headlessly, extracts
to Markdown (equations → LaTeX via MinerU, or the arXiv LaTeX source for preprints) and writes a note to
`papers/`. It prints the full-text path — **Read that file to actually read the paper**. Blocked? Add
`--headed`; re-runs reuse the cache unless `--refresh`. Fetch order: [read.md](references/read.md).

Step 4 needs no new query string: `citegraph` returns the seed's references (**backward**) and citing
papers (**forward**), and `--save` re-enters step 2. Step 8 scaffolds
`reviews/{YYYY-MM}_{topic}_survey.md` then **stops** — the prose is your judgement; `review status` audits
links and counters, `review sync` recomputes them. **A local PDF folder with no DOIs** goes through
`ingest`, not `read`/`add`: [ingest.md](references/ingest.md).

## Output locations

All under `data/skills/literature_research/`: `papers/` notes (`{year}_{author}_{slug}.md`) · `shortlists/`
· `reviews/` · `INDEX.md` · `ingest/manifest.json` · `data/` (`scimago_index.json` + `SOURCE.md`) ·
`cache/` PDFs and browser bundles (git-ignored) **but `cache/extracted/` IS git-tracked** · `cache/rag/`
(rebuildable).

## When something breaks

`research doctor` reports every source, backend and Zotero status. Browser-fetch / MinerU / adapter /
config: [maintenance.md](references/maintenance.md). Source limits and where journal-quality numbers come
from: [sources.md](references/sources.md).

## Reference files

（`*`=分册索引，按需读单册）

- [search.md](references/search.md) * — `search` / `get` / `library` / `journal` / `citegraph`: flags, screening, shortlist schema, snowballing.
- [read.md](references/read.md) * — `read` / `add` / `citecheck` / `index` / `review`: fetch order, extraction, note-merge contract, note schema, **evaluation rubric**.
- [citecheck.md](references/citecheck.md) — the citation gate: three sources, severity model, audit modes.
- [rag.md](references/rag.md) — the local semantic index: cleaning, incremental indexing, cost, `.env` keys.
- [ingest.md](references/ingest.md) — bulk local-PDF ingestion: manifest ledger, priorities, resumability.
- [sources.md](references/sources.md) — data sources and their limits, debug backdoors.
- [maintenance.md](references/maintenance.md) * — backend internals; fixing and extending.
