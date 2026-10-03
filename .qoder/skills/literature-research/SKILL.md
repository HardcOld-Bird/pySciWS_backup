---
name: literature-research
description: Search, read, evaluate, and manage academic physics literature (acoustics, non-Hermitian / exceptional points, topological, BIC, CPA, phononic). Retrieves via a community multi-source search MCP (primary path) plus a unified `research` CLI (fallback and differentiated layer) that fuses OpenAlex + arXiv search, fetches paywalled full text with a headless browser, extracts PDFs to Markdown with LaTeX equations (MinerU cloud), enriches metadata with journal impact factors and citation counts, generates structured evaluation notes, syncs to Zotero, and semantically searches the user's own extracted corpus via a local embedding index (PaperQA2 + free embeddings). Use when the user asks to find or search papers/literature, read or analyze a specific paper, extract a PDF, run a literature review or survey, assess a paper's novelty / rigor / journal tier, build paper notes, query and manage their reference library, or semantically search / ask questions across the papers they have already collected.
---

# Literature Research

A single CLI drives the whole literature workflow: **search → read → note → library → index**.
Retrieval itself has two tiers — a community **search MCP** is the primary multi-source path, and
the `research` CLI is the fallback plus the differentiated layer. See **Retrieval layers** below.

The backend lives in `src/pysci/skills/literature_research/` (11 backend modules + a `research` facade).
**You do not need to read the backend code** — treat it as a black box and drive everything
through the `research` CLI below. Only open the code when maintaining it (see
[references/maintenance.md](references/maintenance.md)).

## Retrieval layers (MCP primary + CLI fallback)

Paper discovery runs on two tiers, mirroring the comsol-simulation architecture
(“community MCP for the commodity work + in-house CLI for the differentiated work + fallback”):

- **Primary — community `paper-search-mcp` (MCP).** The Agent calls its `search_papers`
  (concurrent multi-source search + dedup across ~22 sources: arXiv, PubMed, bioRxiv, medRxiv,
  OpenAlex, Crossref, CORE, Europe PMC, dblp, Zenodo, HAL, IACR, …) and `download_with_fallback`
  (OA full text via a source-native → OpenAIRE/CORE/PMC → Unpaywall fallback chain). It is
  registered in Qoder over `uvx` (stdio); setup + provenance live in `scripts/paper_search_mcp/`.
- **Fallback + differentiated — the `research` CLI.** Use `research search` when the MCP is
  unavailable, and **always** use the CLI for what the MCP cannot do: the source-output →
  **frontmatter normalization contract**, **WoS enrichment** (`--enrich`; WoS/Scopus are
  unimplemented upstream), and the **arXiv LaTeX-source** path `read` relies on for
  equation-faithful extraction. Its own search fuses the in-house `openalex_client` (API-key
  injected, abstract reconstruction) + `arxiv_client`.

The separately-registered **`arxiv-mcp-server`** is **kept as complementary** — its deep arXiv tools
(`get_paper_latex`, `read_paper_section`, `semantic_search`, `citation_graph`, `watch_topic`) are
unique to it. Paywalled full text still goes through the CLI's `browser_fetch` (institutional-IP
Playwright + Cloudflare); no community MCP replaces that.

## Library layer (Zotero: MCP primary + CLI fallback)

Reference-library work runs on two tiers too, same architecture:

- **Primary — community `zotero-mcp` (MCP).** The Agent reads/writes the user's Zotero directly:
  search (title/author/tag/collection/full text), read metadata / BibTeX / full text / **PDF page
  images** (image evidence when equation/figure/table text extraction is lossy), add items by
  DOI/URL/ISBN/BibTeX/file, merge duplicates, and **Scite retraction alerts**. Registered in Qoder
  as the `zotero` server (`command=zotero-mcp`, local mode); setup + provenance live in
  `scripts/zotero_mcp/`. Unlike the stateless search MCP, it must be installed **persistently**
  (`uv tool install "zotero-mcp-server[pdf,scite]"`) because the CLI shells out to its `zotero-cli`.
- **Fallback + differentiated — the `research` CLI (`add`/`library`) and `document_writing`'s
  `tex refs` export.** They delegate the actual Zotero I/O to `zotero-cli --json` and own what the
  MCP doesn't: the **frontmatter → BibTeX normalization** (custom enrichment fields folded into the
  entry) and the **`zotero_key`/`zotero_uri` write-back** into paper notes. When `zotero-cli` isn't
  installed they degrade gracefully (`add` → skeleton-only note; `library`/`refs` → clear notice).

## Citation integrity gate (OpenAlex + Crossref + arXiv)

Before anything is written to the library, every citation is cross-checked against **three
independent open scholarly sources** — OpenAlex, **Crossref** (free REST, no key), and arXiv —
deliberately bypassing Semantic Scholar. This borrows the *idea* of a citation-audit step (no external
skill body is vendored in); the engine lives in `citation_verify.py`.

- **Auto-gate on `add`.** `research add` verifies the fused frontmatter before creating the Zotero
  item / writing the note. A **FAIL** (≥2 reachable sources hard-conflict on title / DOI / year)
  blocks the write; `--force` overrides, `--no-verify` skips. A verification error degrades
  gracefully (never blocks the main flow).
- **Standalone `citecheck`.** Verify bare DOIs / arXiv ids / OpenAlex ids / titles, or audit existing
  notes (`--note <path>`, `--all` over `papers/`). The exit code is CI-friendly: **1 if any FAIL**,
  else 0.
- **Severity model.** Only a **hard** conflict (title mismatch, DOI mismatch, year off by ≥2) is a
  FAIL. Soft signals — author-surname mismatch, year off by 1 (online-first vs issue year),
  journal-name-only difference — are a **WARN** and never block. All three sources missing the record
  is **NOT_FOUND** (suspicious, flagged for human review, not blocked). An unreachable source only
  downgrades; it is never treated as a conflict.

## Semantic RAG layer (search your own extracted corpus)

Everything `read`/`ingest` extracts lands in `cache/extracted/**/*.md`. The `rag` commands build a
**local embedding index** over that corpus (PaperQA2 + SiliconFlow `bge-m3`) so you can retrieve
across *your own* library **by meaning** — not just discover external papers via `search`.

- **`research rag index`** — build/refresh the index. Incremental (unchanged files skipped, changed
  files flagged stale, new files embedded); persisted as a pickle + JSON meta under `cache/rag/`.
  **Embedding-only, free** (no LLM). Re-run it after `read`/`ingest` adds new full text.
- **`research rag search '<query>'`** — the agent's **main** tool here: pure embedding retrieval that
  returns the top-k most relevant chunks **with their source citation + file path**. No LLM, no cost,
  no blocking. Read the returned chunks and synthesize across papers yourself.
- **`research rag ask '<query>'`** — *optional* one-shot summary convenience. Tries the **free** LLM
  (`Qwen2.5-7B`) first, falls back smoothly to a paid model (`Qwen2.5-32B`) if the free tier is
  unavailable, and **never blocks**: if every LLM path fails it silently degrades to returning the
  `rag search` results. Since you are already a strong LLM, prefer `rag search` + your own synthesis;
  use `ask` only when a quick canned answer is wanted.
- **`research rag status`** — backend readiness (`SILICONFLOW_API_KEY`) + whether a local index
  exists (doc/chunk counts, embedding model, built-at).

Requires `SILICONFLOW_API_KEY` (+ `SILICONFLOW_BASE_URL`); model/home overrides are the optional
`PQA_*` vars. Title/year/DOI/first-author → citation is derived from each MinerU `.md` head, so
retrieved chunks carry a human-readable source. Indexing bypasses Semantic Scholar / Crossref
enrichment (`use_doc_details=False`) — it stays free-embedding + offline-metadata only. See
[references/search.md](references/search.md) for flags and [references/maintenance.md](references/maintenance.md) §5d for internals.

## Invocation

Run from the **project root**. The skill installs a console script `pysci-research`:

```
uv run pysci-research <command> [options]
```

Below, `research …` is shorthand for `uv run pysci-research …`. (Fallback if the script
isn't installed: `uv run python -m pysci.skills.literature_research.tools.research …`.)

> **PowerShell rule (critical):** wrap multi-word arguments in **single quotes**, e.g.
> `research search 'acoustic exceptional point'`. Double quotes get stripped by the shell and
> break the command. Use `;` (never `&&`) to chain commands.

## Commands at a glance

| Command | Use when | Key output |
|---|---|---|
| `doctor` | Session start, or anything seems broken | Config + capability self-check |
| `search "<query>"` | Finding papers on a topic | Ranked, deduped multi-source list |
| `read <doi｜url｜arxiv-id>` | Deep-reading one paper | Full-text Markdown + note skeleton in `papers/` |
| `get <id>` | You only need metadata / citation data | Fused frontmatter (YAML, or `--json`) |
| `add <doi>` | Adding a paper to the library (auto-verifies the citation first) | Zotero item + note skeleton |
| `citecheck <doi｜id｜title>` | Verifying a citation, or auditing notes before/after writing | Three-source verdict (✓PASS / △WARN / ✗FAIL / ?NOT_FOUND) |
| `library <ping｜list｜search｜get>` | Querying the user's Zotero | Items from the local library |
| `index` | After adding or editing notes | Rebuilt `INDEX.md` |
| `ingest` | Bulk-archiving a **local** folder of PDFs (no DOI) | Copied PDFs + extracted Markdown + `ingest/manifest.json` ledger |
| `rag <index｜search｜ask｜status>` | Semantically searching your own extracted corpus (`cache/extracted/`) | Top-k chunks + source citation (embedding-only `search`); optional one-shot `ask` summary |
| `cache <stats｜clean｜prune>` | Disk pressure, or checking what's cached | Two-tier cache stats / cleanup / LRU prune |

Run `research <command> -h` for the full option list of any command.

## Quick start

**Find papers** (fuses OpenAlex + arXiv, deduped):
```
research search 'non-Hermitian acoustic exceptional point' --year 2020-2026 --limit 15 --sort citations
```
Add `--save --purpose "<goal>"` to write a screening snapshot to `shortlists/`.

**Read one paper** (full text + note skeleton):
```
research read 10.1103/PhysRevLett.121.124501
```
This resolves the DOI to the publisher, downloads the PDF, extracts it to Markdown
(equations → LaTeX via MinerU cloud), and writes a structured note to `papers/`. It prints the
path of the extracted full text — **Read that file to actually read the paper**, then fill in the
note's evaluation sections (rubric in [references/read.md](references/read.md)).
Paywalled or Cloudflare-blocked? Add `--headed`.

Re-running `read` on the same paper **reuses the cached full text** (skips fetch + extraction) —
add `--force` to re-fetch and re-extract. Expensive artifacts (PDFs, extracted Markdown, fetched
HTML) are kept permanently; manage disk with `research cache stats｜clean｜prune`.

## Standard workflow: topic → reviewed notes

```
- [ ] 1. research search '<topic>' --save --purpose '<goal>'   # → shortlists/
- [ ] 2. Screen the shortlist; pick the keepers
- [ ] 3. research add <doi>          # for each keeper → citation gate → Zotero + note skeleton
- [ ] 4. research read <doi>         # for the ones to read closely → full text
- [ ] 5. Fill each note's TLDR / Key Claims / Novelty / Rigor / Relevance (see references/read.md)
- [ ] 6. research index              # refresh INDEX.md
- [ ] 7. (optional) Synthesize a survey in reviews/
```

`add` runs the **citation integrity gate** automatically (step 3). To audit notes you already have —
or to re-check a batch of DOIs before citing them in a survey — run `research citecheck --all` (or
`citecheck <doi> …`); see **Citation integrity gate** above and [references/read.md](references/read.md).

## Bulk-ingesting a local PDF repository

When the user already has a folder of PDFs (no DOIs to resolve), use `ingest` instead of
`read`/`add`. It is driven by a curated, **git-tracked** ledger `ingest/manifest.json` that records
each file's `theme / slug / type / priority / pages / source / status`:

```
research ingest --status                     # progress summary (done / pending / failed, by priority & theme)
research ingest --priority 1 --dry-run       # preview a batch, no extraction
research ingest --priority 1                 # run it: copy PDF → MinerU extract → cache/extracted/<theme>/<slug>.md
research ingest --priority 1 --limit-pages 800   # cap pages this run (respect MinerU daily quota)
```

Behaviour: PDFs are **copied** (originals untouched) to `cache/pdfs/<theme>/<slug>.pdf`; Markdown is
extracted to `cache/extracted/<theme>/<slug>.md`. It is **resumable** — the manifest is rewritten
after every file, `done` entries are skipped on re-run, `failed` entries can be retried. Use
`priority` to batch by cost (1 = short papers, 2 = reviews/theses, 3 = big textbooks) so you stay
within MinerU's daily page quota and can continue on a later day. Author/edit `manifest.json` by
hand to (re)classify; the `source` paths must match the real files.

## Data sources & expectations

These are the sources the `research` CLI queries **directly** (the fallback + differentiated path).
The `paper-search-mcp` MCP aggregates ~22 sources on the primary path — see **Retrieval layers** above.

- **OpenAlex** — primary source, rich metadata + citation counts. Since 2026-02-13 an API key
  (`OPENALEX_API_KEY`) is required for full quota; without it, requests are capped at ~100
  credits/day (testing only).
- **arXiv** — preprints, no key. Use for preprint-only or latest work.
- **Crossref** — free REST (no key; `mailto` polite pool), queried by the **citation integrity gate**
  (`citecheck` / `add`), not by `search`. It is the second independent source (alongside OpenAlex +
  arXiv) that cross-checks title / DOI / year before a citation is written. See **Citation integrity
  gate** above.
- **Web of Science** (Starter API) — *optional* enrichment: adds the WoS accession no. (`wos_id`)
  + authoritative Times Cited, and a JCR URL. It carries **no official JIF / JCR quartile / ESI**
  (the Starter API doesn't expose those), so those frontmatter fields stay OpenAlex-derived. Degrades
  silently when unavailable; a one-line notice is printed, no action needed.
- **MinerU cloud** (via `mineru-open-sdk`) — primary PDF → Markdown backend (equations → LaTeX,
  tables → HTML). Needs `MINERU_TOKEN`. `pymupdf4llm` is the local fallback (fast, but equations
  are lost). Paywalled HTML full text is body-extracted with `trafilatura`.
- **SiliconFlow** (OpenAI-compatible) — powers the local **RAG layer** (`rag index`/`search`/`ask`):
  `bge-m3` embeddings (free) for the semantic index + retrieval, plus an *optional* Qwen chat model
  for `rag ask`. Needs `SILICONFLOW_API_KEY`. Embedding-only retrieval is free; the `ask` LLM tries a
  free tier first, falls back to paid, and never blocks (degrades to `search`).

## Output locations

| Path | Contents |
|---|---|
| `data/skills/literature_research/papers/` | One structured note per paper (`{year}_{author}_{slug}.md`) |
| `data/skills/literature_research/shortlists/` | One search snapshot per query |
| `data/skills/literature_research/reviews/` | Multi-paper surveys |
| `data/skills/literature_research/INDEX.md` | Auto-generated library index (`research index`) |
| `data/skills/literature_research/ingest/manifest.json` | Ledger for bulk local-PDF ingestion (`research ingest`); git-tracked |
| `data/skills/literature_research/cache/` | Downloaded/copied PDFs + fetched HTML (git-ignored) **+ extracted full text under `cache/extracted/`, which IS git-tracked** (MinerU-quota-expensive `.md`, worth backing up) |
| `data/skills/literature_research/cache/rag/` | Local PaperQA2 embedding index (`index.pkl` + `index_meta.json`) behind `rag search`/`ask` (git-ignored; rebuildable with `rag index`) |

## When something breaks

1. Run `research doctor` — it reports every source, backend, and Zotero status.
2. For browser-fetch / MinerU / adapter / config issues, see
   [references/maintenance.md](references/maintenance.md).

## Reference files

- [references/search.md](references/search.md) — `search` / `get` / `library` / `rag` details: all
  flags, source selection, metrics, screening, shortlist schema, and the local semantic-RAG layer.
- [references/read.md](references/read.md) — `read` / `add` / `citecheck` / `index` details: full-text
  extraction, the paper-note schema, the **citation integrity gate**, and the **evaluation rubric**
  for judging a paper.
- [references/maintenance.md](references/maintenance.md) — how the backend works and how to fix or
  extend it (publisher adapters, PDF backends, config, common failures).
