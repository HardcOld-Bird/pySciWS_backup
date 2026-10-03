# search / get / library / rag — detailed reference

All commands are invoked as `research <cmd> …` (see SKILL.md for the full invocation and the
PowerShell single-quote rule).

---

## Retrieval: MCP primary, CLI fallback

`research search` (below) is the **CLI** retrieval path — the fallback plus the differentiated layer.
The **primary** multi-source path is the community **`paper-search-mcp`** MCP, which the Agent drives
directly (registered in Qoder over `uvx`; setup + provenance in `scripts/paper_search_mcp/`):

- `search_papers(query, sources=[…])` — concurrent search + dedup across ~22 sources (arXiv,
  PubMed, bioRxiv, medRxiv, OpenAlex, Crossref, CORE, Europe PMC, dblp, Zenodo, HAL, IACR, …),
  returning a standardized `Paper` list.
- `download_with_fallback(…)` — OA full text via a source-native → OpenAIRE/CORE/PMC → Unpaywall
  fallback chain.

Reach for the MCP first for broad discovery. Use `research search` when the MCP is unavailable, or
when you need what only the CLI provides: the **frontmatter normalization contract**, **WoS
enrichment** (`--enrich`; WoS/Scopus are unimplemented in the MCP), and the **arXiv LaTeX-source**
path behind `read`. Two caveats: the MCP's OpenAlex connector runs **key-less** (no API-key env var
upstream), so it is subject to OpenAlex's ~100 credits/day cap — the CLI's `openalex_client` injects
`OPENALEX_API_KEY` and stays the quota-safe OpenAlex path; and the separately-registered
`arxiv-mcp-server` is kept as complementary (deep arXiv tools the MCP lacks).

---

## `search` — multi-source retrieval (CLI)

```
research search '<query>' [--source S] [--year Y] [--limit N] [--sort K]
                          [--min-citations M] [--oa-only] [--enrich]
                          [--save --purpose '<goal>']
```

| Flag | Values / example | Effect |
|---|---|---|
| `--source` | `auto` (default), `openalex`, `arxiv`, `wos` | Which source(s) to query. |
| `--year` | `2020-2026`, `2023-`, `2024` | Publication-year filter. |
| `--limit` | integer (default 15) | Max results per source; `auto` truncates the merged list to this. |
| `--sort` | `relevance` (default), `date`, `citations` | Ranking. |
| `--min-citations` | integer | Drop works cited fewer times (OpenAlex). |
| `--oa-only` | flag | Only open-access works. |
| `--enrich` | flag | Per-result WoS lookup — attaches the WoS accession no. (`wos_id`) + authoritative Times Cited. **Slower** (one call per item); off by default. |
| `--save` | flag | Write a screening snapshot to `shortlists/{date}_{slug}.md`. |
| `--purpose` | text | The goal recorded in the snapshot. |

### Source behavior

- **`auto` (default)** — queries **OpenAlex + arXiv**, merges and **dedupes** by DOI → arXiv id →
  title. OpenAlex records are preferred (richer: citations, JIF estimate, OA link). This is fast
  and needs no keys. Use this unless you have a reason not to.
- **`openalex`** — OpenAlex only. Best metadata + citation counts.
- **`arxiv`** — arXiv only. Best for preprints / very recent work. Note: arXiv records have **no
  citation count or JIF**.
- **`wos`** — Web of Science (Starter API) as the primary source. Requires `WOS_API_KEY`; errors
  out if unset. Returns authoritative **Times Cited** + the WoS accession no., but **no JIF /
  quartile / ESI** (the Starter API does not carry those — its `/journals` gives a JCR URL to look JIF up).

### `--sort` mapping

| `--sort` | OpenAlex | arXiv |
|---|---|---|
| `relevance` | `relevance_score:desc` | `relevance` |
| `date` | `publication_date:desc` | `submittedDate` |
| `citations` | `cited_by_count:desc` | `relevance` (arXiv has no citation sort) |

### `--enrich` (opt-in deep fusion)

By default `search` does **not** call WoS per result (that would be N slow network calls).
With `--enrich`, each OpenAlex result that has a DOI is looked up in WoS to attach its **WoS
accession number** (`wos_id`). The WoS **Starter API returns no JIF / JCR quartile / ESI** — those
frontmatter fields keep their OpenAlex-derived values (JIF estimate) or stay empty. Enrichment
failures degrade silently; at most one summary line is printed. Single-paper commands
(`get`, `read`, `add`) **always** enrich, since one call is cheap.

### Reading the output

```
 1. [2018] 被引  230  green   Physical Review Letters  (openalex)
     zhu—— Simultaneous Observation of a Topological Edge State and Exceptional Point …
     DOI:10.1103/physrevlett.121.124501 | arXiv:1803.04110 | W2789790776
```
Line 1 = `[year] citations  oa_status  journal  (source)`; line 2 = first author + title;
line 3 = DOI / arXiv id / OpenAlex id. Feed any of these ids to
`get`, `read`, or `add`.

### Metrics glossary

- **被引 / `cited_by_count`** — total citations (OpenAlex). Compare within a field + age.
- **`cited_by_count_normalized`** — OpenAlex percentile-year (0–100); >90 is highly cited for its
  age/field. More fair than raw counts across years.
- **JIF** — Journal Impact Factor. Only OpenAlex's 2-year mean citedness **estimate** is available;
  the WoS Starter API returns no official JIF (its `/journals` gives a JCR URL to look one up).
- **JCR quartile** — `Q1`…`Q4`. Not supplied by any current source (the WoS Starter API has no
  quartile data), so it stays empty.
- **`oa_status`** — `gold` (OA journal) / `green` (repository preprint) / `bronze` (free on
  publisher site, no explicit license) / `hybrid` / `closed`.
- **ESI highly-cited / hot paper** — top 1% / top 0.1%. Not supplied by the WoS Starter API; these flags stay `false`.

---

## `search --save` — the shortlist snapshot

Writes `shortlists/{YYYY-MM-DD}_{query-slug}.md` from `templates/shortlist.md`. The frontmatter
records the reproducible query (date, sources, query string, filters, sort, counts). The body
contains the screening-criteria checklist plus an auto-generated **candidate list** (all results
with authors, journal/year, citations, OA, ids). Typical use: generate it, then edit it to sort
candidates into *Highly Recommended / Worth Following / Excluded* and record meta-observations
(field heat, leading groups, methodological trends, gaps).

---

## `get` — fused metadata for one paper

```
research get <doi｜arxiv-id｜openalex-id> [--json]
```
Resolves the id, **always enriches** (attaches the WoS accession no. `wos_id` when the DOI is indexed in WoS), and
prints the full `paper_note` frontmatter as YAML (or JSON with `--json`). Use it to check a
paper's venue quality and citation standing before deciding to read or add it. No files are
written.

> **Verifying a citation before you write it?** `get` shows one paper's fused metadata; to
> cross-check a DOI / title against **OpenAlex + Crossref + arXiv** (fabricated or mis-paired
> reference detection) use `research citecheck` — the same three-source gate `add` runs
> automatically. Crossref here is a *free, key-less* REST source queried directly by the CLI (not one
> of the `search` sources). See the `citecheck` section in [read.md](read.md).

---

## `library` — query the user's Zotero

These subcommands are a thin differentiated facade over the community **zotero-mcp**'s
`zotero-cli --json`; they exit with a clear install notice when `zotero-cli` isn't on PATH. For
deeper library work — reading full text / **PDF page images**, merging duplicates, Scite
retraction alerts — drive the registered `zotero` MCP directly.

```
research library ping
research library list [--limit N] [--type journalArticle]
research library search --query '<terms>' [--limit N]
research library get --key <ITEMKEY>
```
- `ping` — connectivity + backend (local `zotero.sqlite` preferred; falls back to Web API).
- `list` — recent items, optionally filtered by `itemType`.
- `search` — metadata search (title/author/tag). Full-text search needs the local backend.
- `get` — one item's full JSON by key.

Use `library` to check whether a paper is **already in the user's Zotero** before `add`-ing it
(avoids duplicates), and to find the `zotero_key` that links a note to its library entry.

---

## `rag` — semantic retrieval over your own extracted corpus (PaperQA2)

Where `search` discovers **external** papers, `rag` queries the corpus you have **already** extracted
to `cache/extracted/**/*.md` (via `read`/`ingest`). It builds a local embedding index (SiliconFlow
`bge-m3`, free) and retrieves **by meaning**. Needs `SILICONFLOW_API_KEY` (+ `SILICONFLOW_BASE_URL`).

```
research rag index  [--path P …] [--rebuild]
research rag search '<query>' [-k N] [--chars N] [--json]
research rag ask    '<query>' [-k N] [--chars N] [--json]
research rag status [--json]
```

| Subcommand | Effect |
|---|---|
| `index` | Build/refresh the embedding index over `cache/extracted/**/*.md` (or explicit `--path` files/dirs). **Incremental**: unchanged files skipped, changed files flagged stale, new files embedded; `--rebuild` forces from-scratch. Persisted to `cache/rag/index.pkl` + `index_meta.json`. Embedding-only (no LLM, free). |
| `search` | **The agent's main tool.** Pure embedding retrieval — top-`k` (default 8) most relevant chunks, each with **rank, text, source citation, file path**. No LLM, no cost, never blocks. Read the chunks and synthesize across papers yourself. |
| `ask` | *Optional* one-shot summary. PaperQA2 answer over the retrieved evidence: **free** `Qwen2.5-7B` first → smooth fallback to paid `Qwen2.5-32B` if the free tier is unavailable → if every LLM path fails, **silently degrades** to the `search` results (never raises). Prefer `search` + your own synthesis. |
| `status` | Backend readiness (`SILICONFLOW_API_KEY`) + local index state (exists, doc/chunk counts, embedding model, built-at, paperqa version). Does not import the backend or touch the network. |

Flags: `-k/--top-k` (chunks to retrieve, default 8), `--chars` (per-chunk truncation in the human
render, default 400), `--json` (machine-readable result), `--path` (index only these files/dirs instead
of the whole `cache/extracted/`), `--rebuild` (ignore the existing index).

### Reading `search` output

```
[1] Xia et al. (2025)  —  cpa_ep__paper_prl  (…/cache/extracted/cpa_ep/paper_prl.md)
    …chunk text (truncated to --chars)…
```
Each hit = `rank`, `citation` (derived from the MinerU `.md` head: first author + year, else title),
`docname` (the `__`-joined relative-path stem), and `source_path`. Retrieval is MMR-ranked (diversity),
so hits come back in relevance order without a numeric score. `--json` emits
`{query, chunks:[{rank,text,docname,citation,source_path,chunk_name}], n_docs, n_chunks, index_path,
embedding_model}`.

### Notes

- **Metadata is derived offline** from each `.md` head (`# title`, author line, `published/received`
  year, `DOI:`) — no Semantic Scholar / Crossref enrichment (`use_doc_details=False`), so indexing pays
  no per-doc network cost and stays S2-free.
- **Index lifetime**: the pickle is version-stamped; a paperqa-version mismatch or a corrupt file is
  treated as "no index" (rebuild). `cache/rag/` is git-ignored and fully rebuildable with `rag index`.
- **When `search` reports "no index"**, run `research rag index` first. **When `ask` degrades**, it
  prints the reason and returns `search` results instead — by design (never block, never noise).
