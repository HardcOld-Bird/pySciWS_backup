# search 分册 1/5

> 含小节：概览；Retrieval: MCP primary, CLI fallback；`search` — multi-source retrieval (CLI)；`search` — multi-source retrieval (CLI)（续）· Source behavior；`search` — multi-source retrieval (CLI)（
> 原 `search.md` 超 8KB 预算拆分（2026-10-10，backlog 20261010-refs-split-8kb）；分册索引见同目录 `search.md`，按需只读所需分册。

# search / get / citegraph / journal / library / rag — detailed reference

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
upstream), so it is subject to OpenAlex's keyless **$0.10/day** usage budget — the CLI's `openalex_client` injects
`OPENALEX_API_KEY` and stays the quota-safe OpenAlex path; and the separately-registered
`arxiv-mcp-server` is kept as complementary (deep arXiv tools the MCP lacks). Its `citation_graph` is
the **one overlap** with the CLI, and the split is by *coverage*: it reaches arXiv papers only,
whereas `research citegraph` (below) snowballs through everything OpenAlex indexes — journal
articles, books, datasets — in both directions. Prefer the CLI unless the seed is arXiv-only.

---

## `search` — multi-source retrieval (CLI)

```
research search '<query>' [--source S] [--year Y] [--limit N] [--sort K]
                          [--min-citations M] [--oa-only] [--enrich]
                          [--save --purpose '<goal>'] [--json]
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
| `--json` | flag | Print the rows as a JSON array (schema below) instead of the aligned human text. **stdout carries only the JSON** — progress and diagnostics go to stderr. |

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
     DOI:10.1103/physrevlett.121.124501 | arXiv:1803.04110 | W2789790776  [JIF 8.97 tier:top]
```
Line 1 = `[year] citations  oa_status  journal  (source)`; line 2 = first author + title;
line 3 = DOI / arXiv id / OpenAlex id, plus a trailing `[…]` bracket when the venue has metrics —
`JIF <n>` and then either the official quartile (`Q1`) or, while that stays unavailable,
`tier:<journal_tier>`. Feed any of these ids to `get`, `read`, `add`, or `citegraph`.

For an arXiv-only row the journal column is the venue name **parsed out of** `journal_ref`
(`Phys. Rev. Lett.`, not `Phys. Rev. Lett. 121, 124501 (2018)`), and falls back to `arXiv` when the
preprint has no journal-ref at all. Citation count and `JIF` are absent for such rows.
