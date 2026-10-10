# search 分册 2/5

> 含小节：`search` — multi-source retrieval (CLI)（续）· `--json` output schema；`search` — multi-source retrieval (CLI)（续）· Metrics glossary；`search --save` — the shortlist snapshot；`get` — fus
> 原 `search.md` 超 8KB 预算拆分（2026-10-10，backlog 20261010-refs-split-8kb）；分册索引见同目录 `search.md`，按需只读所需分册。

### `--json` output schema

`search --json` prints a **bare JSON array**, one object per row, in the order the human render shows.
Every key is always present (absent data is `""` or `null`, never a missing key). `citegraph --json`
reuses this exact row shape, nested under `backward` / `forward`.

| Key | Type | Notes |
|---|---|---|
| `title` | string | |
| `first_author_last_name` | string | `""` when unknown |
| `year` | int｜null | |
| `journal` | string | arXiv rows: the venue parsed from `journal_ref`, else `"arXiv"` |
| `doi`, `arxiv_id`, `openalex_id` | string | `""` when absent |
| `cited_by_count` | int｜null | `null` for arXiv-only rows (arXiv carries no citation data) |
| `oa_status` | string | `gold`/`green`/`bronze`/`hybrid`/`closed`; arXiv rows default to `green` |
| `jif` | float｜null | OpenAlex `2yr_mean_citedness` **estimate** (see glossary) |
| `jcr_quartile` | string | Always `""` until the WoS Journals API is wired up |
| `journal_tier` | string | `top`/`leading`/`basic`/`""` — derived (see glossary) |
| `source` | string | `openalex`｜`arxiv`｜`wos` — which connector produced the row |

Two contract details worth relying on:

- An empty result set is a legal `[]`, never a human "（无结果）" string — so a caller can always
  distinguish "this topic has no literature" from "my parser broke".
- When OpenAlex is running **key-less** (no `OPENALEX_API_KEY`) *and* the result set is empty,
  `search` prints one extra warning to **stderr**. This is the single deliberate exception to the
  module's otherwise-silent degradation: an empty list is indistinguishable from an exhausted
  $0.10/day usage budget, and quietly returning `[]` there is exactly how a survey ends up asserting
  that a field is empty. It fires only when the OpenAlex branch actually ran — `--source arxiv` never
  queried OpenAlex, so warning about its quota would be misleading.

### Metrics glossary

- **被引 / `cited_by_count`** — total citations (OpenAlex). Compare within a field + age.
- **`cited_by_count_normalized`** — OpenAlex percentile-year (0–100); >90 is highly cited for its
  age/field. More fair than raw counts across years.
- **`jif`** — Journal Impact Factor. Only OpenAlex's 2-year mean citedness **estimate** is available;
  the WoS Starter API returns no official JIF (its `/journals` gives a JCR URL to look one up). Read
  it as *ordinal*: it tracks reasonably in the mid range (3–9) and can understate top journals and
  low-citation-density fields by **2–3×**.
- **`journal_tier`** (+ **`journal_tier_basis`**) — `top` / `leading` / `basic` / `""`, derived from
  OpenAlex's `listed_in` by `notes.derive_journal_tier()`, taking the highest grade across three
  **expert-panel** lists: JUFO (3→top, 2→leading, 1→basic), Norway (2→top, 1→basic), KI-JL
  (3→top, 2→leading, 1→basic). Costs **zero extra requests** — the work/source record already carries
  `listed_in`. Because these panels grade *per discipline*, the tier stays meaningful where JIF does
  not: **JASA** has a 2-year mean citedness of only **0.82** yet is rated top-tier (JUFO-3), the same
  grade as Nature and PRL. `journal_tier_basis` keeps the exact tags it came from
  (`["jufo-3", "norway-2"]`) so the grade is auditable rather than asserted. Tags carrying no grade
  (`cwts-core`, `medline`, `erih-plus`, `doyens`) are ignored — a journal listed only in those gets
  `""`, not a fabricated tier.
- **`scimago_quartile`** — a real `Q1`…`Q4`, from a local index built off the official **SCImago SJR**
  CSV (Scopus-based). Needs a one-time `research journal build-scimago --csv …`; matched **by ISSN
  exactly** (no title fuzzy-matching), so it stays `""` when the venue's metadata has no ISSN or the
  index isn't built. See [`journal`](#journal--venue-quality-from-the-free-layers) below.
- **`jcr_quartile`** — the **official** JCR quartile. Not supplied by any current source (the WoS
  Starter API has no quartile data), so it stays `""`. `scimago_quartile` and `journal_tier` are its
  two free stand-ins; all three measure different things and coexist without conflicting.
- **`oa_status`** — `gold` (OA journal) / `green` (repository preprint) / `bronze` (free on
  publisher site, no explicit license) / `hybrid` / `closed`.
- **`esi_highly_cited` / `esi_hot_paper`** — top 1% / top 0.1%. Only the WoS Journals API can answer
  this, so both stay **`null`** — deliberately *not* `false`. `false` would assert "this paper is not
  an ESI highly-cited paper", which is a lie in the data for one that is; `null` says "unknown".
- **`journal_h_index`** — the venue's h-index (OpenAlex). A coarse but honest signal that, unlike
  `jif`, is not skewed by a handful of mega-cited reviews.

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
