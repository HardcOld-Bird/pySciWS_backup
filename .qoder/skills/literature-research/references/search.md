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
upstream), so it is subject to OpenAlex's ~100 credits/day cap — the CLI's `openalex_client` injects
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
  ~100 credits/day quota, and quietly returning `[]` there is exactly how a survey ends up asserting
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

## `citegraph` — snowball the citation graph

```
research citegraph <doi｜arxiv-id｜openalex-id>
                   [--direction backward｜forward｜both] [--limit N] [--year Y]
                   [--sort relevance｜date｜citations] [--save] [--purpose '<goal>'] [--json]
```

Rolls the citation graph around one seed paper through **OpenAlex**: `backward` = what it cites (its
reference list), `forward` = what cites it. This is the snowballing step — run it on a seed you trust,
then feed the interesting hits back into `get` / `read` / `add`.

| Flag | Default | Effect |
|---|---|---|
| `--direction` | `both` | `backward` needs only the reference list; `forward` needs a real OpenAlex id. |
| `--limit` | `25` | **Per direction.** `0` = unlimited. |
| `--year` | — | **forward only** — `2020-2026` / `2020-` / `2020`. |
| `--sort` | `citations` | `relevance`｜`date`｜`citations`. Defaults to citation-descending so the first screenful is the most influential, not the most recent. |
| `--save` | — | Write a snapshot to `shortlists/` (same template as `search --save`), so a snowball round is as reproducible as a query. |
| `--purpose` | — | Recorded in the snapshot. |
| `--json` | — | Emit both directions **separately** (shape below). |

Behavior worth knowing before you rely on it:

- **`forward` requires an OpenAlex id.** An arXiv id or DOI is first resolved through OpenAlex; if
  that fails, `--direction forward` exits **2** with the reason, while `both` degrades to
  backward-only and keeps going. Citation graphs exist only in OpenAlex — there is no fallback source.
- **Sorting happens where it is cheapest.** `forward` is sorted *and* truncated **server-side**
  (OpenAlex `sort=`), so `--limit 25` fetches 25 records, not 200. `backward` comes back from a bulk
  id lookup in no guaranteed order, so it is fetched in full, sorted **client-side**, then truncated
  — otherwise `--sort citations --limit 20` would silently mean "the best 20 of an arbitrary first 50".
- **`both` de-duplicates across directions.** A paper cannot both cite and be cited by the seed, but
  OpenAlex's citation data carries a few dirty bidirectional records. Overlaps are dropped using the
  same key as `search` (DOI → arXiv id → title slug); `backward` is inserted first, so it wins.
- **Progress lines go to stderr** (`[citegraph] backward：参考文献 42 条，OpenAlex 取回 39 条…`), keeping
  stdout clean for `--json`. A shortfall is normal — those are merged or unindexed records — and is
  reported, not treated as an error.
- **Exit codes:** `2` when the seed cannot be resolved to an OpenAlex record at all, or `forward` was
  requested without an id; `0` otherwise, *including* "both directions empty". A network failure in
  the graph layer degrades to an empty list plus one line, never a traceback.

`--json` shape — the two directions are **not** merged into one array, because whether a hit *cites
the seed* or *is cited by it* changes what you do with it next:

```json
{
  "id": "<as given>", "openalex_id": "W…", "title": "…",
  "year": 2018, "cited_by_count": 230,
  "backward": {"referenced_works_count": 42, "resolved_count": 39, "rows": []},
  "forward":  {"total_citing": 187, "returned_count": 25, "rows": []}
}
```

Each `rows` entry is the same 13-key object `search --json` emits (all tagged `source: "openalex"`).
`total_citing` is the server's own count, so `returned_count < total_citing` tells you to raise
`--limit` rather than concluding the forward set is small.

---

## `journal` — venue quality from the free layers

```
research journal lookup <issn> [--json]
research journal build-scimago --csv <path> [--year 2024]
research journal status [--json]
```

Answers "how good is this venue?" **with no paid credential**, by putting the free layers side by
side. The official JIF / JCR quartile / JCI / ESI can only come from the **WoS Journals API** (applied
for, not yet granted; the WoS *Starter* API does not carry them, and Elsevier/Scopus need paid keys),
so `lookup` prints four numbered sections and is explicit about the fourth:

1. **SCImago SJR** — quartile / SJR value / h-index / index version year, from the local index, matched **by ISSN exactly**.
2. **OpenAlex `listed_in` → `journal_tier`** — the expert-panel grade, the exact tags it was derived from, and the full `listed_in` list. Zero extra requests.
3. **OpenAlex citation metrics** — `2yr_mean_citedness` (the `jif` estimate) and the journal `h_index`.
4. **Official JCR** — `jcr_quartile` / official JIF / JCI / ESI: printed as *未接入* rather than left blank, so an unavailable number is never mistaken for a measured zero.

`lookup` also states **why** a layer is empty, because the two causes call for different fixes:
*索引未建* → run `build-scimago`; *该 ISSN 不在索引里* → the venue genuinely has no SJR record. Same
for OpenAlex: *未收录该 ISSN，或网络不可达*. The `jif` value is rendered at **2 decimals** in the human
output to match what a note's frontmatter stores (`round(x, 2)`); `--json` keeps full precision, since
that output is for programs and dropping digits there would be a loss with no upside.

`lookup --json` keys: `issn`, `issn_normalized`, `scimago` (`{issn, quartile, sjr, h_index,
sjr_year}` or `null`), `openalex` (the raw source fields — `display_name`, `issn`, `issn_l`,
`publisher`, `country_code`, `h_index`, `works_count`, `2yr_mean_citedness`, `listed_in`), `journal`,
`jif`, `journal_h_index`, `listed_in`, `journal_tier`, `journal_tier_basis`. A missing ISSN exits **2**.

### `build-scimago` — one-time local index

The official CSV must be **downloaded by hand** from `https://www.scimagojr.com/journalrank.php` (the
site returns 403 to programmatic requests and needs form/JS interaction, so no auto-fetch is
implemented):

```
research journal build-scimago --csv <path-to-csv> [--year 2024]
```

Only `Issn` / `SJR` / `SJR Best Quartile` / `H index` are kept — enough for quartile judgment, and it
compresses a ~15 MB CSV into a ~1.4 MB JSON that git can carry. Output:
`data/skills/literature_research/scimago_index.json` (**tracked**); `*.csv` there is git-ignored, so
don't leave the raw download in the data dir. Multi-valued `Issn` cells (`0031-9007;1079-7114`)
produce **one key per ISSN** pointing at the same record. The version year is inferred from the
header (`Total Docs. (2024)`) or the file name, and `--year` overrides it — it is never invented.
ISSNs are normalized by stripping hyphens, so `0031-9007` and `00319007` both match.

> **This one path is deliberately *not* silent.** Everywhere else in the module a missing input
> degrades quietly, but a failed build that quietly wrote an empty index would leave
> `scimago_quartile` blank in *every* note with no visible cause — far harder to diagnose than an
> error. So: missing `--csv` → exit **2**; unreadable, empty, or column-less CSV → exit **1** with the
> reason on stderr. After a successful build, record the download URL / version year / download date
> in `data/skills/literature_research/SOURCE.md` (the attribution line is printed for you), and
> refresh yearly — see `references/maintenance.md`.

### `status`

Reports `exists` / `readable` / `path` / `sjr_year` / `n_entries` / `built_at` / `source_csv` /
`attribution` / `download_url` (also as `--json`). Exit **0** when the index is simply absent — that
is a supported state: `scimago_quartile` stays empty while `journal_tier` and `jif` are unaffected —
and exit **1** when the file exists but cannot be parsed (rebuild it). `research doctor` prints the
same summary under its journal-metrics section.

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
- `list` — recent items, optionally filtered by `itemType`. **Best-effort, not a real enumeration**:
  `zotero-cli` has no "list everything" command, so this degrades to an *empty-keyword* search capped
  at 100 items upstream. It says so on stderr every time, and if it returns 0 items it says that too
  — a local library that demonstrably has papers can legitimately yield nothing here, and printing
  "（无条目）" would read as "your library is empty", which is simply false. **When you need to know
  whether a specific paper is in the library, use `search --query`, not `list`.**
- `search` — metadata search (title/author/tag). Full-text search needs the local backend.
- `get` — one item's full JSON by key.

`list` and `search` failures (Zotero not running, local API not authorized — both ordinary states,
not exceptions) degrade to one stderr line + exit **1**; `search` without `--query` and `get` without
`--key` exit **2**.

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
