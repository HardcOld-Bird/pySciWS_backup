# search / get / library — detailed reference

All commands are invoked as `research <cmd> …` (see SKILL.md for the full invocation and the
PowerShell single-quote rule).

---

## `search` — multi-source retrieval

```
research search '<query>' [--source S] [--year Y] [--limit N] [--sort K]
                          [--min-citations M] [--oa-only] [--enrich]
                          [--save --purpose '<goal>']
```

| Flag | Values / example | Effect |
|---|---|---|
| `--source` | `auto` (default), `openalex`, `arxiv`, `wos`, `s2` | Which source(s) to query. |
| `--year` | `2020-2026`, `2023-`, `2024` | Publication-year filter. |
| `--limit` | integer (default 15) | Max results per source; `auto` truncates the merged list to this. |
| `--sort` | `relevance` (default), `date`, `citations` | Ranking. |
| `--min-citations` | integer | Drop works cited fewer times (OpenAlex). |
| `--oa-only` | flag | Only open-access works. |
| `--enrich` | flag | Per-result WoS JIF/JCR + S2 TLDR. **Slower** (one call per item); off by default. |
| `--save` | flag | Write a screening snapshot to `shortlists/{date}_{slug}.md`. |
| `--purpose` | text | The goal recorded in the snapshot. |

### Source behavior

- **`auto` (default)** — queries **OpenAlex + arXiv**, merges and **dedupes** by DOI → arXiv id →
  title. OpenAlex records are preferred (richer: citations, JIF estimate, OA link). This is fast
  and needs no keys. Use this unless you have a reason not to.
- **`openalex`** — OpenAlex only. Best metadata + citation counts.
- **`arxiv`** — arXiv only. Best for preprints / very recent work. Note: arXiv records have **no
  citation count or JIF**.
- **`wos`** — Web of Science as the primary source. Requires `WOS_API_KEY`; errors out if unset.
- **`s2`** — Semantic Scholar. **Skipped with a one-line notice when no API key is set.** See the
  S2 policy below.

### `--sort` mapping

| `--sort` | OpenAlex | arXiv |
|---|---|---|
| `relevance` | `relevance_score:desc` | `relevance` |
| `date` | `publication_date:desc` | `submittedDate` |
| `citations` | `cited_by_count:desc` | `relevance` (arXiv has no citation sort) |

### `--enrich` (opt-in deep fusion)

By default `search` does **not** call WoS/S2 per result (that would be N slow network calls).
With `--enrich`, each OpenAlex result that has a DOI is enriched with the official **WoS JIF /
JCR quartile / ESI** flags and, if an S2 key exists, the **TLDR**. Enrichment failures degrade
silently to OpenAlex estimates; at most one summary line is printed. Single-paper commands
(`get`, `read`, `add`) **always** enrich, since one call is cheap.

### Reading the output

```
 1. [2018] 被引  230  green   Physical Review Letters  (openalex)
     zhu—— Simultaneous Observation of a Topological Edge State and Exceptional Point …
     DOI:10.1103/physrevlett.121.124501 | arXiv:1803.04110 | W2789790776
```
Line 1 = `[year] citations  oa_status  journal  (source)`; line 2 = first author + title;
line 3 = DOI / arXiv id / OpenAlex id (and `[JIF … Qn]` when enriched). Feed any of these ids to
`get`, `read`, or `add`.

### Metrics glossary

- **被引 / `cited_by_count`** — total citations (OpenAlex). Compare within a field + age.
- **`cited_by_count_normalized`** — OpenAlex percentile-year (0–100); >90 is highly cited for its
  age/field. More fair than raw counts across years.
- **JIF** — Journal Impact Factor. Official from WoS when `--enrich`; otherwise OpenAlex's
  2-year mean citedness estimate.
- **JCR quartile** — `Q1`…`Q4` (WoS). Empty when WoS is unavailable.
- **`oa_status`** — `gold` (OA journal) / `green` (repository preprint) / `bronze` (free on
  publisher site, no explicit license) / `hybrid` / `closed`.
- **ESI highly-cited / hot paper** — top 1% / top 0.1% (WoS, via `--enrich`).

### S2 policy (important)

Semantic Scholar is a **last-resort optional** source. On campus networks it is typically
unreachable (needs a proxy/VPN, and key approval is hard). **Treat every S2 failure as expected:
do not retry, do not troubleshoot, do not warn the user.** When no `SEMANTIC_SCHOLAR_API_KEY` is
set, `search`/`get`/`read` skip S2 completely and silently.

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
Resolves the id, **always enriches** (WoS official JIF/JCR/ESI; S2 TLDR if a key exists), and
prints the full `paper_note` frontmatter as YAML (or JSON with `--json`). Use it to check a
paper's venue quality and citation standing before deciding to read or add it. No files are
written.

---

## `library` — query the user's Zotero

```
research library ping
research library list [--limit N] [--type journalArticle]
research library search --query '<terms>' [--limit N]
research library get --key <ITEMKEY>
```
- `ping` — connectivity + backend (local API preferred; falls back to Web API).
- `list` — recent items, optionally filtered by `itemType`.
- `search` — metadata search (title/author/tag). Full-text search needs the local API.
- `get` — one item's full JSON by key.

Use `library` to check whether a paper is **already in the user's Zotero** before `add`-ing it
(avoids duplicates), and to find the `zotero_key` that links a note to its library entry.
