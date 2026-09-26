---
name: literature-research
description: Search, read, evaluate, and manage academic physics literature (acoustics, non-Hermitian / exceptional points, topological, BIC, CPA, phononic). Drives a unified `research` CLI that fuses OpenAlex + arXiv search, fetches paywalled full text with a headless browser, extracts PDFs to Markdown with LaTeX equations (MinerU cloud), enriches metadata with journal impact factors and citation counts, generates structured evaluation notes, and syncs to Zotero. Use when the user asks to find or search papers/literature, read or analyze a specific paper, extract a PDF, run a literature review or survey, assess a paper's novelty / rigor / journal tier, build paper notes, or query and manage their reference library.
---

# Literature Research

A single CLI drives the whole literature workflow: **search → read → note → library → index**.

The backend lives in `src/pysci/skills/literature_research/` (9 client modules + a `research` facade).
**You do not need to read the backend code** — treat it as a black box and drive everything
through the `research` CLI below. Only open the code when maintaining it (see
[references/maintenance.md](references/maintenance.md)).

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
| `add <doi>` | Adding a paper to the library | Zotero item + note skeleton |
| `library <ping｜list｜search｜get>` | Querying the user's Zotero | Items from the local library |
| `index` | After adding or editing notes | Rebuilt `INDEX.md` |
| `ingest` | Bulk-archiving a **local** folder of PDFs (no DOI) | Copied PDFs + extracted Markdown + `ingest/manifest.json` ledger |
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
- [ ] 3. research add <doi>          # for each keeper → Zotero + note skeleton
- [ ] 4. research read <doi>         # for the ones to read closely → full text
- [ ] 5. Fill each note's TLDR / Key Claims / Novelty / Rigor / Relevance (see references/read.md)
- [ ] 6. research index              # refresh INDEX.md
- [ ] 7. (optional) Synthesize a survey in reviews/
```

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

- **OpenAlex** — primary source, no key, rich metadata + citation counts. Always available.
- **arXiv** — preprints, no key. Use for preprint-only or latest work.
- **Web of Science** — *optional* enrichment (official JIF / JCR quartile / ESI). Degrades
  silently to OpenAlex estimates when unavailable; a one-line notice is printed, no action needed.
- **Semantic Scholar** — *optional, last-resort*. **On campus networks it is usually unreachable,
  so failures are expected. Do NOT troubleshoot it, retry it, or warn the user about it.** It is
  skipped entirely when no API key is configured.
- **MinerU cloud** — primary PDF → Markdown backend (equations → LaTeX, tables → HTML). Needs
  `MINERU_TOKEN`. `pymupdf4llm` is the local fallback (fast, but equations are lost).

## Output locations

| Path | Contents |
|---|---|
| `data/skills/literature_research/papers/` | One structured note per paper (`{year}_{author}_{slug}.md`) |
| `data/skills/literature_research/shortlists/` | One search snapshot per query |
| `data/skills/literature_research/reviews/` | Multi-paper surveys |
| `data/skills/literature_research/INDEX.md` | Auto-generated library index (`research index`) |
| `data/skills/literature_research/ingest/manifest.json` | Ledger for bulk local-PDF ingestion (`research ingest`); git-tracked |
| `data/skills/literature_research/cache/` | Downloaded/copied PDFs + fetched HTML (git-ignored) **+ extracted full text under `cache/extracted/`, which IS git-tracked** (MinerU-quota-expensive `.md`, worth backing up) |

## When something breaks

1. Run `research doctor` — it reports every source, backend, and Zotero status.
2. For browser-fetch / MinerU / adapter / config issues, see
   [references/maintenance.md](references/maintenance.md).

## Reference files

- [references/search.md](references/search.md) — `search` / `get` / `library` details: all flags,
  source selection, metrics, screening, shortlist schema.
- [references/read.md](references/read.md) — `read` / `add` / `index` details: full-text extraction,
  the paper-note schema, and the **evaluation rubric** for judging a paper.
- [references/maintenance.md](references/maintenance.md) — how the backend works and how to fix or
  extend it (publisher adapters, PDF backends, config, common failures).
