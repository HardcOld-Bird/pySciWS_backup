# maintenance 分册 5/11

> 含小节：5b. Bulk local-PDF ingestion (`local_ingest.py` / `research ingest`)；5c. Citation integrity gate (`citation_verify.py` / `research citecheck` + `add`)
> 原 `maintenance.md` 超 8KB 预算拆分（2026-10-10，backlog 20261010-refs-split-8kb）；分册索引见同目录 `maintenance.md`，按需只读所需分册。

## 5b. Bulk local-PDF ingestion (`local_ingest.py` / `research ingest`)

For a **local folder of PDFs** with no DOIs to resolve (e.g. a user's hand-maintained literature
repo), `read`/`add` don't fit — they are DOI/URL/download-centric. `research ingest` instead drives
a curated, git-tracked ledger `data/skills/literature_research/ingest/manifest.json`:

- Each entry: `id, theme, slug, type, priority, pages, source, status, pdf_path, md_path, note`.
  `source` is the real path relative to `source_root` (author it from an actual filesystem walk so
  special chars like `&`, `‐`, full-width parens match exactly). `priority` batches by cost
  (1 ≤30 pp, 2 = 31–150 pp, 3 >150 pp / unknown) so expensive jobs can be deferred — MinerU's
  ~1000-page allowance is a *fast-track* quota (beyond it jobs still run, just slower), not a hard
  daily cap.
- `run_ingest()` selects pending entries (skip `done` unless `--refresh`), sorts by `(priority, pages)`,
  then per entry: **copies** the PDF (originals untouched) to `cache/pdfs/<theme>/<slug>.pdf`,
  extracts via `pdf_extract.extract_pdf(..., write_cache=False)` (single canonical copy, same as
  `read`), writes `cache/extracted/<theme>/<slug>.md`, updates the entry, and **saves the manifest
  after every file** → fully resumable across sessions/interruptions. A per-entry exception is caught
  (`status=failed`, `note=<err>`), never aborting the batch.
- **Windows long paths**: sources deep in nested CJK-named folders can exceed MAX_PATH (260 chars),
  making plain `exists()`/`open()` fail. `_resolve_source()` retries with the `\\?\` extended-length
  prefix (absolute + backslashes) so those sources still copy; the copy lands at a short
  `cache/pdfs/<theme>/<slug>.pdf`, so extraction/caching are unaffected. When the long source blocked
  measuring `pages` at manifest time, `run_ingest()` backfills it from the copied PDF via fitz.
- Shape: `ingest [action] [flags]`, where `action` is an **optional positional** with
  `choices=[run, status]` defaulting to `run` — so `ingest`, `ingest run` and `ingest --status` all
  work. The `--status` boolean predates the positional and is kept (H3 converged the CLI onto two
  shapes without breaking either spelling).
- Flags: `--manifest`, `--status`, `--priority N`, `--theme T`, `--backend`, `--limit-pages N`,
  `--limit-files N`, `--dry-run`, `--refresh`. `main()` does NOT attach the autoclean hook to `ingest`.
- The manifest is generated once by a throwaway walk+classify script, then hand-curated; it is the
  single source of truth (no parallel catalog). Non-PDF assets (`.nb/.wls/.epub/.txt`) are listed
  under `non_pdf_assets` with `status=skipped` for provenance, never converted.
- **The manifest doubles as `cache prune`'s protection list** (§5a): ingest products hang off no
  note, so `cache_manager.ingest_manifest_paths()` reads `md_path` / `pdf_path` straight out of it.
  Consequence when editing it by hand: deleting an entry, or blanking its `md_path`, silently
  un-protects that artifact from LRU eviction. Entries with `status=pending`/`skipped` have no path
  fields and contribute nothing, which is expected.

---

## 5c. Citation integrity gate (`citation_verify.py` / `research citecheck` + `add`)

Catches fabricated / mis-paired references before they enter the library. **Three independent open
sources** — OpenAlex, **Crossref** (free REST, no key), arXiv — are queried for the same citation,
normalized to a common `SourceRecord`, then compared field-by-field. Semantic Scholar is deliberately
not used (removed in stage 1).

- **Crossref** is the only new HTTP source added here (`api.crossref.org/works/{doi}`, or
  `?query.bibliographic=<title>&rows=1` for a title-only fallback). It reuses `openalex_client`'s
  two-tier cache (`crossref_`-prefixed keys) and puts `settings.openalex_email` in the `mailto`
  polite-pool param. A **404 = “not found”** (`_crossref_message` returns `None`, i.e. reachable but
  absent) — distinct from a network error, which sets `reachable=False`.
- **Severity, not equality.** Each field comparator returns `hard` / `soft` / `None` (`_FIELD_SPECS`):
  title (`_cmp_title`, sim < 0.82 → hard), DOI (`_cmp_exact` → hard), year (`_cmp_year`, ±1 → soft,
  ≥2 → hard), first author (`_cmp_name` → **soft**), journal (`_cmp_journal` → soft, containment
  tolerated). Any hard → **FAIL**; only soft → **WARN**; else **PASS**; all sources absent →
  **NOT_FOUND**. Only FAIL blocks (`CitationVerdict.passed()` = `status != FAIL`).
- **Graceful degradation is the core invariant.** An unreachable source is recorded (`reachable=False`)
  and **never counted as a conflict** — a network blip must not fail a real citation. A FAIL requires
  ≥2 reachable sources to hard-conflict.
- **Transliteration pitfall (fixed in stage 6).** The per-client surname guessers *deleted*
  diacritics (`Büttner` → `bttner`) while `normalize_last_name` *transliterates* them (`→ buttner`).
  Comparing those two directly caused a false FAIL. Fix: every source derives the first-author surname
  from the **full author name** (`authors[0]`) through the same `normalize_last_name`, never trusting an
  upstream pre-processed `first_author_last_name`; and first-author is only a **soft** signal anyway.
  `test_openalex_and_crossref_transliterate_author_consistently` guards this.
  **There is now only one implementation**: the duplicate `_guess_last_name` helpers in
  `openalex_client` / `arxiv_client` were deleted in favour of `notes.normalize_last_name`, so the two
  cannot drift apart again. Be aware the fix *changes note filenames* for accented surnames
  (`bttner` → `buttner`): `research index --fix` therefore **reports** existing files whose name no
  longer matches `notes.note_filename(fm)` but never renames them — renaming would break
  `papers_reviewed` wiki links and any external citation of those paths.
- **The `claim` pseudo-source.** The note's own asserted values are folded in as `source="claim"` and
  compared against the real sources — this is what detects a DOI that doesn't match the title claimed.
- Entry points: `verify_citation(cite)` (a dict), `verify_frontmatter(fm)`, `verify_note_file(path)`
  (reads the note's frontmatter through `notes.load_frontmatter`; the `notes` **leaf** keeps this
  cycle-free without importing `research`. `_parse_frontmatter_scalars` survives only as an alias of
  that function).
  `research.py` wraps these: `_citation_gate` (the `add` pre-write check, degrades to allow on error)
  and `cmd_citecheck` (standalone; exit 1 on any FAIL). Render via `render_verdict` / `render_report`
  (`✓ △ ✗ ?`, ANSI red on FAIL when a TTY).

---
