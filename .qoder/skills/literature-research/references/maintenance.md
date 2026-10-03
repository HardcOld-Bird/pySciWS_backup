# Maintenance guide

The `research` CLI is a thin facade over 13 backend modules in
`src/pysci/skills/literature_research/tools/`. You rarely need to touch them; this guide is for when a
source, the PDF extractor, or a publisher page breaks, or when you want to extend the system.

> Reminder: for files under `src/pysci/skills/`, verify current content from disk (e.g. PowerShell
> `Get-Content -Encoding UTF8` / `Select-String`); an IDE/index cache may show stale content right
> after the directory is moved or renamed.

---

## 1. Architecture

```
research.py            ← CLI facade, 14 subcommands: doctor/search/read/get/add/citecheck/citegraph/
                          library/journal/review/rag/index/cache/ingest (orchestration only)
  ├─ config.py         ← loads project-root .env; exposes `settings` + `http_session()`
  ├─ notes.py          ← LEAF: paper-note frontmatter read/write/normalize/merge + slugify +
  │                       normalize_last_name + derive_journal_tier (stdlib + PyYAML + config only)
  ├─ journal_metrics   ← free journal-quality layer: SCImago SJR local index (exact-ISSN match); `research journal`
  ├─ openalex_client   ← primary search + metadata + work_to_note_frontmatter()
  ├─ arxiv_client      ← preprints + download_pdf() + arxiv_to_note_frontmatter()
  ├─ wos_client        ← enrich_openalex_work(): WoS accession no. (wos_id) + Times Cited (silent-fail)
  ├─ citation_verify   ← citation integrity gate: OpenAlex + Crossref + arXiv 3-source cross-check; verify_citation/verify_frontmatter + render (`research citecheck` + `add` gate)
  ├─ zotero_cli        ← ZoteroCli: delegates to community `zotero-cli --json` (zotero-mcp): ping/list/search/get/create_item_from_metadata/add_note + frontmatter_to_bibtex
  ├─ browser_fetch     ← Playwright: fetch_all/fetch_pdf/fetch_html + PublisherAdapter
  ├─ pdf_extract       ← extract_pdf(): arXiv LaTeX source first (equation-faithful) / MinerU cloud via
  │                       mineru-open-sdk / pymupdf4llm (fallback)
  ├─ local_ingest      ← bulk-ingest a LOCAL folder of PDFs (copy → extract → manifest ledger); `research ingest`
  ├─ rag               ← PaperQA2 semantic RAG over cache/extracted/: build_index/search (embedding-only, free) + optional ask (free→paid→degrade); `research rag`
  └─ cache_manager     ← two-tier cache governance: stats/clean/prune + bump_mtime (LRU)
```

Dependency direction is one-way: `research` → clients → {`notes`, `cache_manager`,
`journal_metrics`} → `config`. Every client uses **relative imports** (`from .config import
settings`), so moving the whole `literature_research/` package does not break imports; `config.py` in
turn resolves all paths via `pysci.paths` (marker-based, see §2), so relocating the package does not
break path resolution either. Two modules are deliberately **leaves**:

- `notes` depends only on stdlib + PyYAML + `config`. It must never import a sibling client, or the
  frontmatter helpers would turn into a cycle hub — every client needs them.
- `cache_manager` depends only on `config` + stdlib + `notes.load_frontmatter`. Clients import
  `bump_mtime` **and the Tier B trio** (`cache_key` / `read_cache` / `write_cache`) from it; still
  acyclic, because `cache_manager` never imports the clients. Before that trio was promoted,
  `arxiv_client` and `citation_verify` reached into `openalex_client._cache_key` / `._read_cache` /
  `._write_cache` by **private name** across module lines — a hidden coupling that also made the
  cache filenames lie (arXiv responses were written as `openalex_arxiv_search_*.json`).
  `cache_key(..., prefix=)` now names the real source. `openalex_client` keeps thin delegating shims
  under the old private names so its own tests stay put.

**The unified work-dict contract.** `openalex_client._extract_work_summary()` defines the canonical
shape that all sources aim for:
`openalex_id, doi, arxiv_id, title, publication_year, publication_date, type, cited_by_count,
cited_by_percentile_year, is_oa, oa_status, oa_url, journal, journal_issn_l, journal_issn[],
journal_openalex_id, listed_in[], journal_tier, journal_tier_basis[], publisher, volume, issue,
first_page, last_page, authors[{name, is_corresponding, institutions, …}], first_author_last_name,
abstract, concepts[{name, id, score}], topics[{name, id, score}], referenced_works_count,
referenced_works[], related_works[], counts_by_year[], _raw`. Each source has a
`*_to_note_frontmatter()` converter mapping its raw record to the shared `paper_note` frontmatter.
Keep new sources compatible with this contract.

`referenced_works` is truncated to `max_refs` (default **100**; `referenced_works_count` always keeps
the true total) — `research citegraph` snowballs through it, then `works_by_ids()` re-fetches those
ids in batches of `MAX_IDS_PER_REQUEST` (50, OpenAlex's per-filter `|` limit). Forward citations come
from `works_citing()` (`filter=cites:W…`), which needs a real OpenAlex id — an arXiv-only paper must
be resolved through DOI/title first.

---

## 2. Config & paths (`config.py`)

Paths come from `pysci.paths`, which locates the project root by **marker** (the first ancestor dir
containing both `pyproject.toml` and `.python-version`) — not by fragile `parents[N]` math:

```python
from pysci.paths import LITERATURE_ROOT, PROJECT_ROOT  # marker-based root discovery

# MODULE_DIR is the literature data area (papers/shortlists/reviews/templates/cache)
MODULE_DIR = LITERATURE_ROOT  # = data/skills/literature_research/
CACHE_DIR_ENV_DEFAULT = "data/skills/literature_research/cache"
```

- `.env` is loaded from `PROJECT_ROOT/.env`; `CACHE_DIR` there overrides the default.
- The **code** lives in `src/pysci/skills/literature_research/` (installed as part of the `pysci`
  package); the **data** (papers/shortlists/reviews/templates/cache) lives at `data/skills/literature_research/` (mirroring the code tree),
  keeping git-ignored cache + PDFs out of the package tree.
- **If paths go wrong**, the usual cause is the project-root markers moving: check `_ROOT_MARKERS` in
  `src/pysci/paths.py`, then run `research doctor` to confirm `模块根 / 缓存目录 / 项目根` are correct.
- `settings.module_dir` is the base for `papers/`, `shortlists/`, `reviews/`, `templates/`,
  `INDEX.md`; `settings.cache_dir` for `pdfs/`, `extracted/`, `api_responses/`, `html_fulltext/`.

Credentials (all optional except none are strictly required for OpenAlex/arXiv):
`OPENALEX_EMAIL`, `OPENALEX_API_KEY`, `WOS_API_KEY`, `ZOTERO_USER_ID`, `ZOTERO_API_KEY`,
`MINERU_TOKEN`, `PDF_EXTRACT_BACKEND`, `SILICONFLOW_API_KEY`, `SILICONFLOW_BASE_URL`,
`HTTP_TIMEOUT_SECONDS`, `HTTP_MAX_RETRIES`. The RAG layer (§5d) gates on `SILICONFLOW_API_KEY`
(`settings.pqa_ready`) and reads the optional `PQA_EMBEDDING`/`PQA_LLM`/`PQA_LLM_FALLBACK`/`PQA_HOME`
overrides (all have built-in defaults).

---

## 3. PDF → Markdown (`pdf_extract.py`)

- `available_backends()` returns what the environment can actually run: `mineru-cloud` when
  `MINERU_TOKEN` is set, `pymupdf4llm` when that module is importable.
- `extract_pdf(path, backend=None)` with `backend=None`/`auto` picks the best available
  (MinerU cloud first). It caches results in `cache/extracted/` (`use_cache`/`write_cache`).
- **MinerU cloud** is the primary: a VLM pipeline with OCR that renders equations as LaTeX and
  tables as HTML. It goes through the community **`mineru-open-sdk`** (`from mineru import MinerU`),
  which owns auth / upload / polling / result unpacking (only needs `httpx` + `MINERU_TOKEN`);
  `_mineru_extract_pages()` just calls `client.extract(path, model="vlm", ocr=True, formula=True,
  table=True, language="en", extra_formats=["latex"], timeout=…)` and reads `result.markdown`.
  Typical cost ~1–3s/page.
- **MinerU Precision Extract has a 600-page single-file limit** (exceeding it raises the SDK's
  `PageLimitError`). `_extract_with_mineru_cloud()` handles this transparently: it reads the page
  count via fitz and, if > `MINERU_MAX_PAGES` (600), converts in `MINERU_CHUNK_PAGES` (600) page
  segments via the SDK's `pages="a-b"` range (no more fitz temp-file splitting), then concatenates
  the parts (each tagged `<!-- MinerU pages a-b (i/N) -->`). So `read`/`ingest` on a big textbook
  "just works" — expect `MinerU 分段 i/N: 页 a-b` log lines. Quota: MinerU's ~1000-page allowance
  is a *fast-track* quota, not a daily hard cap — beyond it jobs still run, just slower (normal
  queue); the daily file cap is 5000.
- **pymupdf4llm** is the local fallback: fast, CPU-only, but **equations are lost**. Use it only
  when equations don't matter or MinerU is down.
- **arXiv LaTeX source path** (`_tex_to_markdown`, tried *before* any PDF backend once an arXiv id is
  known): unpacks the e-print tarball, picks the main `.tex`, converts it. Step 1 is
  `_strip_tex_comments` — it removes `%` comments (honouring `\%` escapes), passes `verbatim`-class
  environments through untouched, and drops `comment` environments whole. Step 2 cuts everything
  outside `\begin{document}…\end{document}`. Only *then* do the two targeted regexes run:
  `_TEX_FRONT_MATTER_DROP_RE` deletes the REVTeX front-matter macros that carry no reading value
  (`\affiliation`, `\altaffiliation`, `\email`, `\onlinemail`, `\homepage`, `\thanks`, `\date`,
  `\address`, `\maketitle`, `\tableofcontents`, `\keywords`, `\preprint`, `\fax`), while
  `_TEX_TITLE_RE` / `_TEX_AUTHOR_RE` promote `\title` to an `# H1` and merge `\author` into one
  `**Authors:**` line; `_TEX_GRAPHICS_DROP_RE` removes `\includegraphics` and the figure wrappers but
  keeps `\caption` (already turned into `_Figure/Table caption: …_`). Verified end-to-end on
  `1803.04110`: 717 lines out, **0** comment lines, **0**
  `\documentclass`/`\usepackage`/`\begin{document}` leaks, all eight body sections (`Introduction` …
  `Conclusion`) intact.
  **Order matters and is load-bearing**: the comment strip must come *first*, because the
  `\begin{document}` cut routinely fails on multi-file arXiv projects (`\input{sec1.tex}`), so it
  cannot be relied on to swallow commented-out lines. The same reason means a `\newcommand`/`\def`
  written *inside* the body survives — those are only removed when the preamble cut succeeds.
  **Further known limits** (pre-existing; none touch body prose or equations): the `thebibliography`
  block is kept as *raw REVTeX* (`\bibinfo`/`\bibfield`/`\BibitemShut`) — lines 136–~705 of that same
  717-line file, i.e. ~80 % of it and pure noise to `rag`'s embedder; in-text citations stay as
  BibTeX *keys* (`systems[EP2]`), key → number mapping not being implemented; `\ref` becomes
  `§label`; accents are not converted (`Aubry-Andr{\'e}-Harper`). Pinned by
  `test_tex_to_markdown.py`.

---

## 4. Paywalled full text (`browser_fetch.py`) — the most fragile part

`browser_fetch` drives a real browser via Playwright to get HTML full text, the article PDF, and
supplementary files. It auto-selects a browser channel (chrome → msedge → bundled chromium) and,
on a Cloudflare challenge, retries in headed mode. Results come back as a `BundleResult`
(`ok, title, slug, html_path, pdf_path, supp_paths, cloudflare, institutional_access, seconds,
adapter, browser, char_count, notes`).

### Publisher adapters (why it depends on page structure)

Each publisher's full text sits in a different DOM container. An adapter maps a host to CSS
selectors:

```python
@dataclass
class PublisherAdapter:
    name: str
    hosts: tuple[str, ...]              # matched against the URL hostname (subdomains included)
    fulltext_selectors: tuple[str, ...] # tried in order; first hit wins
    wait_selector: str | None = None    # element to wait for before extracting

APS_ADAPTER = PublisherAdapter(
    name="aps", hosts=("journals.aps.org",),
    fulltext_selectors=("#fulltext-content", "section.fulltext div.content", "div.article-fulltext"),
    wait_selector="#fulltext-content",
)
GENERIC_ADAPTER = PublisherAdapter(      # fallback for any unmatched host
    name="generic", hosts=(),
    fulltext_selectors=("article", "main", "[role=main]", "#content", "body"),
)
ADAPTERS: tuple[PublisherAdapter, ...] = (APS_ADAPTER,)   # ← the registry

def detect_adapter(url):  # hostname match, else GENERIC_ADAPTER
```

Body extraction runs through **`trafilatura`**: an in-page JS routine (`_JS_EXTRACT_HTML`) pulls the
`outerHTML` of the first matching selector, `_wrap_document()` re-roots it as a full `<html>` doc,
and `_trafilatura_body()` converts it to structured Markdown (`favor_precision`; tables/images/links
kept). `_extract_body()` falls back to trafilatura-on-full-page, then to the old innerText routine
(`_JS_EXTRACT`) if trafilatura is unavailable or yields <200 chars — so the adapter's
`fulltext_selectors` still drive scoping. Same-host PDF / supplement links are collected by separate
JS routines (`_JS_PDF_LINK` / `_JS_SUPP_LINKS`).

### Add support for a new publisher

1. Open one of its article pages in a browser; DevTools → find the element that wraps the article
   body (ideally excluding nav/ads). Note a stable CSS selector (id or class).
2. In `browser_fetch.py`, add an adapter and register it:
   ```python
   NATURE_ADAPTER = PublisherAdapter(
       name="nature",
       hosts=("www.nature.com", "nature.com"),
       fulltext_selectors=("article", "div.c-article-body", "main"),
       wait_selector="article",
   )
   ADAPTERS = (APS_ADAPTER, NATURE_ADAPTER)
   ```
3. Test: `research read <a-doi-from-that-publisher> --headed` and check the printed `char_count` /
   the extracted full-text file looks like the article, not the navigation.

### Fix a publisher that stopped working

Symptom: `read` returns a page but the full text is empty/truncated, or it's all nav/boilerplate —
the publisher changed their markup. Re-inspect the DOM (step 1 above) and update that adapter's
`fulltext_selectors` (put the most specific selector first, keep generic ones as fallbacks). If a
login/cookie wall appeared, `institutional_access` in the result will be false and you'll need
`--headed` to sign in interactively.

---

## 5. Cache governance (`cache_manager.py`)

The module keeps a **two-tier** cache under `settings.cache_dir` and is the only place eviction
policy lives (`research cache` is a thin facade over it).

| Tier | Dirs | Re-fetch cost | Lifetime | Eviction |
|---|---|---|---|---|
| **A — persistent** | `pdfs/`, `extracted/`, `html_fulltext/` | High (MinerU quota / browser / download) | Kept **forever**, reused on hit | **Manual only**: `cache prune` (LRU) |
| **B — volatile** | `api_responses/` | Low (re-fetchable JSON) | Dead once TTL expires | **Auto** hook + manual `cache clean` |

- `research cache stats` — per-tier size / file count / newest mtime / expired-B count / total vs
  soft limit / last autoclean time.
- `research cache clean [--all] [--older-than N] [--dry-run]` — delete expired (or all) Tier B.
- `research cache prune [--max-mb N] [--dry-run] [--keep-referenced/--no-keep-referenced]` — evict
  Tier A by LRU until total ≤ target (default `CACHE_SOFT_LIMIT_MB`); `html_fulltext/<slug>/` is
  evicted as a whole bundle directory.
- **LRU via mtime**: Windows atime is unreliable, so every cache *hit* calls `bump_mtime(path)`
  (`os.utime`) to make mtime ≈ last access. Tier B hits are bumped **centrally** inside
  `cache_manager.read_cache`, so OpenAlex / arXiv / Crossref all get it for free. The remaining call
  sites are `pdf_extract._read_cache` (its own *text* cache — same function names, different
  semantics and return type, deliberately **not** merged with the Tier B trio),
  `arxiv_client.download_pdf`, the browser-fetch bundle-reuse path, and `research.py`'s
  `{stem}_fulltext.md` hit.
- **Auto-clean hook**: `research.main()` calls `cache_manager.maybe_autoclean()` for the six
  network-producing commands (`search`/`read`/`get`/`add`/`citecheck`/`citegraph`) only. It reads
  `cache/.autoclean_state.json` (`last_autoclean`); if older than `CACHE_AUTOCLEAN_INTERVAL_DAYS`
  it runs `clean_tier_b()` and rewrites the state. Fully `try/except` — never fatal, prints one line
  only when it actually deletes something.
- **`keep_referenced`**: `prune` protects `protected_paths()`, the union of **two** sources — and
  losing either one fails *silently*. The first is files referenced by any `papers/*.md` frontmatter —
  `REFERENCED_FIELDS` = (`local_pdf_path`, `extracted_md_path`, `extracted_html_path`). The third
  one must stay in that tuple: when `read` cannot get a PDF it records *only*
  `extracted_html_path` (see *When no PDF can be had: the HTML fallback* in `read.md`), so dropping
  it would leave those notes'
  `cache/html_fulltext/` bundles unprotected. `referenced_paths()` parses that frontmatter via
  `notes.load_frontmatter` (`yaml.safe_load`) — importing the `notes` **leaf** keeps this module
  cycle-free without hand-rolling a parser. It used to be a line-based regex that skipped every
  `-`-prefixed line and returned quoted values *quotes included*; on block-style and flow-style notes
  alike it therefore yielded an empty set, and `--keep-referenced` protected **nothing**. Do not
  revert it (regression pinned by `test_referenced_paths.py`).
- **The second source is `ingest/manifest.json`** (`ingest_manifest_paths()`, fields
  `MANIFEST_PATH_FIELDS` = `md_path` + `pdf_path`). `research ingest`'s products hang off **no** note
  — the manifest is their only record (§5b) — so `referenced_paths()` alone returns an empty set for
  them and `--keep-referenced` protected *nothing*. Measured on the real tree, `prune --max-mb 0
  --dry-run` listed **41** eviction units before this and **8** after; the 33 rescued files are the
  entire 2026-09 ingest batch, which is also the *oldest* by mtime and therefore precisely what LRU
  eats first. Re-extraction costs MinerU quota. `protected_paths()` is the union; keep
  `referenced_paths()` semantically pure (notes only) so each source stays separately testable rather
  than hiding behind one assertion that can't tell which side worked. It parses the JSON **directly**
  instead of calling `local_ingest.load_manifest`, for two reasons: that would close an import cycle
  (`local_ingest` → `pdf_extract` → `cache_manager`), and its "a missing manifest is an error"
  semantics are wrong on a *protection* path — raising there turns "protect" into "refuse to clean",
  which is worse than not protecting. A missing / corrupt / mis-shaped manifest therefore degrades to
  an empty set (invariant 1).
- **HTML bundle reuse (`.by_url`)**: a bundle's slug is only known after the page title is fetched,
  so hits can't be predicted by slug up front. `browser_fetch` therefore keeps a URL index at
  `cache/html_fulltext/.by_url/<sha1(url)>.json` recording `{url, slug, out_dir, ts, ok, ...}`.
  On `fetch_bundle(use_cache=True)`, a fresh entry whose files still exist rebuilds the
  `BundleResult` **without launching a browser**; `research read` passes `use_cache=not --refresh`.
- **No double copy (G4)**: `research read` extracts with `write_cache=False`, so the canonical
  `cache/extracted/{stem}_fulltext.md` is the only extracted artifact (pdf_extract's own CLI still
  caches by default and is unaffected). On a later `read` of the same paper, that file is a hit and
  fetch + extraction are skipped entirely.

Cache governance config (`.env`, all optional): `CACHE_B_MAX_AGE_DAYS` (default 7),
`CACHE_AUTOCLEAN_INTERVAL_DAYS` (default 7), `CACHE_SOFT_LIMIT_MB` (default 2048).

> **`cache/extracted/` is git-TRACKED (deliberate exception).** The literature `.gitignore` uses
> `cache/*` + `!cache/extracted/` so the whole cache stays ignored **except** the extracted Markdown,
> which is MinerU-quota-expensive and worth version-controlling. (The project-root `.gitignore` must
> NOT exclude `.../cache/` as a whole directory, or Git's "parent dir excluded" rule makes the
> `!cache/extracted/` re-inclusion inert.) Consequence for `prune`: it can delete tracked `.md` under
> `cache/extracted/` (showing as Git deletions). Measured on the real tree, Tier A is 339 MB against
> the 2 GB soft limit and `extracted/` is only 19 MB of that — but `prune` evicts on **total** Tier A
> across all three dirs, which is dominated by `pdfs/` (316 MB). The limit is therefore reached by
> *PDF* growth while the units eaten first are the *oldest* by mtime, i.e. the extracted corpus.
> Keep `--keep-referenced` on (it is the default) and read the `--dry-run` list before a real prune —
> re-extraction costs quota.
>
> **Not everything under `cache/extracted/` is protected *or* tracked.** Two classes fall outside both
> sources. (i) `read`'s `<slug>_fulltext.md` before its note records it — transient, since the `read`
> that produced it also wrote the note. (ii) Cross-skill spillover: `comsol_simulation/tools/docs.py`
> calls `pdf_extract.extract_pdf()` **without** `write_cache=False`, so MinerU's own cache copy of
> each COMSOL manual lands here as `*_<hash>.mineru-cloud.md` (5 files, 12.5 MB, git-**untracked**,
> absent from the literature manifest). A prune can therefore delete them outright — quota lost, but
> *not* data: `docs.py` writes the canonical copy to `comsol_simulation/docs/*.md`, and that is what
> its FTS5 index reads.

---

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

## 5d. Semantic RAG layer (`rag.py` / `research rag`)

A local, **embedding-first** retrieval layer over the MinerU-extracted corpus (`cache/extracted/**/*.md`),
built on **PaperQA2** (`paper-qa`, a core dependency — no torch) + **SiliconFlow** (OpenAI-compatible).
Design intent: the Agent's own infrastructure for reading *across* the local library — `search` is the
primary path (pure embedding, free, no LLM); `ask` is an optional convenience that must never block.

- **Backend import is lazy + guarded.** `_import_backend(models=…)` first checks `settings.pqa_ready`
  (= `bool(SILICONFLOW_API_KEY)`) and raises a clear `RuntimeError` naming the missing key if not; it sets
  `LITELLM_LOCAL_MODEL_COST_MAP=True` **before** importing litellm (avoids a remote cost-map fetch that can
  hang), silences litellm's debug banner + a known harmless `async_success_handler` RuntimeWarning, then
  `_register_litellm_models()` declares each model (with **and** without the `openai/` prefix) incl.
  `max_input_tokens` — omitting it makes embedding calls `KeyError`. `paperqa`/`litellm` are imported only
  here, so `import rag` / `rag status` / `index_status()` never pull the heavy stack or touch the network.
- **S2-free by construction.** `_build_pqa_settings()` sets `parsing={"use_doc_details": False,
  "multimodal": False}`, so PaperQA2 does **not** call Semantic Scholar / Crossref for per-doc metadata.
  Citation/title/year/DOI/first-author are derived **offline** from the `.md` head by `_derive_meta_from_md`
  (first `#` H1 → title; a four-digit year only next to `published`/`accepted`/`received`/`©`; the first
  `DOI:`; the first plausible author line — skipping URL/affiliation lines and requiring ≥2 name tokens).
  A journal paper yields `Xia et al. (2025)`; a manual/no-author doc degrades to its title. `_map_mailto_env`
  still forwards `OPENALEX_EMAIL` → `CROSSREF_MAILTO`/`OPENALEX_MAILTO` for politeness if any source is hit.
- **Persistence + incremental index.** `build_index(paths=None, rebuild=False)` embeds each candidate `.md`
  (docname = `__`-joined path-relative-to-`cache/extracted` stem) via `Docs.aadd(citation=…, title=…, doi=…)`,
  then pickles the `Docs` to `cache/rag/index.pkl` + writes `index_meta.json` (`files{docname:{path,mtime,
  title,year,doi,first_author,citation}}`, `n_docs`, `n_chunks`, `embedding_model`, `paperqa_version`,
  `built_at`). Re-running is incremental: same mtime → **skip**, changed mtime → **stale** (not silently
  overwritten), new → **add**; `--rebuild` starts fresh. `_load_docs` returns `None` on a version mismatch or
  corrupt pickle (→ treated as "no index"). `PQA_HOME` overrides the index dir (default `cache/rag/`).
- **`search` (primary).** `Docs.retrieve_texts(query, k)` → `list[Text]` (MMR-ranked, embedding-only, no LLM).
  Each `Text` maps to a `RagChunk(rank, text, docname, citation, source_path, chunk_name)`; the source
  attribution chain is `chunk.text` + `chunk.doc.docname` + `chunk.doc.citation`, with `source_path` recovered
  from `index_meta.json`'s docname→path map. No index → a clear "run `research rag index`" error.
- **`ask` (optional, never blocks).** Tries the free `PQA_LLM` (`Qwen2.5-7B`, cost 0) → on any failure the paid
  `PQA_LLM_FALLBACK` (`Qwen2.5-32B`) → if both fail (or the backend/index is unavailable), `_degrade` sets
  `backend="none"`, `degraded=True`, and fills `result.search` with plain `search()` output instead of raising.
  `render_ask` labels the degraded case. This "free → paid → degrade, no noise" ladder is a hard product
  requirement (the Agent is itself a strong LLM; `ask` is a convenience, never a dependency).
- **Data structures** `RagChunk` / `RagSearchResult` / `IndexReport` / `RagAskResult` are dataclasses with
  `to_dict()` (JSON-serializable, nested dataclasses folded) for `--json`. `research.py`'s `cmd_rag` dispatches
  `index/search/ask/status`; `main()` does **not** attach the Tier-B autoclean hook to `rag` (it writes no
  `api_responses`). Config: `settings.pqa_embedding`/`pqa_llm`/`pqa_llm_fallback`/`pqa_home` (+ `pqa_ready`).

---

## 5e. Note frontmatter & merging (`notes.py`)

A **leaf** module (stdlib + PyYAML + `config` only) holding every helper that touches a note's
frontmatter. It exists because these helpers had been copied into five modules
(`research.py` / `cache_manager.py` / `citation_verify.py` / `browser_fetch.py` /
`{arxiv,openalex}_client.py`) and drifted apart; all copies are deleted in favour of this one. It
must never import a sibling client (§1) — every client needs it, so one wrong import makes it a
cycle hub.

- **`split_note(text)` → `(fm, body)`** and **`load_frontmatter(text)` → `fm`** parse with
  `yaml.safe_load`, so block-style *and* flow-style lists both round-trip. Never hand-roll this
  again: the previous line-based parser skipped every `-`-prefixed line (losing `authors` /
  `topics` wholesale) and returned values *quotes included* — three modules carried that same bug
  independently. Both return `({}, original_text)` on a missing or unparseable frontmatter instead
  of raising, so `index --fix` and `referenced_paths()` skip a corrupt or handwritten file
  gracefully.
- **The body-fidelity invariant is enforced inside `_FM_BLOCK_RE`.** Its delimiter allows `[ \t]*`,
  not `\s*` — `\s` contains `\n`, and a greedy match would eat the blank line following the closing
  `---` (real notes have one). Only *one* newline after that `---` is consumed, and `render_note`
  puts exactly one back. That is what makes `split_note → render_note` byte-identical, and so what
  makes "the body never changes" a checkable property rather than an aspiration.
- **`dump_frontmatter` is deterministic and deliberately unwrapped.** `_order_fields` emits known
  keys in `FIELD_ORDER` (mirroring `templates/paper_note.md`) and unknown keys after them in
  insertion order; `sort_keys=False`, `allow_unicode=True`, `default_flow_style=False`. The
  `_DUMP_WIDTH = 4096` is not a style preference: PyYAML's default (or the `width=100` this design
  originally proposed) folds a long title across lines, so the value survives but the file's shape
  changes — which would drown the "the diff must contain only value changes" check that validates
  `index --fix`. `_NoteDumper.increase_indent` indents block sequences two spaces under their
  parent key to match existing notes; PyYAML's default puts them in the parent's column and would
  add pure-indentation noise to every note touched.
- **`merge_frontmatter(existing, incoming)` → `(merged, changed_keys)`** fills *blank* keys only
  and never overwrites a non-blank value. `_is_blank` counts `None`/`""`/`[]`/`{}` as blank but
  **not** `0` or `False` — those are real values, and overwriting one is data loss. Blank
  *incoming* values are skipped too (writing `None` into a missing key is just noise; presence is
  `normalize_frontmatter`'s job). An empty `changed_keys` means the caller must not touch the file
  at all, so mtime survives. Worth remembering: **merging cannot correct a wrong value.** A note
  written before a field's semantics were fixed keeps the old value — fix such a value by hand. But
  **don't hand-type the bibliographic data**: recompute it through the project's own funnel
  (`research._resolve_work` → `_enrich_work` → `_build_frontmatter`) in a throwaway script, then
  overwrite an **explicit whitelist** of machine fields and nothing else. That is how the three
  pre-WP-D notes were repaired (37 fields, bodies byte-identical): the values come out identical to
  what today's code gives a brand-new note, instead of being a transcription of what a website showed
  me. Four classes stay out of the whitelist *by name* — filename-determining keys
  (`title`/`short_title`/`year`/`first_author_last_name`, since renaming breaks `INDEX.md` and the
  `reviews/` wiki links), local facts (`zotero_*`, `*_path`, `added_date`), hand-written evaluation
  (`status`/`my_rating`/`related_to_my_work*`/`topics`/`methods`/`systems`), and `oa_url`/`oa_status`
  (an arXiv green-OA direct link beats the Cloudflare-gated publisher URL OpenAlex would substitute).
  A key the authoritative source also leaves blank is **skipped, not cleared** — an honest gap stays
  visible. Never use `add` for this (it creates a duplicate Zotero item) or `--overwrite` (it resets
  the body).
- **`normalize_frontmatter(fm, template=…)` → `(normalized, added_keys)`** is *not* interchangeable
  with `merge_frontmatter`: it adds key **presence** from `template_defaults()` even when the
  template default is itself empty, because the template is the field contract. A note missing a
  key makes `fm["x"]` raise and makes every downstream consumer (`INDEX.md` columns,
  `cache_manager.REFERENCED_FIELDS`, §5f) silently get nothing. This is what `index --fix` runs.
  It only ever adds — existing keys, blank ones included, are preserved verbatim, so a hand-filled
  `my_rating` / `status` / `related_to_my_work` is never clobbered.
- **`append_changelog(body, line)`** is the *only* body edit any automatic write path performs. It
  appends at the end of the `## Changelog` section (creating one at EOF if absent) and never
  rewrites an existing line. `_CHANGELOG_HEAD_RE` accepts an optional numeric prefix — without it,
  `review_note.md`'s `## 8. Changelog` would be missed and every `review sync` would pile up a
  duplicate section at EOF while §8 stayed empty forever. Trailing newlines are preserved verbatim:
  when Changelog isn't the last section, they carry the blank-line separator to the next heading.
- **`normalize_last_name`** *transliterates* diacritics (`Büttner` → `buttner`) rather than
  deleting them (`bttner`), and handles `Last, First` / `First M. Last` / `Last`. Single
  authoritative implementation — see §5c for the false FAIL the two divergent copies caused and for
  the filename consequence (`index --fix` reports a mismatch, never renames).
- **`derive_journal_tier(listed_in)` → `(tier, basis)`** maps OpenAlex's expert-panel lists
  (`TIER_LISTS`: JUFO / Norway / KI-JL) onto `top` / `leading` / `basic`, taking the highest tier
  reached and listing every entry that reached it, so a bare `top` is auditable back to *which*
  panel said so. `""` means **undeterminable** (none of the three covers the venue) and is
  deliberately not folded into `basic`. Binary membership marks (`cwts-core`, `erih-plus`,
  `medline`, `doaj`, `doyens`) carry no level and are ignored. Fully fault-tolerant: a
  non-iterable, a non-string element, an unknown list name or an out-of-range level is skipped,
  never raised. Pinned by `test_notes.py`.

---

## 5f. Journal quality metrics (`journal_metrics.py`)

The free substitute layer for official JIF / JCR quartile / JCI / ESI, which only the **WoS
Journals API** can supply (application still pending — `wos_client.py`'s module docstring records
the upgrade path so it needn't be researched again). Three fields coexist and mean different
things:

| Field | Source | Cost |
|---|---|---|
| `journal_tier` (+ `journal_tier_basis`, `listed_in`) | OpenAlex `listed_in` → `notes.derive_journal_tier` (§5e) | zero extra requests |
| `scimago_quartile` | local SCImago SJR index, exact-ISSN match | one manual CSV download per year |
| `jcr_quartile`, `esi_highly_cited`, `esi_hot_paper` | WoS Journals API | not wired — stays `""` / `null` |

**The module never touches the network.** It answers from a compact JSON index at
`settings.scimago_index_path` = `data/skills/literature_research/data/scimago_index.json`
(git-tracked; `settings.scimago_ready` is just `.exists()`). It is a *data asset*, not a cache
artifact, which is why it lives under `data/` rather than `cache/` and so escapes the `cache/*`
ignore rule.

- **`build_scimago_index(csv_path, *, year=None, out_path=None)`** turns the official CSV
  (scimagojr.com → *Journal rank*, ~15 MB) into `{"_meta": {…}, "by_issn": {"00319007":
  ["Q1", 2.845, 982], …}}` — only four columns survive (`Issn`, `SJR`, `SJR Best Quartile`,
  `H index`), written with `separators=(",", ":")` and no indent, ≈1.4 MB. Each detail below
  covers a real failure mode:
  - `_detect_delimiter` picks `;` or `,` by whichever appears more often in the header. The
    official export is semicolon-delimited (European style) but comma mirrors exist; hardcoding
    `;` turns every row into a single column and **silently produces an empty index**.
  - `_to_float` accepts the European decimal comma (`2,845` → `2.845`).
  - `_extract_issns` splits multi-value cells, and its separator set deliberately **excludes** `-`
    (else `0031-9007` splits in half). Since `;` is *also* the field delimiter, an unquoted
    multi-value Issn cell arrives split across fields; the builder compensates for the resulting
    column shift and rejoins them.
  - `_at()` reads columns out-of-bounds-safely — SCImago rows occasionally lack trailing columns,
    which must not cost the whole row.
  - `_infer_year` looks for a parenthesised year in the header (`Total Docs. (2024)`), else in the
    filename, else returns `None`. It never invents one; pass `--year` when it can't tell.
  - Raises `FileNotFoundError` / `ValueError` (empty CSV, missing required column, not one ISSN
    parsed). **This is the one path in the module that must not degrade silently**: a failed build
    that quietly wrote an empty index would blank `scimago_quartile` in every note with no visible
    cause — far harder to diagnose than an error. `cmd_journal` maps it to exit 1 with the reason
    on stderr, and a missing `--csv` to exit 2.
- **Query paths degrade silently**, per the module-wide invariant. `lookup(issn)` accepts a single
  ISSN *or* a candidate list (OpenAlex's `issn` array) and returns the first hit as
  `{issn, quartile, sjr, h_index, sjr_year}`, else `None` — index missing, unreadable, corrupt, or
  no match. A corrupt index is treated as no index: that is safer than half-working.
  `quartile_for(issn)` returns `""` rather than `None` because `scimago_quartile` is a string
  field, and only `""` reads as blank to `merge_frontmatter` (§5e). `_load_index` caches a single
  entry keyed on `(path, mtime, size)`, so a rebuilt index is picked up without a restart and tests
  can redirect the path freely.
- **Matching is exact-ISSN only, by design.** No fuzzy title matching: a title map would add
  ~1.5 MB and introduce mis-matches, while an ISSN is always available from OpenAlex / Crossref /
  WoS metadata. `research journal lookup <issn>` distinguishes "index not built" from "this ISSN
  isn't in the index" — the fixes are completely different, and collapsing them into one empty
  result is exactly what makes venue metrics look broken when they aren't.

### Refreshing the SCImago index (yearly)

1. Download the CSV **by hand** from `journal_metrics.DOWNLOAD_URL`
   (<https://www.scimagojr.com/journalrank.php>). The site returns 403 to programmatic fetches and
   needs form/JS interaction, so no auto-download is implemented or planned. `*.csv` is git-ignored
   under `data/skills/literature_research/` to keep the 15 MB original out of the repo.
2. `research journal build-scimago --csv <path> [--year 2024]`, then check the reported entry count
   is in the tens of thousands. A count near zero means delimiter or column detection missed; the
   command refuses to write that (exit 1), but verify the number anyway.
3. Update `data/skills/literature_research/data/SOURCE.md` — download URL, SJR year, download date,
   attribution (`SCImago Journal & Country Rank, data based on Scopus (Elsevier B.V.)`). The index
   JSON *is* tracked; the CSV is not.
4. Confirm with `research journal status` and the 【期刊质量指标】 section of `research doctor` that
   `sjr_year` moved.

Existing notes are **not** rewritten by a refresh. A note whose `scimago_quartile` is blank gets it
filled on the next `read` / `add` / `get`; a note that already has a value keeps the old quartile,
because `merge_frontmatter` never overwrites (§5e). Re-running `read <id>` therefore only helps
while the field is still blank — after that, update it by hand.

---

## 6. Common failures → fixes

### Renamed flags (the old spellings still work)

`--force` used to carry four unrelated meanings across four subcommands. It was split by semantics;
the old name survives everywhere as a **hidden** deprecated alias (`argparse.SUPPRESS`, so `-h`
doesn't advertise it) that sets the same `dest` and prints a one-line migration hint on stderr via
`_migrate_force_alias`. `_FORCE_RENAMED` is the authoritative table:

| Subcommand | Now | Was | Why it moved |
|---|---|---|---|
| `read` | `--refresh` | `--force` | It bypasses the *cache*, not a safety gate. |
| `ingest` | `--refresh` | `--force` | Same — re-extract entries already marked `done`. |
| `add` | `--allow-fail` | `--force` | It opens the **citation-integrity gate**. Sharing a name with an ordinary cache-refresh switch is precisely how an agent ends up bypassing verification without meaning to. |
| `index` | `--force` (unchanged) | — | Here the word literally means "write even when `papers/` is empty" — closest to its plain sense, so it stayed. |

Aliases are kept **indefinitely, with no removal date**: this project has no CI and one user, so a
breaking rename buys less than it risks. `_migrate_force_alias` runs only from `main()`; code that
builds a `Namespace` and calls `cmd_*` directly (including the existing tests) bypasses it, so
every read of the flag goes through `getattr` and tolerates a missing `force_deprecated` key.

`ingest` also gained an optional `action` positional (`run` | `status`, default `run`) in the same
pass — see §5b.

| Symptom | Likely cause | Fix |
|---|---|---|
| `read` gets no PDF; `cloudflare=true` | Bot challenge | Re-run with `--headed`; complete the challenge in the window. |
| Full text empty / all navigation | Publisher changed markup | Update that adapter's `fulltext_selectors` (§4). |
| `institutional_access=false`, paywalled | No subscription via this network | Use a campus VPN, or `read` the arXiv id / OA copy instead. |
| MinerU 401 / job fails | Bad/expired `MINERU_TOKEN` or quota | Check `.env`; temporarily `--backend pymupdf4llm`. |
| Equations missing in output | Fell back to pymupdf4llm | Ensure `MINERU_TOKEN` is set; check `research doctor` lists `mineru-cloud`. |
| `[wos] enrichment skipped` | `WOS_API_KEY` unset or endpoint changed | Expected — enrichment only adds `wos_id` + Times Cited; degrades silently. The Starter API never returns JIF / quartile / ESI, so where those fields actually come from: `jif` = OpenAlex `2yr_mean_citedness` **estimate**; `scimago_quartile` = the local SCImago SJR index (`journal_metrics`); `journal_tier` = OpenAlex `listed_in` (JUFO/Norway/KI-JL expert panels); `jcr_quartile` stays **empty** and `esi_*` stays **`null` (unknown)** until the WoS Journals API lands. Their absence is normal, not an error. |
| Zotero `不可达` / `zotero-cli 未安装` | Desktop app closed / local API off / CLI not on PATH | Start Zotero; Settings → Advanced → *Allow other applications*; run `zotero-mcp authorize-local` (choose *Always Allow*); or set Web API creds (`ZOTERO_API_KEY`+`ZOTERO_LIBRARY_ID`). If `zotero-cli` is missing, run `scripts\zotero_mcp\setup_zotero_mcp.ps1` then `uv tool update-shell` and restart the shell. |
| Paths wrong / `.env` not loaded | Project-root markers moved; `pysci.paths` can't find root | Check `_ROOT_MARKERS` in `src/pysci/paths.py` (§2); confirm with `research doctor`. |
| PowerShell mangles the command | Double quotes stripped / `&&` used | Single-quote multi-word args; chain with `;`. |
| `add` created a duplicate Zotero item | Ran `add` twice for one DOI | Check `library search` before adding; merge the dup via the `zotero` MCP (`duplicates find`) or delete it in Zotero. |
| Cache growing / disk pressure | Tier A artifacts kept forever by design | `research cache stats`; then `prune --max-mb N` (Tier A) or `clean` (Tier B). |
| `read` shows `命中缓存全文` but you want a fresh fetch | Cached `{stem}_fulltext.md` was reused | Re-run with `--refresh` (`--force` is a deprecated alias) to re-fetch + re-extract. |
| `add` blocked with `✗ 引用核验未通过` | Citation gate FAIL (≥2 sources hard-conflict on title/DOI/year) | Inspect with `research citecheck <doi> --json`; fix the mismatched field, or `--allow-fail` to override / `--no-verify` to skip. |
| `citecheck` reports `?NOT_FOUND` for a real paper | All three sources missed it (typo'd DOI, very new, or offline) | Check the DOI/id; a lone `openalex=不可达`/`crossref=不可达` is a network blip (downgraded, not a FAIL) — re-run. |
| `citecheck` first-author `△WARN` on an accented name | Cross-source transliteration/abbreviation noise | Expected — author surname is a soft signal and never blocks; title/DOI/year are the hard signals. |
| `rag search`/`ask` errors "未配置 SILICONFLOW_API_KEY" | `pqa_ready` false (no key in `.env`) | Set `SILICONFLOW_API_KEY` (+ `SILICONFLOW_BASE_URL`); confirm with `research rag status`. |
| `rag search` says "no index" / `先运行 research rag index` | Index never built, or paperqa version changed / pickle corrupt | Run `research rag index` (add `--rebuild` to force). `cache/rag/` is rebuildable + git-ignored. |
| `rag ask` prints a degrade notice + returns search results | Free **and** paid LLM both failed (quota / 503 / network) | By design — `ask` never blocks; use the returned `search` chunks or retry later. Check the `SILICONFLOW_API_KEY` quota. |
| `rag` embedding `KeyError` on max_input_tokens | A model was registered without `max_input_tokens` | Ensure `_register_litellm_models` declares it for both the bare and `openai/`-prefixed name (§5d). |

---

## 7. Extending the system

- **New data source**: add `tools/<source>_client.py` exposing `search_*()` and `get_*()` that
  return records convertible to the unified work-dict contract, plus a
  `<source>_to_note_frontmatter()`. Wire it into `cmd_search` (and optionally an enricher like
  `enrich_openalex_work`). Keep failures non-fatal.
- **New output type**: add a template under `templates/` and a `cmd_*` in `research.py` that fills it
  via `notes.render_note` / `notes.dump_frontmatter` + `_fill_placeholders`. The hand-rolled
  `_dump_yaml` / `_yaml_scalar` / `_slugify` / `_note_filename` / `_fmt_bytes` helpers are **gone** —
  all YAML and filename logic now lives in the `notes` leaf, so a new writer must go through it too
  (otherwise the byte-identical frontmatter rendering that `index --fix` and the merge path rely on
  stops being idempotent).
- **New PDF backend**: extend `pdf_extract.available_backends()` + `_pick_backend()` + `extract_pdf()`.

## 8. Verifying changes

```
.venv\Scripts\python.exe -m py_compile src/pysci/skills/literature_research/tools/<file>.py
.venv\Scripts\python.exe -m pysci.skills.literature_research.tools.research doctor
```
Then smoke-test the affected command (`search`/`read`/`get`/`add`/`citecheck`/`citegraph`/`library`/
`journal`/`review`/`index`/`rag`/`cache`/`ingest`). `doctor`
is the fastest way to confirm config, sources, backends, Playwright, Zotero, and the RAG layer are all wired
up. For `rag` specifically: `research rag status` (offline) then a small `research rag index --path <one .md>`
+ `research rag search '<q>' -k 3` is the cheapest end-to-end check (uses real SiliconFlow embedding).
