# Maintenance guide

The `research` CLI is a thin facade over 11 backend modules in
`src/pysci/skills/literature_research/tools/`. You rarely need to touch them; this guide is for when a
source, the PDF extractor, or a publisher page breaks, or when you want to extend the system.

> Reminder: for files under `src/pysci/skills/`, verify current content from disk (e.g. PowerShell
> `Get-Content -Encoding UTF8` / `Select-String`); an IDE/index cache may show stale content right
> after the directory is moved or renamed.

---

## 1. Architecture

```
research.py            ← CLI facade: doctor/search/read/get/add/citecheck/library/index/ingest/rag/cache (orchestration only)
  ├─ config.py         ← loads project-root .env; exposes `settings` + `http_session()`
  ├─ openalex_client   ← primary search + metadata + work_to_note_frontmatter()
  ├─ arxiv_client      ← preprints + download_pdf() + arxiv_to_note_frontmatter()
  ├─ wos_client        ← enrich_openalex_work(): WoS accession no. (wos_id) + Times Cited (silent-fail)
  ├─ citation_verify   ← citation integrity gate: OpenAlex + Crossref + arXiv 3-source cross-check; verify_citation/verify_frontmatter + render (`research citecheck` + `add` gate)
  ├─ zotero_cli        ← ZoteroCli: delegates to community `zotero-cli --json` (zotero-mcp): ping/list/search/get/create_item_from_metadata/add_note + frontmatter_to_bibtex
  ├─ browser_fetch     ← Playwright: fetch_all/fetch_pdf/fetch_html + PublisherAdapter
  ├─ pdf_extract       ← extract_pdf(): MinerU cloud via mineru-open-sdk (primary) / pymupdf4llm (fallback)
  ├─ local_ingest      ← bulk-ingest a LOCAL folder of PDFs (copy → extract → manifest ledger); `research ingest`
  ├─ rag               ← PaperQA2 semantic RAG over cache/extracted/: build_index/search (embedding-only, free) + optional ask (free→paid→degrade); `research rag`
  └─ cache_manager     ← two-tier cache governance: stats/clean/prune + bump_mtime (LRU)
```

Dependency direction is one-way: `research` → clients → `config`. Every client uses **relative
imports** (`from .config import settings`), so moving the whole `literature_research/` package does
not break imports; `config.py` in turn resolves all paths via `pysci.paths` (marker-based, see §2),
so relocating the package does not break path resolution either. `cache_manager` depends
only on `config` + stdlib; the clients import `bump_mtime` from it, which does **not** create a
cycle (cache_manager never imports the clients).

**The unified work-dict contract.** `openalex_client._extract_work_summary()` defines the canonical
shape that all sources aim for:
`openalex_id, doi, arxiv_id, title, publication_year, publication_date, cited_by_count,
cited_by_percentile_year, oa_status, oa_url, journal, publisher, volume, issue, first_page,
last_page, authors[{name, is_corresponding, institutions, …}], first_author_last_name, abstract,
concepts[{name, id, score}]`. Each source has a `*_to_note_frontmatter()` converter mapping its raw
record to the shared `paper_note` frontmatter. Keep new sources compatible with this contract.

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
  (`os.utime`) to make mtime ≈ last access. Hits are bumped in `openalex_client._read_cache`,
  `pdf_extract._read_cache`, `arxiv_client.download_pdf`, and the browser-fetch reuse path.
- **Auto-clean hook**: `research.main()` calls `cache_manager.maybe_autoclean()` for the
  network-producing commands (`search`/`read`/`get`/`add`) only. It reads
  `cache/.autoclean_state.json` (`last_autoclean`); if older than `CACHE_AUTOCLEAN_INTERVAL_DAYS`
  it runs `clean_tier_b()` and rewrites the state. Fully `try/except` — never fatal, prints one line
  only when it actually deletes something.
- **`keep_referenced`**: `prune` protects files referenced by any `papers/*.md` frontmatter
  (`local_pdf_path` + `extracted_md_path`). `referenced_paths()` parses frontmatter with a built-in
  regex (no PyYAML) to avoid importing `research` (which would create a cycle).
- **HTML bundle reuse (`.by_url`)**: a bundle's slug is only known after the page title is fetched,
  so hits can't be predicted by slug up front. `browser_fetch` therefore keeps a URL index at
  `cache/html_fulltext/.by_url/<sha1(url)>.json` recording `{url, slug, out_dir, ts, ok, ...}`.
  On `fetch_bundle(use_cache=True)`, a fresh entry whose files still exist rebuilds the
  `BundleResult` **without launching a browser**; `research read` passes `use_cache=not --force`.
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
> `cache/extracted/` (showing as Git deletions). These files are small, so the 2 GB soft limit is
> effectively never hit by text; still, prefer `prune --keep-referenced` and think before pruning
> ingested full text — re-extraction costs quota.

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
- `run_ingest()` selects pending entries (skip `done` unless `--force`), sorts by `(priority, pages)`,
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
- Flags: `--status` (summary only), `--priority N`, `--theme T`, `--backend`, `--limit-pages N`,
  `--limit-files N`, `--dry-run`, `--force`. `main()` does NOT attach the autoclean hook to `ingest`.
- The manifest is generated once by a throwaway walk+classify script, then hand-curated; it is the
  single source of truth (no parallel catalog). Non-PDF assets (`.nb/.wls/.epub/.txt`) are listed
  under `non_pdf_assets` with `status=skipped` for provenance, never converted.

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
- **Transliteration pitfall (fixed in stage 6).** `openalex_client._guess_last_name` *deletes*
  diacritics (`Büttner` → `bttner`) while `normalize_last_name` *transliterates* them (`→ buttner`).
  Comparing those two directly caused a false FAIL. Fix: every source derives the first-author surname
  from the **full author name** (`authors[0]`) through the same `normalize_last_name`, never trusting an
  upstream pre-processed `first_author_last_name`; and first-author is only a **soft** signal anyway.
  `test_openalex_and_crossref_transliterate_author_consistently` guards this.
- **The `claim` pseudo-source.** The note's own asserted values are folded in as `source="claim"` and
  compared against the real sources — this is what detects a DOI that doesn't match the title claimed.
- Entry points: `verify_citation(cite)` (a dict), `verify_frontmatter(fm)`, `verify_note_file(path)`
  (parses frontmatter scalars without PyYAML / without importing `research`, avoiding a cycle).
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

## 6. Common failures → fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| `read` gets no PDF; `cloudflare=true` | Bot challenge | Re-run with `--headed`; complete the challenge in the window. |
| Full text empty / all navigation | Publisher changed markup | Update that adapter's `fulltext_selectors` (§4). |
| `institutional_access=false`, paywalled | No subscription via this network | Use a campus VPN, or `read` the arXiv id / OA copy instead. |
| MinerU 401 / job fails | Bad/expired `MINERU_TOKEN` or quota | Check `.env`; temporarily `--backend pymupdf4llm`. |
| Equations missing in output | Fell back to pymupdf4llm | Ensure `MINERU_TOKEN` is set; check `research doctor` lists `mineru-cloud`. |
| `[wos] enrichment skipped` | `WOS_API_KEY` unset or endpoint changed | Expected — enrichment only adds `wos_id`; degrades silently. The Starter API never returns JIF / quartile / ESI (those come from OpenAlex's estimate), so their absence is normal, not an error. |
| Zotero `不可达` / `zotero-cli 未安装` | Desktop app closed / local API off / CLI not on PATH | Start Zotero; Settings → Advanced → *Allow other applications*; run `zotero-mcp authorize-local` (choose *Always Allow*); or set Web API creds (`ZOTERO_API_KEY`+`ZOTERO_LIBRARY_ID`). If `zotero-cli` is missing, run `scripts\zotero_mcp\setup_zotero_mcp.ps1` then `uv tool update-shell` and restart the shell. |
| Paths wrong / `.env` not loaded | Project-root markers moved; `pysci.paths` can't find root | Check `_ROOT_MARKERS` in `src/pysci/paths.py` (§2); confirm with `research doctor`. |
| PowerShell mangles the command | Double quotes stripped / `&&` used | Single-quote multi-word args; chain with `;`. |
| `add` created a duplicate Zotero item | Ran `add` twice for one DOI | Check `library search` before adding; merge the dup via the `zotero` MCP (`duplicates find`) or delete it in Zotero. |
| Cache growing / disk pressure | Tier A artifacts kept forever by design | `research cache stats`; then `prune --max-mb N` (Tier A) or `clean` (Tier B). |
| `read` shows `命中缓存全文` but you want a fresh fetch | Cached `{stem}_fulltext.md` was reused | Re-run with `--force` (alias `--refresh`) to re-fetch + re-extract. |
| `add` blocked with `✗ 引用核验未通过` | Citation gate FAIL (≥2 sources hard-conflict on title/DOI/year) | Inspect with `research citecheck <doi> --json`; fix the mismatched field, or `--force` to override / `--no-verify` to skip. |
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
- **New output type**: add a template under `templates/` and a `cmd_*` in `research.py` that fills
  it via the existing `_dump_yaml` / `_fill_placeholders` helpers.
- **New PDF backend**: extend `pdf_extract.available_backends()` + `_pick_backend()` + `extract_pdf()`.

## 8. Verifying changes

```
.venv\Scripts\python.exe -m py_compile src/pysci/skills/literature_research/tools/<file>.py
.venv\Scripts\python.exe -m pysci.skills.literature_research.tools.research doctor
```
Then smoke-test the affected command (`search`/`read`/`get`/`add`/`citecheck`/`library`/`index`/`rag`). `doctor`
is the fastest way to confirm config, sources, backends, Playwright, Zotero, and the RAG layer are all wired
up. For `rag` specifically: `research rag status` (offline) then a small `research rag index --path <one .md>`
+ `research rag search '<q>' -k 3` is the cheapest end-to-end check (uses real SiliconFlow embedding).
