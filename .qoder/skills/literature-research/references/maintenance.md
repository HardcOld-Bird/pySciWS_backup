# Maintenance guide

The `research` CLI is a thin facade over 8 backend modules in
`src/pysci/skills/literature_research/tools/`. You rarely need to touch them; this guide is for when a
source, the PDF extractor, or a publisher page breaks, or when you want to extend the system.

> Reminder: for files under `src/pysci/skills/`, verify current content from disk (e.g. PowerShell
> `Get-Content -Encoding UTF8` / `Select-String`); an IDE/index cache may show stale content right
> after the directory is moved or renamed.

---

## 1. Architecture

```
research.py            ← CLI facade: doctor/search/read/get/add/library/index/cache (orchestration only)
  ├─ config.py         ← loads project-root .env; exposes `settings` + `http_session()`
  ├─ openalex_client   ← primary search + metadata + work_to_note_frontmatter()
  ├─ arxiv_client      ← preprints + download_pdf() + arxiv_to_note_frontmatter()
  ├─ wos_client        ← enrich_openalex_work(): official JIF/JCR/ESI (silent-fail)
  ├─ semantic_scholar_client ← enrich_from_s2(): TLDR (silent-fail; usually unavailable)
  ├─ zotero_bridge     ← ZoteroBridge: ping/list/search/get/create_item_from_metadata/add_note
  ├─ browser_fetch     ← Playwright: fetch_all/fetch_pdf/fetch_html + PublisherAdapter
  ├─ pdf_extract       ← extract_pdf(): MinerU cloud (primary) / pymupdf4llm (fallback)
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
`OPENALEX_EMAIL`, `WOS_API_KEY`, `SEMANTIC_SCHOLAR_API_KEY`, `ZOTERO_USER_ID`, `ZOTERO_API_KEY`,
`MINERU_TOKEN`, `PDF_EXTRACT_BACKEND`, `HTTP_TIMEOUT_SECONDS`, `HTTP_MAX_RETRIES`.

---

## 3. PDF → Markdown (`pdf_extract.py`)

- `available_backends()` returns what the environment can actually run: `mineru-cloud` when
  `MINERU_TOKEN` is set, `pymupdf4llm` when that module is importable.
- `extract_pdf(path, backend=None)` with `backend=None`/`auto` picks the best available
  (MinerU cloud first). It caches results in `cache/extracted/` (`use_cache`/`write_cache`).
- **MinerU cloud** is the primary: a VLM pipeline with OCR that renders equations as LaTeX and
  tables as HTML. It uploads the PDF to the MinerU Open API and polls the job (you'll see
  `MinerU running: k/N 页`). Typical cost ~3s/page.
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

An in-page JS routine (`_JS_EXTRACT`) pulls text from the first matching selector and collects
same-host PDF links.

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
`CACHE_AUTOCLEAN_INTERVAL_DAYS` (default 7), `CACHE_SOFT_LIMIT_MB` (default 2048). The whole
`cache/` dir (including `.autoclean_state.json` and `.by_url/`) is git-ignored.

---

## 6. Common failures → fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| `read` gets no PDF; `cloudflare=true` | Bot challenge | Re-run with `--headed`; complete the challenge in the window. |
| Full text empty / all navigation | Publisher changed markup | Update that adapter's `fulltext_selectors` (§4). |
| `institutional_access=false`, paywalled | No subscription via this network | Use a campus VPN, or `read` the arXiv id / OA copy instead. |
| MinerU 401 / job fails | Bad/expired `MINERU_TOKEN` or quota | Check `.env`; temporarily `--backend pymupdf4llm`. |
| Equations missing in output | Fell back to pymupdf4llm | Ensure `MINERU_TOKEN` is set; check `research doctor` lists `mineru-cloud`. |
| `[wos] enrichment skipped` / no JIF | WoS key pending approval or endpoint changed | Expected — degrades to OpenAlex estimate. No action unless you have a working key. |
| Anything Semantic Scholar | Campus network blocks S2 | **By design: ignore.** Do not troubleshoot or warn. |
| Zotero `不可达` | Desktop app closed / local API off | Start Zotero; Settings → Advanced → *Allow other applications*; or set Web API creds. |
| Paths wrong / `.env` not loaded | Project-root markers moved; `pysci.paths` can't find root | Check `_ROOT_MARKERS` in `src/pysci/paths.py` (§2); confirm with `research doctor`. |
| PowerShell mangles the command | Double quotes stripped / `&&` used | Single-quote multi-word args; chain with `;`. |
| `add` created a duplicate Zotero item | Ran `add` twice for one DOI | Check `library search` before adding; delete the dup in Zotero. |
| Cache growing / disk pressure | Tier A artifacts kept forever by design | `research cache stats`; then `prune --max-mb N` (Tier A) or `clean` (Tier B). |
| `read` shows `命中缓存全文` but you want a fresh fetch | Cached `{stem}_fulltext.md` was reused | Re-run with `--force` (alias `--refresh`) to re-fetch + re-extract. |

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
Then smoke-test the affected command (`search`/`read`/`get`/`add`/`library`/`index`). `doctor`
is the fastest way to confirm config, sources, backends, Playwright, and Zotero are all wired up.
