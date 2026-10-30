# maintenance 分册 1/11

> 含小节：概览；1. Architecture；2. Config & paths (`config.py`)
> 原 `maintenance.md` 超 8KB 预算拆分（2026-10-10，backlog 20261010-refs-split-8kb）；分册索引见同目录 `maintenance.md`，按需只读所需分册。

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
