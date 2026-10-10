# maintenance 分册 4/11

> 含小节：5. Cache governance (`cache_manager.py`)
> 原 `maintenance.md` 超 8KB 预算拆分（2026-10-10，backlog 20261010-refs-split-8kb）；分册索引见同目录 `maintenance.md`，按需只读所需分册。

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
