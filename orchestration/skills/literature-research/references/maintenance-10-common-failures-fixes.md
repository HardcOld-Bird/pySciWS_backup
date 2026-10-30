# maintenance 分册 10/11

> 含小节：6. Common failures → fixes
> 原 `maintenance.md` 超 8KB 预算拆分（2026-10-10，backlog 20261010-refs-split-8kb）；分册索引见同目录 `maintenance.md`，按需只读所需分册。

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
| `read <id>` on an existing note wrote a **second** file | `_merge_note` derives the path from the *fresh* `short_title`, and `_make_short_title` output drifted from the stored value | Closed at the source (2026-10-04): `_find_note_by_identity` redirects the write to the note carrying the same `doi` / `openalex_id` / `arxiv_id` and prints both names (§5f). It can still happen if the note has **none** of those three keys — give it one. No command *reports* the drift (`index --fix` compares against the note's *own stored* frontmatter, `--check` only diffs `INDEX.md`); compare `research get <id>`'s `short_title` with the note's. |
| `add` blocked with `✗ 引用核验未通过` | Citation gate FAIL (≥2 sources hard-conflict on title/DOI/year) | Inspect with `research citecheck <doi> --json`; fix the mismatched field, or `--allow-fail` to override / `--no-verify` to skip. |
| `citecheck` reports `?NOT_FOUND` for a real paper | All three sources missed it (typo'd DOI, very new, or offline) | Check the DOI/id; a lone `openalex=不可达`/`crossref=不可达` is a network blip (downgraded, not a FAIL) — re-run. |
| `citecheck` first-author `△WARN` on an accented name | Cross-source transliteration/abbreviation noise | Expected — author surname is a soft signal and never blocks; title/DOI/year are the hard signals. |
| `rag search`/`ask` errors "未配置 SILICONFLOW_API_KEY" | `pqa_ready` false (no key in `.env`) | Set `SILICONFLOW_API_KEY` (+ `SILICONFLOW_BASE_URL`); confirm with `research rag status`. |
| `rag search` says "no index" / `先运行 research rag index` | Index never built, or paperqa version changed / pickle corrupt | Run `research rag index` (add `--rebuild` to force). `cache/rag/` is rebuildable + git-ignored. |
| `rag index` prints `语料规范化版本变更（N → M），转为全量重建` | `corpus_clean.CLEANER_VERSION` was bumped, or `--no-clean` was flipped | Expected and self-healing — the old index embedded a differently-cleaned corpus, and the mtime-based incremental path could never notice. Let the rebuild finish. |
| A `rag search` hit's text has no image lines / reference list | `corpus_clean` removed them from the embedded **copy** | Expected. `source_path` still points at the untouched original under `cache/extracted/`, which has everything. Pass `--no-clean` to `rag index` if you want the raw text embedded instead. |
| `rag ask` prints a degrade notice + returns search results | Free **and** paid LLM both failed (quota / 503 / network) | By design — `ask` never blocks; use the returned `search` chunks or retry later. Check the `SILICONFLOW_API_KEY` quota. |
| `rag` embedding `KeyError` on max_input_tokens | A model was registered without `max_input_tokens` | Ensure `_register_litellm_models` declares it for both the bare and `openai/`-prefixed name (§5d). |

---
