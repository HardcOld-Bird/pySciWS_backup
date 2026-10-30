# mcp 分册 2/3

> 含小节：Where the community MCP lives, and how it is pinned
> 原 `mcp.md` 超 8KB 预算拆分（2026-10-10，backlog 20261010-refs-split-8kb）；分册索引见同目录 `mcp.md`，按需只读所需分册。

## Where the community MCP lives, and how it is pinned

Deliberately **outside this repo**, as a sibling directory:
`D:\XXXIIIGGG\projects\pySci\COMSOL_Multiphysics_MCP` (override: `setup_comsol_mcp.ps1 -RepoDir`).

**Not a git submodule — on purpose.** Upstream tracks ~700 MB of binaries (`pdf/` 535 MB,
`comsol_models/` 122 MB, `knowledge_base/chroma.sqlite3`, 42 `.pyc` — 269 tracked files in all), so
vendoring would drag ~2.2 GB into this tree *and* leave the submodule permanently dirty: Chroma rewrites
`chroma.sqlite3` on every query and the `.pyc` regenerate on every run. Upstream did add a `.gitignore`
(`a9de20e`), but never `git rm --cached` those paths — and `.gitignore` has no effect on already-tracked
files — so the dirt is still there on the current pin. Silencing it with `ignore = all` would also hide
real upstream changes. Its 1 GB `.venv` (Python 3.12) is not relocatable on Windows either. Note that
Python-level venv conflict is *not* the issue: the two venvs are hermetic and this project never imports
upstream code (Qoder launches it as a separate process). The real cost is IDE/lint bleed, so it stays out
of the tree.

Reproducibility is pinned by tracked artifacts inside this repo instead:

| Artifact | Role |
|---|---|
| `scripts/comsol_mcp/UPSTREAM.lock.json` | **source of truth** — upstream URLs, pinned commit, venv Python, patch state, resolved `command`, verification checks, and the full no-submodule rationale |
| `scripts/comsol_mcp/setup_comsol_mcp.ps1` | idempotent installer; its `-Commit` default must match the lock. `-StatusOnly` is a read-only check that lock / `-Commit` / actual `HEAD` all agree, plus the true RAG index count and whether a COMSOL JVM is holding the license right now (incl. duplicate-instance detection) |
| `scripts/comsol_mcp/probe_kb.py` | **true** RAG index status. Upstream's `build_knowledge_base.py --status` always prints `Documents: 0` (it calls `get_stats()` without `initialize()`); run this with the *community* venv's python instead — no model, no JVM, no license. It pages Chroma, so it survives indexes past 32766 chunks |
| `scripts/comsol_mcp/mcp_servers.template.json` | shape of the Qoder `mcp.json` entry |
| `scripts/comsol_mcp/probe_mcp_http.py` | path C diagnostics: the FastMCP **Origin** allow-list matrix plus a real MCP handshake over `sse` / `streamable-http`. Deliberately calls only `pdf_list_modules`, so it proves the transport **without** starting COMSOL or touching the license. Run it with the *community* venv's python |
| `scripts/comsol_mcp/smoke_mcp_chain.py` | end-to-end proof of the modelling chain over path C: phase A builds a 2D pressure-acoustics model from scratch up to a **Background Pressure Field**, phase B loads a real research model and runs `study_solve` → `results_evaluate`. Judges by a *non-zero* field, not by `success: true` (exit 0 = both, 1 = phase A, 2 = phase B). **Takes the license**; run it with the *community* venv's python, then `mcp stop` |

> **`pdf_search` coverage.** The community MCP's RAG index is now **fully built** — 35,923 chunks over all
> **52** manual modules / 108 PDFs (rebuilt 2026-10-01 in ~27 min; before that it was a 3669-chunk,
> 4-module subset from the installer's `-RagLimit 8`). A `pdf_search` miss is therefore *meaningful* now.
> Still prefer this skill's own `simulation docs search` for anything equation-heavy: it reads
> MinerU-converted Markdown (equations → LaTeX), whereas `pdf_search` chunks are raw PDF text — usable
> for body prose, but headers come out letter-spaced (`C H A P T E R 2 : P R E S S U R E …`).
>
> **Never trust `pdf_search_status`'s module list.** Upstream enumerates modules with an *unbounded*
> `collection.get()`, and Chroma binds one SQL variable per row — past SQLite's 32766 limit that raises
> `too many SQL variables`, which a bare `except: pass` swallows. At 35,923 chunks it reports the correct
> `count` right next to `modules: []` / `module_count: 0`. Read that `0` as *"upstream couldn't
> enumerate"*, **not** *"nothing is indexed"*. `pdf_search` itself is unaffected (bounded by `n_results`),
> and so is `pdf_list_modules` (filesystem-based). `probe_kb.py` pages, so it tells the truth.
>
> **The first doc lookup may time out.** Inside a fresh MCP process the first `pdf_search` /
> `pdf_search_status` loads SentenceTransformer, which can exceed Qoder's default request timeout
> (observed live: a `40504` timeout while the server kept working — its python working set grew
> 322 → 526 MB). Warm, the same call returns in seconds. So a timeout on the *first* lookup is **not**
> a broken index: just retry, or raise that server's Request Timeout.

Browse upstream code in-IDE via PyCharm **File → Open → Attach** on that folder (zero git / lint / pytest
implications). To upgrade: bump `pinned_commit` in the lock **and** `-Commit` in the installer to the same
SHA, make that SHA reachable, then re-run the installer and confirm with `-StatusOnly`. Behind a proxy a
targeted `git -C <repo_dir> fetch --depth 1 <canonical_url> <SHA>` takes ~2 s and needs no `--unshallow`
(the big blobs rarely change, so they're already local); add `-SkipInstall -SkipRag` when `pyproject.toml`
and `knowledge_base/` are untouched. **Then restart the `comsol` MCP in Qoder** — a running process keeps
the old code in memory. Under **path C the only reliable proof of a reload is calling a tool**: Qoder no
longer spawns the process, so `SERVER_METADATA.json`'s cached `toolCount` stops tracking reality — after
the switch to the URL entry it read `93` (a stale pre-upgrade value) while the `tools/*.json` on disk, the
tool list injected into the session, and a live `pdf_list_modules` call all agreed on **103**. (Under
path A `toolCount` *is* the proof — 93 → 103 on `0f6b2c58` — but it only flips once Qoder has actually
**replaced** the process: measured ~5 min behind a window reload here, and that reload also left an orphan
behind. Path A's faster behavioural check was `comsol_status` returning `connected: true` **and**
`standalone: true` from cold — only the pre-starting pin does that; under path C both stay `false` until
the first COMSOL-touching call.) Prefer a full Qoder quit over a window reload. **Qoder CN has no tray
icon**, so a full quit means ending the `QoderCN` process in Task Manager: that reclaims the stdio
`comsol-mcp` Qoder spawned (same job object) and frees the license, but it does **not** touch the
WMI-spawned path C server — measured, the server outlived the `QoderCN` kill.
