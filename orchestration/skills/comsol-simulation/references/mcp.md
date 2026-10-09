# The community `comsol` MCP in depth

Everything here concerns the **community `comsol` MCP**: transport choice, license behaviour,
server detachment, upstream pinning and upgrades, and the constraints measured while building
a model through it. `SKILL.md` keeps only the MCP-vs-CLI decision and the one `mcp ensure`
command — read this file before driving the MCP by hand or touching its installation.

## Why the MCP must run on HTTP transport (path C)

> **Single license — the `comsol` MCP must run on HTTP transport (path C).** On the pinned upstream
> (`0f6b2c58`) `main()` pre-starts COMSOL *before* `mcp.run()` under **stdio**, so Qoder merely
> launching the MCP takes the one license immediately, and that in-process JVM cannot be attached by
> the CLI. Under **sse / streamable-http** COMSOL instead starts *lazily* on the first tool call: the
> idle server holds **no** license, the handshake is instant, and the duplicate-instance problem below
> disappears. Measured on this box 2026-10-01: both transports serve all **103** tools, and the idle
> server's module list has `_jpype.pyd` but **no `jvm.dll`**.
>
> **Is the license actually free?** `comsol_status` reports `standalone: true`, i.e. MPh runs an
> **in-process JPype JVM** — there is *no* `comsolmphserver.exe` to spot, so "I see no COMSOL process"
> proves nothing. Check for **`jvm.dll` and nothing else**: `_jpype.pyd` is loaded at `import mph` time
> and sits in *every* comsol-mcp process whether or not COMSOL ever started, so counting it reports an
> idle, license-free server as busy. `simulation license` and `setup_comsol_mcp.ps1 -StatusOnly` both
> use the `jvm.dll` criterion.
>
> **Duplicate instances (stdio only).** The eager pre-start delays the MCP handshake by the whole ~30 s
> COMSOL boot, and Qoder rewrites `mcp.json` during startup — which can spawn a *second* `comsol-mcp`
> without reaping the first (observed 2026-10-01: two processes, each with its own JVM, ~320 MB apiece).
> **Fully quit Qoder** to clear them; a window reload is not enough. Path C removes the root cause
> because its handshake is instant.

## Step 0 internals — WMI detachment, lazy boot, restart, 403 Origin

> **Why WMI and not `Start-Process`.** A child of the agent terminal lives inside that terminal's job
> object and is **silently reclaimed** when the shell goes away — no traceback, no Windows Error
> Reporting entry (a 27-minute RAG build vanished this way). Measured A/B on 2026-10-01, two ~15-minute
> `ping` markers spawned by the *same* command: the `Start-Process` one was gone by the next command,
> the WMI one survived — its parent is `WmiPrvSE.exe`, so it never enters the terminal's job.
> `CREATE_BREAKAWAY_FROM_JOB` does not help (Qoder's job lacks `JOB_OBJECT_LIMIT_BREAKAWAY_OK`) and
> `schtasks` needs admin (measured: access denied). Note `IsProcessInJob` is **not** the test: both the
> WMI marker and Qoder's own long-lived stdio instance report `in_job=True` and both survive. The real
> discriminator is whether `WmiPrvSE.exe` appears in the ancestor chain.

> **The first COMSOL-touching tool call still pays the ~30 s JVM boot** — that is lazy start working as
> intended. If Qoder reports a request timeout on it, retry; the server is fine, and raising that
> server's Request Timeout to ~60 s means you rarely have to. Doc-only tools (`pdf_search`,
> `pdf_list_modules`, `docs_get`) never start COMSOL, so they stay cheap.
>
> **Nothing starts implicitly — call `comsol_start` yourself.** Under path C the server is *not*
> pre-started, and `model_create` / `model_load` answer `No active COMSOL session. Start with
> comsol_start first.` instead of booting one (measured 2026-10-01).
>
> **`mcp ensure --restart` invalidates Qoder's session.** Restarting replaces the server process, so the
> SSE `session_id` Qoder is holding is gone and every tool call fails with
> `51500 transport error: 404 Could not find session`. Reconnect the `comsol` MCP in Qoder afterwards.

> **403 `Invalid Origin header` — measured *not* to occur here, but know the fallback.** Qoder connected
> through the URL entry and called `pdf_list_modules` successfully (2026-10-01), so it either sends no
> `Origin` or a loopback one. Kept in case the host/port or upstream defaults ever change: FastMCP
> auto-enables DNS-rebinding protection whenever the host is `127.0.0.1` / `localhost` / `[::1]`, allowing
> only `http://127.0.0.1:*`, `http://localhost:*`, `http://[::1]:*` as `Origin`. Measured identically on
> both transports: **no `Origin` header, or a loopback one → 200; `vscode-file://vscode-app`, `null`, or
> any external origin → 403.** Diagnose with
> `scripts/comsol_mcp/probe_mcp_http.py <sse|streamable-http> <port>` — it prints the full matrix, then
> does a real MCP handshake using a doc-only tool so no license is touched. Upstream exposes **no** env
> switch for `transport_security`, so the only two fixes are: revert `mcp.json` to the `command` form
> (path A, and pay the license cost), or patch upstream `server.py` to pass an explicit
> `TransportSecuritySettings` that also allows Qoder's origin.

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

## Building a model through the MCP — five measured constraints

Every item below was measured end-to-end on 2026-10-01 by `scripts/comsol_mcp/smoke_mcp_chain.py`, and
each cost a license-holding round trip to learn. Read them before driving the MCP by hand.

1. **`study_create` must precede `mesh_create`.** The physics-controlled mesh derives its maximum element
   size from the study frequency; with no study in the model yet COMSOL refuses outright — *"the mesh is
   not used in any study"*.
2. **A Frequency step's frequency property is `plist`, not `freq`.** Upstream's own `study_create`
   docstring suggests `{"freq": "…"}` and COMSOL rejects it as an *unknown property*. The tool reports
   that in `property_errors` while still answering `success: true`, so **check that field**. Units default
   to Hz, so a bare number is enough.
3. **Domain features need an explicit `boundary_dimension`.** `physics_configure_acoustic_boundary`
   defaults to `getSDim() - 1`, i.e. *boundaries*, which is wrong for a domain-level node. A **Background
   Pressure Field** — the node the gain_ep study needs — is built in 2D with `boundary_dimension: 2` plus
   an explicit `boundary_selection`: upstream passes the type straight to `physics.create(tag, type, dim)`
   with no whitelist and echoes it back in `custom_condition_types`. Its knowledge tool
   `physics_get_acoustic_boundary_conditions` lists only 8 common boundary conditions and never mentions
   BPF, so a miss there proves nothing — and `simulation inspect node --path
   component(comp1).physics(acpr).feature(bpf1)` is still how you get the real answer (52 properties,
   allowed enums such as `PressureFieldType: PlaneWave|CylindricalWave|UserDefined`, selection counts).
4. **Upstream refuses an empty selection** (`Missing boundary selection or selection name.`) and never
   calls `selection().all()`. Worth knowing in the good sense: the silent zero-field trap in our own
   pitfalls notes — a BPF on an empty selection solves cleanly and returns `p == 0` everywhere — cannot
   happen through this path; it errors out instead.
5. **`physics_set_material` is an empty shell, so a model built from scratch cannot be meshed.** It only
   runs `material().create(tag, "Common")` + `label(name)` and never loads COMSOL's built-in library, so
   it answers `success: true` right next to the warning *"Material node has no physical properties"*, and
   no other tool can fill in `c` / `rho` afterwards. The physics-controlled mesh needs the sound speed, so
   `mesh_create` then fails with *"the material property `c` required by Pressure Acoustics 1 is not
   defined"*. **Let the MCP drive an existing `.mph`** (`model_load` → `param_set` → `study_solve` →
   `results_evaluate`) and create genuinely new models with a `build` recipe, which runs the Java API
   directly and can attach a real material — that split is exactly what the smoke test's two phases do.

> **Path C's coexistence promise is measured, not assumed.** With the MCP server running, a CLI
> `inspect tree` started its own JVM and read a model successfully (2026-10-01): the two really do share
> the box, because an *idle* server holds no license. Once the server has started COMSOL it holds the one
> license, so `mcp stop` before a CLI solve or opening the GUI.
