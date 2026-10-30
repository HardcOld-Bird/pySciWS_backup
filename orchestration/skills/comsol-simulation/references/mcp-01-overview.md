# mcp 分册 1/3

> 含小节：概览；Why the MCP must run on HTTP transport (path C)；Step 0 internals — WMI detachment, lazy boot, restart, 403 Origin
> 原 `mcp.md` 超 8KB 预算拆分（2026-10-10，backlog 20261010-refs-split-8kb）；分册索引见同目录 `mcp.md`，按需只读所需分册。

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
