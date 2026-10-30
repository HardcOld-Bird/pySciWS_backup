# cli 分册 2/2

> 含小节：When something breaks
> 原 `cli.md` 超 8KB 预算拆分（2026-10-10，backlog 20261010-refs-split-8kb）；分册索引见同目录 `cli.md`，按需只读所需分册。

## When something breaks

1. `simulation doctor` — COMSOL discovery, mph version, MinerU token, indexed manuals.
2. Solve failures: read the `run solve` report's **log tail + Java stack** (that's where COMSOL's real
   error lives), and `inspect tree` to confirm study/solver tags.
3. Empty/blank exported PNG: `export image` now **self-checks** the axis-frame interior and prints
   `[warn] … 近空白` — typical cause: plot group has no dataset. `run solve` **auto-rebinds** every
   plot group to the fresh dataset, so a solve-then-export in separate invocations is safe; if you
   bind by hand use `node set --path result(pg) --set data=dset`. See sidecar `interior.blank`.
   Also check `sourceobject`.
4. Unknown node type/property name: **never guess and never write throwaway probe scripts** —
   run `inspect node --path …` for ground truth, and consult
   `data/skills/comsol_simulation/knowledge/java_api_pitfalls.md` (verified type/property names,
   default-value traps, naming collisions, hygiene conventions).
5. `comsol_start` fails with `Could not locate Comsol <version> installation.` → the version string
   reaching upstream is a **full** version (`6.4.0`) while mph's discovery does an exact match against its
   **short backend name** (`6.4`; `6.4a` when patch > 0). This is the most deceptive failure on path C:
   the HTTP layer looks perfectly healthy — 200 ready, 103 tools listed, doc-only tools answering — while
   *every* COMSOL-touching tool fails. `mcp_server.mph_version_short_name()` normalises it when
   regenerating `runs/start_comsol_mcp.cmd`; check that launcher's `set COMSOL_MCP_VERSION=` line holds a
   short name, then `mcp ensure --restart` (and reconnect in Qoder — see
   [mcp.md](mcp.md#step-0-internals--wmi-detachment-lazy-boot-restart-403-origin)).
