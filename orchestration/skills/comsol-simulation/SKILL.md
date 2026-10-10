---
name: comsol-simulation
description: Create, edit, debug, and evaluate COMSOL Multiphysics (.mph) simulations headlessly and fully automatically — geometry, materials, physics (pressure acoustics), mesh, studies, solving, parameter sweeps, result export, and offscreen rendering. Routine generic driving (load/param/solve/evaluate) goes through the registered community `comsol` MCP, which this skill supervises (`mcp ensure`) so it stays reachable while idle without holding the single COMSOL license; the skill's own `simulation` CLI (mph/JPype Java-API bridge) is the differentiated layer — publication-grade export, pyvista rendering, physics validation, node introspection, recipes, persistent server — and the fallback when the MCP is unavailable. It consults MinerU-converted local COMSOL manuals via an FTS5 index and "sees" results through COMSOL PNG export plus pyvista offscreen rendering. Use when the user asks to build or modify a COMSOL model, run or re-run a simulation, sweep parameters, export plots/fields/meshes, render or inspect a result field, check mesh quality or convergence, or look up COMSOL API/physics documentation.
---

# COMSOL Simulation

**Two drivers**: the community **`comsol` MCP** (*primary* for generic driving — 103 tools)
and this skill's **`simulation` CLI** (the *differentiated* layer the MCP lacks, **and** the fallback).
**MCP vs CLI** below decides between them; the rest of this doc covers the CLI, which drives the closed
loop **docs → build → run → export → render/post → validate**. Its backend in
`src/pysci/skills/comsol_simulation/` is a **black box** — open the code only when maintaining it.

## MCP vs CLI — which to use

| Need | Driver |
|---|---|
| Generic driving: load/create/inspect, params, geometry/physics/mesh, study solve, `results_evaluate` / `export_data` / `export_image`, `pdf_search` | **community `comsol` MCP** — one warm server, no per-call JVM boot or license churn |
| Publication `export image` (color-range / polar-rmax / geom-bbox + `.sidecar.json` + blank self-check); `render`; `post *`; physics validation; `inspect node` / `node set` / `java`; `diagnose`; `build recipes/apply`; `server` + `--connect-port` | **`pysci-simulation` CLI** — the MCP has none of these |
| MCP unavailable, or you want one self-contained script | **CLI** — fully self-sufficient (does its own load/solve/export internally) |

They coexist on this **single-license** box only because an *idle* MCP server holds no license — which
requires HTTP transport (path C). Once it has started COMSOL it holds the one license, so `mcp stop`
before a CLI solve or opening the GUI. Transport choice, license probing (`jvm.dll`, **not** `_jpype.pyd`),
duplicate instances: [references/mcp.md](references/mcp.md).

## Step 0 — bring the MCP server up (every session, before any MCP tool call)

Under path C **Qoder only connects to a URL; it no longer starts the server.** One idempotent command
owns that — run it first, always:

```
uv run pysci-simulation mcp ensure
```

Already running → reuses it; otherwise spawns one **detached through WMI**, waits for the endpoint,
prints the URL. These output lines decide everything:

| Line | Meaning / what to do |
|---|---|
| `running : True` | Go ahead and call `comsol` MCP tools. |
| `detached : 已脱离终端（派生方式 wmi）` | Outlives this session. If it says `⚠ 未脱离`, re-run `mcp ensure --restart`. |
| `license : 空闲` | CLI standalone solves and the COMSOL GUI are both safe. |
| `license : 已被占用 -- PID …` | Something holds the only license: `mcp stop` if ours, else stop that PID. |

Companions: `mcp status` (read-only), `mcp stop`, `license`, `--json`. **Never hand-write spawn/kill
logic.** WMI detachment (a terminal child is silently reclaimed with the shell), the ~30 s lazy JVM boot,
explicit `comsol_start`, `--restart` invalidating Qoder's session, and the 403 `Invalid Origin` fallback:
[references/mcp-01-overview.md](references/mcp-01-overview.md).

**Driving the MCP by hand? Read the
[five measured constraints](references/mcp-03-building-a-model-through-the-m.md)
first** — each cost a license-holding round trip to learn.

## Invocation

From the **project root**: `uv run pysci-simulation <command> [options]`, abbreviated below to
`simulation …`. (Not installed? `python -m pysci.skills.comsol_simulation.tools.simulation`.)

> **PowerShell:** this CLI prints Chinese — apply `.qoder/rules/basic.md` §3 or you read mojibake.
>
> **Cost rule:** every command touching a live model (`inspect`, `run solve`, `export`, `build apply`)
> boots a JVM (~30 s) and takes a license slot. Batch questions: prefer ONE session doing several things
> over many single-purpose calls.

## Commands

| Group | Commands |
|---|---|
| health | `doctor` · `diagnose --mph M` · `license` |
| MCP | `mcp ensure` · `mcp status` · `mcp stop` |
| read | `inspect tree` · `params` · `inventory` · `node --path P` · `java F.java` |
| write | `node set --path P --set k=v [--save O.mph]` · `build recipes` · `build apply --recipe R` |
| run | `run solve [--study S] [--set-param k=v] [--clear]` · `run batch IN.mph --output OUT.mph` |
| see | `export image --plotgroup PG --out O.png` · `export data --source D [--expr E]` · `export table` · `render F.vtk` · `post stats`/`quality`/`framebox` |
| docs | `docs search '<q>'` · `docs read DOC [--heading H]` · `docs convert-all` · `docs index` |
| server | `server start` · `stop` · `status` |

Worth memorising: `run solve` **auto-rebinds** every plot group to the fresh dataset (so solve-then-export
across invocations is safe), and `inspect node` is the only ground truth for a node's type /
properties / allowed enums. Prefix any live command with `--connect-port PORT` to reuse a persistent
server instead of booting a JVM. Exact flags and output fields:
[references/cli.md](references/cli.md), or `simulation <command> -h`.

## Standard closed-loop workflow

Steps 3/5/7 can instead go through the **`comsol` MCP** when you don't need the CLI's differentiated
export / render / validate.

1. `mcp ensure` (only if using the MCP) → 2. `doctor` → 3. `inspect tree --mph M.mph` → 4. edit via
   `build apply` / `node set` / `--set-param k=v` → 5. `run solve --mph M.mph --study std1 --clear` →
   6. `export image --plotgroup pg10 --out out.png`, then **Read that PNG** to actually see the field →
   7. `export data --source pg10 --expr acpr.p_t`, plus `render` / `post stats` / `post quality` as
   programmatic eyes → 8. **trust it**: validate convergence / conservation / vs analytic with the
   `tools/postprocess.py` validators (`compare_to_reference`, `check_conservation`, `convergence_order`,
   `validate_field`).

## Output locations

All under `data/skills/comsol_simulation/` (`paths.COMSOL_ROOT`): `docs/` MinerU manual Markdown ·
`cache/doc_index.db` FTS5 index · `recipes/` + `templates/` · `knowledge/` Java→Python notes and
pitfalls · `runs/` exports/logs (+ `runs/tmp` tempdir).

## When something breaks

`simulation doctor` first. Then: **solve failure** → the `run solve` log tail + Java stack; **blank PNG**
→ sidecar `interior.blank` (usual cause: plot group with no dataset); **unknown node type/property** →
never guess, never write probe scripts — `inspect node --path …` plus
`knowledge/java_api_pitfalls.md`. All failure modes:
[cli-02](references/cli-02-when-something-breaks.md).

## Reference files

（`*`=分册索引，按需读单册）

- [references/mcp.md](references/mcp.md) * — the community MCP: transport, license probing,
  Step 0 internals, pinning, the five modelling constraints.
- [references/cli.md](references/cli.md) * — the CLI: flag table, knowledge pipeline,
  sessions / licensing, troubleshooting.
