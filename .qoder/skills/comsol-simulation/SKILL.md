---
name: comsol-simulation
description: Create, edit, debug, and evaluate COMSOL Multiphysics (.mph) simulations headlessly and fully automatically — geometry, materials, physics (pressure acoustics), mesh, studies, solving, parameter sweeps, result export, and offscreen rendering. Drives a unified `simulation` CLI over the mph/JPype Java-API bridge, consults MinerU-converted local COMSOL manuals via an FTS5 index, and "sees" results through COMSOL PNG export plus pyvista offscreen rendering. Use when the user asks to build or modify a COMSOL model, run or re-run a simulation, sweep parameters, export plots/fields/meshes, render or inspect a result field, check mesh quality or convergence, or look up COMSOL API/physics documentation.
---

# COMSOL Simulation

A single CLI drives the whole closed loop: **docs → build → run → export → render/post → validate**.

The backend lives in `src/pysci/skills/comsol_simulation/` (tools: config/session/inspect/build/run/
export/postprocess/docs + a `simulation` facade). **Treat it as a black box** and drive everything
through the `simulation` CLI below. Only open the code when maintaining it.

## Invocation

Run from the **project root**. The skill installs a console script `pysci-simulation`:

```
uv run pysci-simulation <command> [options]
```

Below, `simulation …` is shorthand for `uv run pysci-simulation …`. (Fallback if the script
isn't installed: `uv run python -m pysci.skills.comsol_simulation.tools.simulation …`.)

> **PowerShell rule (critical):** wrap multi-word arguments in **single quotes**; use `;` (never
> `&&`) to chain commands.

> **Cost rule:** every command that touches a live model (`inspect tree/params/inventory`,
> `run solve`, `export …`, `build apply`) boots a JVM (~30s) and takes a license slot. Batch your
> questions: prefer ONE session doing several things over many single-purpose invocations. The CLI
> opens a standalone session per invocation and releases it on exit.

## Commands at a glance

| Command | Use when | Key output |
|---|---|---|
| `doctor` | Session start / anything broken | Config + COMSOL discovery + indexed manuals |
| `diagnose --mph M` | One-shot health check of a model | doctor + tree + inventory + dataset/plotgroup bindings + pitfalls checklist (single JVM) |
| `inspect tree --mph M` | Understanding a model's structure | Compact model tree (params/geom/physics/mesh/study/results) |
| `inspect params --mph M` | Reading/editing global parameters | `name = value # descr` list |
| `inspect inventory --mph M` | Programmatic node tags | tag lists per subsystem |
| `inspect node --mph M --path P` | Any node's type/props/selection unknown | type, prop values, allowed enums, selection counts |
| `node set --mph M --path P --set k=v [--set …] [--save O.mph]` | Setting a node property with zero custom code | per-key ok/fail report (symmetric to `inspect node`) |
| `inspect java F.java` | GUI-exported `.java` fallback | Structured summary (no model load) |
| `run solve --mph M [--study S] [--set-param k=v] [--clear]` | Running / re-running | Solve report; **auto-rebinds** every plot group to the fresh dataset |
| `run batch IN.mph --output OUT.mph` | Heavy/long solves off-session | comsolbatch result + stdout/stderr |
| `export image --mph M --plotgroup PG --out O.png [--color-range MIN MAX] [--polar-rmax R] [--geom-bbox X0 X1 Y0 Y1] [--extent … --clean]` | "Seeing" a result plot | PNG + `<O>.sidecar.json` (axis-frame px box, `extent_recovered`/`scale_applied`, blank self-check warn) |
| `export data --mph M --source D --out O.csv [--expr E]` | Numeric field data | CSV/VTK (add `--expr` for field values) |
| `export table --mph M --table T --out O.csv` | Probe/global tables | CSV |
| `render F.vtk --out O.png [--scalars S]` | Offscreen look at mesh/field | pyvista PNG (no COMSOL needed) |
| `post stats --file F.vtk --scalars S` / `--csv C` | Field statistics | min/max/mean/rms or CSV columns |
| `post quality F.vtk` | Mesh quality | cell/point counts + quality metrics |
| `post framebox --image P [--sidecar S]` | pixel↔data mapping / cropping a render | axis-frame pixel box (optionally merged into sidecar) |
| `docs search '<q>'` / `docs read DOC [--heading H]` | Looking up COMSOL API/physics | FTS5 hits / faithful Markdown |
| `docs convert-all` / `docs index` | (Re)building manual knowledge | docs/*.md + FTS5 index |
| `build recipes` / `build apply --mph M --recipe R [--param k=v] [--save O.mph]` | Parametric modeling | recipe list / applied model |
| `server start/stop/status` | Reusing one JVM across many CLI calls | persistent `comsolmphserver` + `--connect-port` (see Sessions below) |

Global flag: prefix any live command with `--connect-port PORT` to run it against a persistent
server instead of booting a fresh standalone JVM (see *Sessions, memory, and licensing*).

Run `simulation <command> -h` for full options.

## Standard closed-loop workflow

```
- [ ] 1. simulation doctor                          # COMSOL + manuals ready?
- [ ] 2. simulation inspect tree --mph M.mph        # read the model
- [ ] 3. (edit) simulation build apply / run solve --set-param k=v
- [ ] 4. simulation run solve --mph M.mph --study std1 --clear
- [ ] 5. simulation export image --mph M.mph --plotgroup pg10 --out out.png   # SEE it
- [ ] 6. simulation export data  --mph M.mph --source pg10 --out out.csv --expr acpr.p_t
- [ ] 7. simulation render / post stats / post quality                        # programmatic eyes
- [ ] 8. validate: convergence / conservation / vs analytic (postprocess.py)  # TRUST it
```

Read the exported PNG with the Read tool to actually *see* the field. For numeric evaluation use
`export data --expr …` + `post stats`, or the validators in `tools/postprocess.py`
(`compare_to_reference`, `check_conservation`, `convergence_order`, `validate_field`).

## Documentation pipeline (knowledge)

Local COMSOL PDFs are converted (MinerU cloud, equations→LaTeX) to `data/skills/comsol_simulation/docs/*.md`
and indexed into an SQLite **FTS5** index (`cache/doc_index.db`).

```
simulation docs search 'perfectly matched layer'      # → manual § section [pages] + snippet
simulation docs read COMSOL_ProgrammingReferenceManual --heading '<heading>'
simulation docs convert-all                            # (re)convert the 5 core manuals, background-friendly
simulation docs index                                  # rebuild FTS5 after new conversions
```

Core manuals: ProgrammingReferenceManual (Java API commands), ApplicationProgrammingGuide,
AcousticsModuleUsersGuide, PostprocessingAndVisualization, ReferenceManual. Conversion of big
manuals auto-chunks at ≤199 pages (MinerU 200-page cap) — run `convert-all` in the background.

## Sessions, memory, and licensing

- **Standalone by default**: each CLI invocation boots a JVM, works, and exits (releases ~GBs of RAM
  and the license). Good for the ~5GB free-RAM box.
- **Persistent server (session reuse)**: for many back-to-back live calls, `simulation server start`
  launches a `comsolmphserver` once and records its pid/port in `runs/comsol_server.json`. Then run
  each command with `--connect-port <PORT>` to reuse the *same* JVM (skips the ~30s cold start and
  license churn). Finish with `simulation server stop` to kill it and clear the state file.
  `server status` reports running/pid/port. In scripts, the same reuse is available via the
  `session(mode="connect"|"server"|"standalone")` context manager.
- Solve threads capped by `COMSOL_MAX_CORES` (.env, default 4); disk tempdir under `runs/tmp`.
- Single license: **do not** run a persistent server AND an interactive COMSOL GUI at the same time.
- For tight edit→solve loops, prefer ONE invocation doing many steps, a persistent server, or
  `run batch` for heavy jobs.

## GUI fallback & Blender reservation

- Pure-API modeling can fail on exotic geometry. Fallback: build it once in the COMSOL **GUI**, export
  the model as `.java`, then `simulation inspect java F.java` to read the exact create/set sequence and
  transcribe it into a `build.py` recipe (the `.java` is the Rosetta stone).
- **Blender reservation**: `build.import_external_geomesh()` imports STL/PLY/VTK/STEP into a geometry
  sequence. When Blender modeling lands, export STL/PLY from Blender and import via this primitive.

## Output locations

| Path | Contents |
|---|---|
| `data/skills/comsol_simulation/docs/` | MinerU-converted manual Markdown |
| `data/skills/comsol_simulation/cache/doc_index.db` | FTS5 section index |
| `data/skills/comsol_simulation/recipes/` `templates/` | saved recipes / seed `.mph` templates |
| `data/skills/comsol_simulation/knowledge/` | Java→Python notes, pitfalls |
| `data/skills/comsol_simulation/runs/` | default export/log landing zone (+ `runs/tmp` tempdir) |

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
