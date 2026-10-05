---
name: modeling3d
description: Parametric CAD for 3D printing (build123d, millimeter-exact, watertight-by-construction, trimesh acceptance gate) and concept rendering (headless Blender studio lighting, materials, cameras), plus Tripo cloud image-to-3D for complex reference subjects. Two pipelines kept strictly apart — print geometry never goes through Blender; renders come from versioned scripts, never from an interactive session alone. Interactive scene exploration runs through the blender-mcp addon (community MCP server). Drives a unified `model3d` CLI. Use when the user asks to design a 3D-printable part / fixture / sample holder with exact dimensions, validate an STL before slicing, render a concept image of a printed sample or experimental setup, turn a photograph into a 3D model, or scaffold a Blender scene script.
---

# 3D Modeling (modeling3d)

One CLI drives two pipelines that never mix:

- **Print pipeline (geometry-exact):** `build123d` parametric script (mm, OCCT B-rep, watertight
  by construction) → STL/STEP → **trimesh acceptance gate** (`watertight` / volume / bounding box).
  **Blender is never involved** — mesh modelers have no dimension-constraint system.
- **Render pipeline (aesthetics-first):** model from (a) a print-pipeline STL, (b) Tripo
  image-to-3D GLB, or (c) a Blender scene script → headless `blender -b -P` → PNG.

**Binding rule — deliverables come from versioned scripts.** The blender-mcp interactive channel
(addon + MCP server, both community-maintained) is for *exploration and preview only*; anything
worth delivering gets frozen back into a scene script and re-run headless. This is why MCP and
headless are not duplicate implementations.

## Invocation

From the **project root**: `uv run pysci-model3d <command>` (fallback: `uv run --no-sync python -m
pysci.skills.modeling3d.tools.model3d`). Prints Chinese — apply `.qoder/rules/basic.md` §3 or read
mojibake. Config: `MODELING3D_BLENDER` + `TRIPO_API_KEY` in project-root `.env`; `doctor` probes
everything (Blender exe/version, addon, libs, key validity).

## Commands

| Group | Commands | For |
|---|---|---|
| health | `doctor` | session start / anything broken |
| scaffold | `new <research> <slug> --kind print\|scene` | session dirs + script skeleton |
| print | `build <script.py> [--strict-volume] [--expect-bbox L,W,H]` | run CAD script → **auto acceptance gate** |
| gate | `check <mesh> [--expect-volume] [--expect-bbox] [--no-watertight]` | standalone STL/GLB validation |
| render | `render --mesh <stl/glb> [--out --engine --samples --bg]` · `render <scene.py>` | quick studio render · custom scene |
| cloud | `tripo <image\|prompt> [--out G.glb] [--pbr] [--dry-run]` | image/text → GLB (needs `TRIPO_API_KEY`) |
| inventory | `list <research>` | sessions + scripts |

Full flags, output paths, troubleshooting: [cli.md](references/cli.md). MCP interactive channel
setup + usage rules: [mcp.md](references/mcp.md).

## Workflow A — 3D-printable part (the core loop)

```
- [ ] 1. model3d new gain_ep sample_holder            # scaffolds script + session dirs
- [ ] 2. Edit src/pysci/research/gain_ep/models/sample_holder.py
         (all dims in mm, cite design source per parameter; fill §3 geometry asserts)
- [ ] 3. model3d build '<script>' --strict-volume --expect-bbox 40,20,8
         # runs script, then the gate: watertight + volume + bbox; non-zero exit = NOT deliverable
- [ ] 4. Deliver stl/sample_holder.stl to the user for slicing (STEP alongside for re-editing)
```

Gate failure triage: non-watertight → boolean leftovers/self-intersections, fix in the CAD script
(never patch the mesh); bbox mismatch → unit or parameter error. The gate is the **only** path to
delivery — no "looks fine, ship it".

## Workflow B — concept render of a printed sample

```
- [ ] 1. Complete Workflow A (or take an existing STL)
- [ ] 2. model3d render --mesh '<stl>' --out '<session>/renders/hero.png'   # studio template
- [ ] 3. Read the PNG; iterate: --bg / --engine CYCLES / --samples
- [ ] 4. Need custom materials/lighting/composition? model3d new <r> <slug> --kind scene,
         edit the scene script, model3d render '<scene.py>'
```

## Workflow C — photo → 3D (Tripo) → render

Complex real-world subjects (experimental setups, enclosures) that are impractical to model by hand:

```
- [ ] 1. model3d tripo photo.jpg --out '<session>/tripo/setup.glb' --pbr   # spends credits
         (no key yet? --dry-run prints the whole request plan for free)
- [ ] 2. model3d render --mesh '<glb>' --out '<session>/renders/concept.png'
- [ ] 3. Read the PNG. Cloud meshes are NOT print-ready (non-manifold, arbitrary scale) —
         for printing, remodel in build123d using the GLB as visual reference only.
```

Interactive alternative: the blender-mcp MCP tools can generate via Tripo inside a live Blender
session — same API key, exploration-only per the binding rule.

## Output locations

Research sessions live in `data/research/<n>_<name>/models/<slug>/` (`stl/ step/ renders/ notes.md`);
CAD & scene **scripts** live in `src/pysci/research/<name>/models/`. Skill-level assets (scaffold
templates, recipes, cache) in `data/skills/modeling3d/`. All artifact writes pass
`pysci.paths.assert_within_data`. The script↔CLI protocol: build scripts print
`[model3d] stl: <path>` / `[model3d] volume_mm3: <v>` markers that `build` parses — keep templates'
marker lines intact.

## When something breaks

`model3d doctor` first. The four that matter: **Blender NOT FOUND** → set `MODELING3D_BLENDER` in
`.env` · **engine TypeError in scene scripts** → Blender renamed EEVEE's engine id across versions;
the studio template has a fallback chain, custom scripts need it too · **Tripo 401/403** → key
missing/invalid, `tripo --dry-run` to verify plumbing · **gate says non-watertight** → fix the CAD
script, not the mesh. Everything else: [cli.md](references/cli.md#troubleshooting-in-full).

## Reference files

- [cli.md](references/cli.md) — every flag, acceptance-gate semantics, troubleshooting in full.
- [mcp.md](references/mcp.md) — blender-mcp interactive channel: setup, tools, the freeze-to-script rule.
