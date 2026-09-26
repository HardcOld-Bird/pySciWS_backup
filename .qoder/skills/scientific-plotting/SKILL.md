---
name: scientific-plotting
description: Produce publication-grade scientific figures (multi-panel, nested, data-driven or external-media panels) for journal submission. Drives a unified `figures` CLI that scaffolds a per-figure production pipeline under a research's `article/figures/` dir, applies journal style presets (APS/PRL Arial 8pt 85/170mm; Nature Helvetica 7pt 89/183mm) with colorblind-safe palettes, exports EPS/PDF/SVG deliverables plus a PNG preview, audits publication compliance (width/fontsize/font-embedding/panel labels/colorblind), and closes the visual loop by letting the agent Read the PNG preview to iterate. Use when the user asks to draw / create / revise a paper figure, plot simulation or experiment data into a submission figure, restyle a figure for a different journal, export EPS/SVG, or check a figure against journal specs.
---

# Scientific Plotting

One CLI drives the whole figure workflow: **scaffold → draw → export → audit → visually verify**.

The backend lives in `src/pysci/skills/scientific_plotting/` (style / palette / layout / export /
runner / audit / scaffold + a `figures` facade). **Treat it as a black box** and drive everything
through the `figures` CLI below. Only open the code when maintaining it.

The key design split: **this skill owns the publication *contract*** (widths, fonts, italics,
colorblind palettes, export, self-check); **each paper figure's actual drawing lives as a small
pipeline script** (`build_figure`) under the research's asset dir, so figures iterate in place.

## Invocation

Run from the **project root**. The skill installs a console script `pysci-figures`:

```
uv run pysci-figures <command> [options]
```

Below, `figures …` is shorthand for `uv run pysci-figures …`. (Fallback if the script isn't
installed / offline: `uv run --no-sync python -m pysci.skills.scientific_plotting.tools.figures …`.)

> **PowerShell rule (critical):** wrap multi-word arguments in **single quotes**; use `;`
> (never `&&`) to chain commands.

## Capability boundary (read this first)

**Available now:**
- Journal style presets: `aps` (Arial 8pt, 85/170 mm) and `nature` (Helvetica 7pt, 89/183 mm,
  with automatic fallback to TeX Gyre Heros / Arial when Helvetica isn't installed).
- Colorblind-safe palettes (Okabe-Ito default, Tol bright/vibrant).
- Multi-panel & nested layouts with automatic `(a)(b)(c)` panel labels.
- Export: EPS / PDF / SVG (text kept editable via fonttype=42 / svg fonttype=none) + PNG.
- **PNG preview** for the agent's visual loop (EPS/SVG can't be Read directly).
- Publication audit: width vs design width, min fontsize, font embedding, panel labels,
  colorblind distinguishability.
- Per-figure pipeline scaffolding + discovery + run.

**Not yet (v2):** rasterizing externally-produced SVG (Blender renders, hand-edited SVG) for
preview; journal-specific presets beyond aps/nature. External media still works as an
`imshow` panel fed by a file path.

## Commands at a glance

| Command | Use when | Key output |
|---|---|---|
| `doctor` | Session start, or anything seems broken | config + backends + font availability |
| `styles` | Choosing a journal look / palette | preset + palette listing |
| `new <research> <slug>` | Starting a figure | scaffolded pipeline dir under `article/figures/` |
| `build <figdir>` | Producing deliverables | EPS/PDF/SVG/PNG + `_preview.png` in `out/` |
| `preview <figdir>` | Fast visual iteration (no deliverables) | `_preview.png` only |
| `audit <figdir>` | Checking publication compliance | PASS/FAIL report |
| `list <research>` | Seeing existing figure pipelines | pipeline inventory |

Run `figures <command> -h` for the full option list.

## Quick start: zero → submission-grade figure

```
- [ ] 1. figures doctor                                  # confirm backends + fonts
- [ ] 2. figures new gain_ep fig1_ep_band --style aps --width double --template multi_panel
- [ ] 3. Edit article/figures/fig1_ep_band/fig1_ep_band.py   # replace sample data with real data
- [ ] 4. figures build 'data/research/1_gain_ep/article/figures/fig1_ep_band'
- [ ] 5. Read the printed out/fig1_ep_band_preview.png   # VISUAL CHECK — iterate on step 3-4
- [ ] 6. figures audit '<figdir>' --panels                # final compliance gate
```

**Always Read the `_preview.png` before declaring done** — a figure can export cleanly yet look
wrong (overlapping labels, cramped panels, wrong colors). The preview is your eyes.

## The per-figure pipeline

Each figure is a directory under `data/research/<n>_<name>/article/figures/<slug>/`:

```
<slug>/
├── <slug>.py        # pipeline: build_figure(style=None, research_dir=None, **kw) -> Figure
├── notes.md         # iteration log (date / need / result)
└── out/             # fig.eps (submit), .pdf, .svg (manual tweak), _preview.png (agent eyes)
```

`build_figure` is called **inside** the journal `style_context`, so `plt.subplots()` /
`layout.grid()` already inherit the correct figsize, fonts, sizes and colorblind cycle. Declare
only the kwargs you need (`style`, `research_dir`); extras are ignored safely.

The three common iteration needs map to:
- **change data, keep plot** → edit the data-loading part of `build_figure`.
- **keep data, change plot** → edit the plotting calls (line↔scatter etc.).
- **keep content, change journal style** → re-run with `--style nature` (no code change).

A `STYLE.yaml` at the figures root (or per-figure) pins the default preset/width; CLI flags win.

## Visual closed loop

`build` / `preview` always emit `out/<stem>_preview.png`. **Read it** to see the figure, then
edit `<slug>.py` and re-run. This is the same render→Read pattern as the comsol/document skills.
Do not ask the user to eyeball intermediate results — use the preview yourself.

## Output locations

| Path | Contents |
|---|---|
| `data/research/<n>_<name>/article/figures/<slug>/` | one figure pipeline + `out/` deliverables |
| `data/skills/scientific_plotting/templates/` | custom scaffold templates (user-added) |
| `data/skills/scientific_plotting/recipes/` | reusable cross-research plotting recipes |
| `data/skills/scientific_plotting/cache/` | transient previews (git-ignored) |

## When something breaks

1. Run `figures doctor` — reports backends, resolved fonts per preset, and the Agg backend.
2. A `FALLBACK` font line means the preset's first-choice font is missing (e.g. Helvetica);
   the figure still renders with the fallback. Install the font + clear the matplotlib font
   cache to make it first-choice.
3. `audit` failures point at the exact spec violated (width / fontsize / fonttype / labels /
   colorblind) with a suggested fix.

## Reference files

- [references/plotting.md](references/plotting.md) — the publication conventions in depth:
  mathtext italic/upright/bold rules, colorblind palettes, design widths, manual SVG/EPS
  editing, and the iteration-scenario playbook.
