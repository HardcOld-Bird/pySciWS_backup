# Publication plotting conventions (in depth)

The `style` / `palette` / `layout` / `audit` layers encode these rules. This file is the
human-readable source of truth for *why* they exist and how to apply them when hand-writing
or revising a pipeline's `build_figure`.

## Design widths

| Preset | Single column | Double column |
|---|---|---|
| `aps` (APS/PRL, revtex) | 85 mm | 170 mm |
| `nature` (Nature/Science) | 89 mm | 183 mm |

`style_context(..., width="single"|"double"|"<mm>")` sets `figure.figsize` accordingly
(height = width × `aspect`, default golden ratio 0.618).

### Two width checks (and the deliverable fake-green guard)

`audit` runs two distinct width checks:

1. **In-memory** (`ERROR width`): the re-rendered figure's width vs the design target, ±2 mm.
   Note this is near-tautological — both come from the same resolved `eff_width` — so it only
   catches an explicit `width=<mm>` that disagrees with the preset, not a mis-shipped file.
2. **Deliverable** (`WARN width-deliverable`): `audit_figure_dir` measures the *already
   exported* `out/<stem>.eps` (or `.pdf`) via its `%%HiResBoundingBox` / `/MediaBox` and
   compares to the target by **ratio** (default ±30 %), not absolute mm. `save_figure` uses
   `bbox_inches="tight"`, so the deliverable is always narrower than the design width and the
   crop varies with content — an absolute tolerance would false-warn on every correct figure.
   A ratio still cleanly separates single from double (~2× apart). This is what catches the
   fake green where a research-root `STYLE.yaml width: double` masks a single-column figure:
   the 85 mm deliverable is audited against a 170 mm target and now WARNs.

`STYLE.yaml` is merged **key by key**: the figures-root file is the baseline and the per-figure
`figdir/STYLE.yaml` overrides only the keys it declares (inheriting the rest). So a single-column
figure needs only `width: single` in its own `STYLE.yaml` — it keeps the root's `style`/`palette`
— and the deliverable check then passes. If no deliverable exists yet (never built), check 2 is
skipped, not failed.


## Fonts & sizes

- `aps`: Arial, base 8 pt, ticks 7 pt.
- `nature`: Helvetica, base 7 pt, ticks 6 pt. Helvetica is commercial; the preset's fallback
  chain is `Helvetica → TeX Gyre Heros → Nimbus Sans → Arial → DejaVu Sans`, so an uninstalled
  Helvetica degrades gracefully (doctor reports `FALLBACK`).
- Text is kept **editable** in vector output: `pdf.fonttype=42`, `ps.fonttype=42` (TrueType
  embedding, no outlining) and `svg.fonttype="none"` (real `<text>` nodes). This is what makes
  downstream manual tweaking in Illustrator/Inkscape possible.
- After installing a new font, clear the matplotlib font cache (delete
  `fontlist-*.json` under `matplotlib.get_cachedir()`) or it won't be discovered.

## Italic / upright / bold (physics typesetting)

Use mathtext (`$...$`) for all math-ish tokens; the surrounding text font is sans-serif and
`mathtext.fontset=dejavusans` keeps math visually consistent.

| Element | Convention | Example |
|---|---|---|
| Physical variable, geometric parameter | italic (bare symbol in math mode) | `$x$`, `$L$`, `$k$` |
| Unit, physical constant | upright via `\mathrm{}` | `$\mathrm{mm}$`, `$\mathrm{Hz}$`, `$\mathrm{k}$` |
| Vector | bold italic `\boldsymbol{}` | `$\boldsymbol{v}$` |
| Matrix / operator | bold upright `\mathbf{}` | `$\mathbf{M}$`, `$\mathbf{H}$` |
| Subscript that is a label (not an index) | upright `\mathrm{}` | `$L_\mathrm{c}$` (cavity length) |
| Subscript that is an index/variable | italic | `$x_i$` |

Axis labels follow the pattern `r"$x$ / $\mathrm{rad}$"` (italic variable, upright unit).

## Colorblind-safe palettes

Default `okabe-ito` (8 colors); alternatives `tol-bright`, `tol-vibrant` (see `figures styles`).
`audit` simulates deuteranopia (Viénot–Brettel–Mollon) over the figure's data colors and warns
when two are confusable. If a figure needs more series than distinguishable colors, add
redundant encoding (different line styles / markers), not just more hues.

## Panel labels & layout

- Multi-panel figures get `(a)(b)(c)…` via `layout.label_panels(axes)`; `audit --panels`
  checks the count matches the number of axes.
- Regular grids: `layout.grid(nrows, ncols, ...)`. Composite/nested: `layout.nested(fig, outer,
  inner, outer_ratios=...)` (e.g. left concept + right 2×1 data).
- Top/right spines are off by default (`axes.spines.top/right=False`); ticks point outward.

## External media panels (concept figures)

Blender renders or other raster/vector assets are first-class panels: load with
`plt.imread(path)` (raster) and place via `ax.imshow(...)`; set `ax.axis("off")`. For vector
SVG you want to composite, either rasterize it externally (v2 feature) or embed as an image.
The `concept_plus_data` scaffold template shows the pattern.

## Manual intervention (SVG / EPS)

The pipeline always emits `.svg` with real text (`svg.fonttype="none"`). Open it in
Illustrator/Inkscape for last-mile tweaks (nudging a label, adjusting an arrow), then re-export.
Prefer doing structural changes in `build_figure` (so they survive regeneration) and reserve
manual edits for one-off polish that would otherwise cost many iterations.

## Pipeline discovery (src vs figdir) & the fake-green guard

`runner.discover_pipeline(figdir)` resolves which `.py` to render, **src-first**:

1. `src/pysci/research/<name>/article/figures/<slug>.py` (canonical, Agent-managed);
2. legacy fallback inside the figdir: `<slug>.py` → `fig.py`/`build.py`/`main.py` → the sole `.py`.

When **both** a src module and an explicit figdir entry exist for the same figure, that is a
collision: historically src won *silently*, so a member's hand-written figdir pipeline was shadowed
by the scaffold placeholder — `build` rendered the placeholder yet reported success (**fake green**).

Now a collision prints a `WARNING` naming the file actually chosen **and** the one ignored, and
`RunResult.report` shows the chosen pipeline's full path (not just its basename), so `build` output
is self-evidencing. To resolve intentionally:

- one-off: `figures build <figdir> --pipeline-in-figdir` (same flag on `preview`) forces figdir;
- pinned: `pipeline: figdir|src` in the figdir's (or figures-root) `STYLE.yaml`.

Precedence: CLI flag > `STYLE.yaml pipeline:` > src-first default; an unknown `pipeline:` value is
warned and ignored. `figures list` stays quiet (no per-dir warning) and instead tags each entry
`[src]`/`[figdir]`. The canonical fix for a real collision is to keep **one** pipeline — normally
the src module — and delete the stale figdir copy.

## Iteration playbook (the three common asks)

1. **Same plot, new data** — only touch the data-loading block of `build_figure`; the plotting
   and style layers are untouched. Re-`build`, Read preview.
2. **Same data, new plot type** — change the plotting calls (line↔scatter↔imshow, add error
   bars, change colormap). Re-`build`, Read preview.
3. **Same content, new journal** — no code change: re-run `figures build <figdir> --style nature`
   (or edit `STYLE.yaml`). Widths/fonts/sizes/palette all follow the preset. Then `audit`.

Log each iteration in the figure's `notes.md` (date / need / result) so the history is
reproducible.
