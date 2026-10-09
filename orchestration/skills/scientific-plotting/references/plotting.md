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
(height = width × `aspect`, default golden ratio 0.618). `audit` verifies the exported width
matches the target within ±2 mm (tight/constrained layout introduces tiny drift).

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

## Iteration playbook (the three common asks)

1. **Same plot, new data** — only touch the data-loading block of `build_figure`; the plotting
   and style layers are untouched. Re-`build`, Read preview.
2. **Same data, new plot type** — change the plotting calls (line↔scatter↔imshow, add error
   bars, change colormap). Re-`build`, Read preview.
3. **Same content, new journal** — no code change: re-run `figures build <figdir> --style nature`
   (or edit `STYLE.yaml`). Widths/fonts/sizes/palette all follow the preset. Then `audit`.

Log each iteration in the figure's `notes.md` (date / need / result) so the history is
reproducible.
