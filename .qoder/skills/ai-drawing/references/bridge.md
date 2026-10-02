# Bridge — aesthetic reference → data-accurate reproduction

This is the skill's **killer workflow (A)**. AI image models produce gorgeous compositions whose
*data, labels, and text are nonsense*. The bridge turns such an image into a **design template** and
scaffolds a real `scientific_plotting` pipeline so you can reproduce the *look* with **real data**.

```
AI reference (beautiful, fake)  ──bridge──▶  figure pipeline (scaffold) + design_spec.md
                                                     │  fill in REAL data
                                                     ▼
                                             figures build → preview.png
                                                     │  Read & compare to reference
                                                     ▼
                                                iterate (color/layout)
```

The bridge never fabricates data — it extracts **design intent** (palette, aspect, composition) and
hands you a skeleton plus a spec. You (or the Agent) fill in the real numbers.

---

## What `imagine bridge` does

```powershell
uv run pysci-imagine bridge --ref <reference.png> --research <line> --slug <figN_name> `
    [--style aps|nature] [--width single|double] [--template single|multi_panel|concept_plus_data] `
    [--n 6] [--palette-name NAME] [--no-copy-ref] [--overwrite]
```

Step by step:

1. **Analyze the reference** (`bridge.analyze_reference`): width/height, aspect (`w/h` and `h/w`),
   orientation (landscape/portrait/square), mean brightness, and the **dominant palette** — via
   Pillow median-cut color quantization on a thumbnail (`postprocess.extract_palette`), returned as
   hex strings sorted by pixel-share (most dominant first).
2. **Scaffold a figure pipeline** by calling scientific_plotting's `scaffold_figure(research, slug,
   style, width, template)`. This creates:
   - the pipeline module `src/pysci/research/<line>/article/figures/<slug>.py`, and
   - the data dir `data/research/<n>_<line>/article/figures/<slug>/` (with `out/` and `notes.md`).
3. **Copy the reference** into the figure dir as `_reference.png` (so `design_spec.md` can embed it
   and you can eyeball it next to the preview). Skip with `--no-copy-ref`.
4. **Write `design_spec.md`** into the figure dir: the embedded reference, dimensions/aspect/
   brightness, a palette table with approximate roles, ready-to-paste `palette.color("#hex")`
   snippets, layout hints, and the exact next-step commands.
5. **(Optional) register the palette** (`--palette-name NAME`): calls scientific_plotting's
   `register_palette(NAME, hexes)` — persisted to `data/skills/scientific_plotting/palettes.json` —
   and writes a figure-dir `STYLE.yaml` with `palette: NAME` so `figures build` picks it up.

It returns the figure dir, pipeline path, spec path, and the analysis — the CLI prints them plus the
next steps.

---

## `imagine palette` — offline color extraction only

When you just want the hexes (no scaffolding), or to sanity-check a reference first:

```powershell
uv run pysci-imagine palette --src <reference.png> --n 6
```

Prints dimensions/orientation/brightness and the dominant hex palette. Pure Pillow, offline, no key,
no network — safe to run any time. This is the exact extraction the bridge uses.

**Tuning `--n`:** 4–6 gives a clean base + accents for most figures; 8–12 for richly colored
references. Median-cut blends adjacent hues, so very high `n` yields near-duplicate shades.

---

## Using the extracted colors in the pipeline

`scientific_plotting`'s `palette.color()` **accepts a raw hex directly**, so the simplest path needs
no registration — just paste from `design_spec.md`:

```python
from pysci.skills.scientific_plotting.tools import palette

ax.plot(x, y_real, color=palette.color("#121A2E"))  # dominant / background tone
ax.plot(x, y2_real, color=palette.color("#BDA869"))  # accent 1
```

Or, if you passed `--palette-name`, reference the registered palette by name (works across the
separate `figures build` process because it's persisted, and is bound via the figure's `STYLE.yaml`):

```python
palette.color(0, "my-ref")  # by index into the registered palette
# or let the whole color cycle come from it: STYLE.yaml → palette: my-ref
```

> **Colorblind-safety caveat:** AI-extracted palettes are **not** guaranteed colorblind-safe. The
> skill defaults stay Okabe-Ito; `pysci-figures audit` will still flag a registered AI palette that
> fails color-vision readability. For a submission-critical figure, prefer the built-in safe
> palettes and borrow only the reference's *composition*, not its exact hues.

---

## Worked example (Workflow A end to end)

```powershell
# 1. Produce an aesthetic reference (Tier 1 gen, or Tier 0 ImageGen + ingest)
#    --size takes 1K/2K/3K/4K or an explicit WxH. No --seed here on purpose: Ark does **not**
#    guarantee the same image from the same prompt+seed, so a seed is not a reproduction handle —
#    curating the winner into gallery/ is (see SKILL.md “Cost rule”).
uv run pysci-imagine gen --prompt 'elegant journal cover: chiral edge states, glowing arcs, deep navy, minimalist, no text' --size 2K --research gain_ep --slug ref_cover

# 2. LOOK at it (visual check) — Read the PNG the command printed

# 3. Peek at its palette (optional)
uv run pysci-imagine palette --src data/research/1_gain_ep/article/artwork/ref_cover.png --n 6

# 4. Bridge it into a real figure pipeline
uv run pysci-imagine bridge --ref data/research/1_gain_ep/article/artwork/ref_cover.png --research gain_ep --slug fig1_cover --palette-name gain-cover --style aps --width double

# 5. Fill in REAL data: edit src/pysci/research/gain_ep/article/figures/fig1_cover.py
#    (design_spec.md in the figure dir tells you the hexes + layout to aim for)

# 6. Build + Read the preview, compare against _reference.png, iterate
uv run pysci-figures build data/research/1_gain_ep/article/figures/fig1_cover
#    → Read data/research/1_gain_ep/article/figures/fig1_cover/out/fig1_cover_preview.png
```

---

## What the bridge deliberately does **not** do

- **It does not fill in data.** The scaffolded pipeline keeps the template's placeholder series; you
  replace them with real simulation/experiment data. That separation is the whole point — the AI
  supplies *aesthetics*, you supply *truth*.
- **It does not replicate the layout pixel-for-pixel.** It gives aspect/orientation/brightness hints
  and picks a scaffold `template`; matching the reference's composition is a human/Agent editing task
  guided by `design_spec.md` + the `_reference.png` side-by-side.
- **It does not override journal specs.** `STYLE.yaml` keeps the journal `style`/`width`; the
  reference's aspect ratio is a *hint* in the spec, not forced onto the figure (submissions must obey
  the journal's width/font rules, audited by `pysci-figures audit`).

Re-run with `--overwrite` to refresh `design_spec.md` after curating a better reference; the pipeline
module is only rewritten when `--overwrite` is passed, so your filled-in data isn't clobbered by
accident.
