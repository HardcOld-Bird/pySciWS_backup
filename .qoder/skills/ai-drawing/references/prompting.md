# Prompting — scientific-aesthetic recipes

Goal: images that look like **polished journal artwork** (covers, graphical abstracts, schematics),
not generic AI illustration. Two very different jobs live here:

1. **Final artwork** (Workflow B) — the image *is* the deliverable. Push aesthetics hard.
2. **Aesthetic reference** (Workflow A) — the image is a *design template* to reproduce with real
   data. Here you want clean composition + a good palette, and you deliberately **ignore** any
   text/numbers the model invents (they're nonsense; `bridge` only takes colors/layout).

Prompt language: Seedream/Jimeng handles both Chinese and English. English tends to give more
controllable style vocabulary; Chinese is fine for domain subjects. Keep prompts **specific and
layered**: subject → composition → style → lighting/color → quality tags → (negative).

## The prompt anatomy

```
[subject]        what it is: "a non-Hermitian phononic crystal waveguide with exceptional points"
[composition]    layout/framing: "centered hero object, rule-of-thirds, generous negative space"
[style]          visual language: "scientific journal cover, clean vector-illustration, isometric"
[light/color]    mood + palette: "soft studio lighting, teal-to-magenta gradient, dark navy background"
[quality]        fidelity tags: "high detail, sharp focus, 4k, professional graphic design"
[negative]       what to avoid: (see below)
```

## Recipe — journal cover

```
A <journal> cover illustration of <subject>. Cinematic centered composition with strong negative
space at the top for a title. Semi-realistic scientific visualization mixed with clean graphic
design. Dramatic soft lighting, a restrained 3-color palette (<c1>, <c2>, <c3>), subtle depth of
field. Elegant, modern, publication-grade, high detail. No text, no words, no letters, no watermark.
```

Tips: name the palette explicitly (3–5 colors) so `bridge`/`palette` extraction is meaningful; always
add `no text` — models render gibberish letters that ruin covers.

## Recipe — graphical abstract

```
A horizontal graphical abstract schematic for <paper topic>, reading left-to-right in 3 stages:
<input/method> → <process> → <result>. Flat modern scientific illustration, thin clean outlines,
consistent iconography, labeled panels separated by arrows. Light background, colorblind-safe
palette (blue, orange, teal, muted red). Balanced, uncluttered, vector-like, high resolution.
No photorealism, no clutter, no lorem text.
```

Tips: ask for **stages/panels and arrows** to get a readable flow; keep it flat/vector so it scales
and so the real figure can mirror the layout.

## Recipe — concept schematic / diagram

```
A clean conceptual diagram of <system>, isometric perspective, minimal flat design, labeled
components as simple geometric shapes, thin connector lines, a limited accent palette on white.
Technical illustration style, precise, uncluttered, generous whitespace. No text labels (leave
space for annotations), no 3D render noise.
```

Tips: "leave space for annotations / no text labels" gives you room to overlay real labels later
(via scientific_plotting `raster-panel` overlays or manual editing).

## Recipe — aesthetic reference for the bridge (Workflow A)

```
An aesthetically refined scientific figure layout about <topic>: a <N>-panel composition with a
dominant <chart type> and supporting insets, harmonious color palette (<c1>, <c2>, <c3>), balanced
margins and clear visual hierarchy, modern minimal data-viz style. The exact data/labels are
placeholders. Clean, publication-quality, high detail.
```

Then: `imagine bridge --ref <png> --research R --slug S` — it extracts the dominant hexes + aspect
ratio and scaffolds a real pipeline; **you** fill in the true data. Do not trust the model's numbers.

## Negative prompt (Tier 1)

A good default negative for scientific artwork:

```
text, words, letters, watermark, signature, logo, blurry, low quality, jpeg artifacts,
oversaturated, cartoonish, cluttered, extra limbs, distorted geometry, noise, grainy,
random gibberish labels, misleading axes
```

## Seed / size / model guidance

- **Seed**: always set one for anything you might reproduce (`--seed 42`). Same seed + prompt +
  model ⇒ near-identical image, so you can iterate on *one* variable without re-rolling (and
  re-paying). The ledger stores it.
- **Size**: `1K` for quick drafts, `2K` for deliverables (default), `4K` only for final large
  covers. Or explicit `WxH` (e.g. `1024x1536` portrait cover, `1536x1024` landscape abstract).
- **Model**: `doubao-seedream-4-0-250828` (default) is a good all-rounder; newer Seedream 5 when
  available for finer detail. Confirm exact IDs via `imagine comfy nodes` (never guess).
- **n**: default 1. Only raise `--n` (capped by `AI_DRAWING_MAX_IMAGES`) when you genuinely want to
  compare candidates — then `imagine sheet` them into a contact sheet and `Read` once.

## Cost discipline while prompting

1. Draft small/cheap (`1K`, `--n 1`) to converge on the prompt.
2. Lock the seed once a draft looks right.
3. Only then spend on the final `2K`/`4K` render.
4. `Read` every result before the next spend; iterate the prompt, not the wallet.
5. Curate winners into `gallery/` and save reusable prompts under `prompts/`.
