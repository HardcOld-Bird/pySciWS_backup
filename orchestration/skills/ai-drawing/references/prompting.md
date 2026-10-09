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

## Negative constraints — there is no `--negative` flag

Ark's OpenAI-compatible `/images/generations` body has **no negative-prompt field** (see
`ark_client.build_payload`). Fold the negatives into the **tail of the prompt itself**, which
Seedream respects well:

```
…, no text, no words, no letters, no watermark, no signature, no logo, not blurry,
not oversaturated, no clutter, no gibberish labels, no misleading axes
```

If a future Ark model ever adds a real field, pass it through the escape hatch without touching the
CLI: `--extra '{"some_field": "…"}'`. Inspect what will actually be sent with `--dry-run` first —
it prints the redacted request body, costs nothing and needs no key.

## Prompting for marker-driven editing (`imagine edit`)

`edit` is `i2i` with a convention: the input image carries **lettered markers** drawn by
`imagine mark`, and the prompt refers to them by those letters. Ark exposes no per-region parameter
— **the binding lives entirely in the prompt text**, so the letters must match exactly.

```
根据图中标记进行修改：将框选区域 A 的配色改为深蓝到青色渐变；在箭头 B 所指位置添加一个
半透明的声波波前示意；保持其余部分、整体透视、光影与画风完全不变。不要添加任何文字。
```

`imagine mark` prints the label list (`A: 框选区域 @ (x,y)` …) and then a ready-to-paste prompt
skeleton built from *your* markers — copy that instead of inventing letter assignments.

Rules that make edits land:

- **One marker = one instruction clause**, in the order `mark` assigned the letters.
- Always close with “keep everything else unchanged” (透视/光影/画风/构图) — otherwise the model
  re-imagines the whole image.
- Keep `--size` matched to the input's aspect; a mismatch is the usual cause of “it moved my layout”.
- `edit` auto-switches to `doubao-seedream-5-0-pro-260628`, the tier Ark documents for interactive
  editing (and for ≤10 reference images). Override with `--model` only if you know why.

## Prompting for layer decomposition (`imagine layers`)

Same mechanism, different intent: ask for **N transparent layers** and enumerate them explicitly.
The model returns a base plate plus up to 16 alpha layers; `imagine img split` then separates each
into `_rgb.png` + `_alpha.png`, and `imagine img composite` re-stacks them.

```
将这张海报拆分为四个透明图层：1) 主标题文字；2) 说明性小字与图例；3) 中央主体物；
4) 纯色背景。保持所有元素的原始尺寸、位置和光影完全不变，不要重绘任何内容。
```

Rules:

- **Number and name every layer** — an unenumerated request comes back as one flat image.
- Say “保持原始尺寸/位置/光影不变，不要重绘”, or the model will happily redesign the elements.
- ≤16 layers; fewer, semantically distinct layers re-composite far better than many overlapping ones.
- After `layers`, `imagine sheet <all outputs>` and `Read` the sheet before spending again.

## Seed / size / model guidance

- **Seed is NOT a reproduction handle.** Ark only honours `seed` on the Seedream **3.0 t2i** model
  (range 0–65535) and explicitly states that *the same seed does not guarantee the same image*.
  `imagine gen` therefore defaults to **no seed** rather than faking determinism. The reliable
  retention path is `imagine gallery --add <png> --slug S` **immediately** after you `Read` a result
  you like — `assets/` is git-ignored and its contents **cannot** be regenerated from the ledger.
- **Size**: `1K` for quick drafts, `2K` for deliverables (default), `4K` only for final large covers.
  Or explicit `--width` / `--height` (e.g. `1024`×`1536` portrait cover). `--output-format png` is
  supported **only** by the plain 5.0 tier (`doubao-seedream-5-0-260128` — note the real ID has
  **no** `lite` suffix); every other tier returns JPEG. The CLI
  sniffs the real bytes when naming files, so the extension is correct either way.
- **Model**: run `imagine models` for the capability matrix (reference-image limits, group
  generation, `png`, web search, seed semantics) — read it instead of guessing. That matrix is
  hand-curated navigation and **will lag**; `imagine models --live` asks Ark directly which IDs
  your account can actually see (read-only, free) — trust it over any document, this one included.
  But catalog ≠ activation: `/models` still lists delisted IDs, so only the console 「开通管理」 or a
  real call (free when rejected at the authorization stage) proves usability. On this account the
  live tiers are `5-0-flash` (default) and `5-0-pro` (2026-10, both verified by a real 1K render).
  IDs are passed through verbatim; the Ark console “模型列表” is the only source of truth.
- **`--n` vs `--group`**: `--n 3` fires **three independent requests** (three different compositions
  — good for picking a winner); `--group --max-images 3` asks for **one coherent set** (4.0 / 4.5 /
  5.0 main tier only; reference images + outputs ≤ 15). Both are clamped by the `AI_DRAWING_MAX_IMAGES`
  cost guard, and `--dry-run` tells you the clamp before you pay.
- **Watermark**: Ark defaults to `watermark: true`. This CLI always sends `false` explicitly and
  only adds the “AI 生成” mark when you pass `--watermark`.

## Cost discipline while prompting

1. Draft small/cheap (`--size 1K`, no `--n`) to converge on the prompt.
2. `--dry-run` any new combination first — exact request body, zero cost, no key required.
3. `Read` every result before the next spend; iterate the prompt, not the wallet.
4. The moment a draft looks right, `imagine gallery --add <png> --slug S` — that copy **is** the
   record, because re-rolling is not guaranteed to bring it back.
5. Only then spend on the final `2K` / `4K` render.
6. Save reusable prompts under `data/skills/ai_drawing/prompts/`, and multi-step procedures as a
   Markdown knowledge card in `data/skills/ai_drawing/recipes/` (same convention as the other
   skills; `imagine list` inventories them).
