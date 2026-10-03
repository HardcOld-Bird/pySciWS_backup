---
name: ai-drawing
description: Generate and edit images with Volcano Ark / Jimeng Seedream cloud models over an OpenAI-compatible API, plus local bitmap processing (Pillow always; scikit-image + OpenCV via the optional `imaging` extra), a generation ledger with PNG-embedded provenance, and an aesthetics→data bridge into scientific_plotting. Two tiers — Tier 0 is Qoder's built-in ImageGen (zero-install sketches and fallback); Tier 1 is a direct Ark HTTP client (text-to-image, image-to-image, multi-reference, coherent groups, marker-driven interactive editing, layer decomposition). No local diffusion, no orchestrator process. Drives a unified `imagine` CLI. Use when the user asks to generate or draw an image, create a journal cover / graphical abstract / schematic, produce an aesthetic reference to later reproduce with real data, edit or retouch a generated image, mark up regions and edit only there, split artwork into layers and recomposite it, seamlessly blend a generated element into a photo or field plot, remove a defect or watermark, measure / align / morphologically clean a bitmap, extract a color palette from a reference image, or bridge an AI artwork into a data-accurate figure.
---

# AI Drawing

One CLI drives the whole loop: **generate → look → curate → adjust → verify**, plus an
**aesthetics → data bridge** that turns a beautiful-but-fake AI reference into a real, data-accurate
figure pipeline.

The backend lives in `src/pysci/skills/ai_drawing/` (tools: `config` / `ark_client` / `postprocess` /
`imaging` / `ledger` / `bridge` + an `imagine` facade). **Treat it as a black box** and drive
everything through the `imagine` CLI below. Only open the code when maintaining it.

**The key design idea:** this skill does **not** run diffusion locally (the box has no CUDA GPU) and
does **not** run an orchestrator process. Generative work is a single plain `requests` POST to
**Volcano Ark / Jimeng Seedream** with your own API key; everything non-generative (crop, blend,
inpaint, measure, align) is done locally by **Pillow / OpenCV / scikit-image**. There is no node
graph, no server to start, no second venv — and no `torch` anywhere near pysci's dependencies. The
reasoning and the conditions that would justify bringing an orchestrator back are recorded in
[references/backends.md](references/backends.md).

## Invocation

Run from the **project root**. The skill installs a console script `pysci-imagine`:

```
uv run pysci-imagine <command> [options]
```

Below, `imagine …` is shorthand for `uv run pysci-imagine …`. (Fallback if the script isn't
installed / offline: `uv run --no-sync python -m pysci.skills.ai_drawing.tools.imagine …`.)

> **PowerShell rules (critical):** first set `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8`
> in the same shell — this CLI prints Chinese, and piping its stdout decodes those UTF-8 bytes with
> the GBK/936 console codepage (mojibake). Then: wrap multi-word arguments (prompts!) in **single
> quotes**; use `;` (never `&&`) to chain commands. See `.qoder/rules/basic.md` §2.

> **Cost rule:** every Tier 1 `gen` / `i2i` / `edit` / `layers` call spends real money (~¥0.2 per
> image on Ark). Default to **one** image; `--n` is explicit and clamped by the cost guard. Preview
> any new flag combination with `--dry-run` — free, keyless, prints the exact request body. Always
> `Read` the result before spending again.

> **Retention rule (read this):** Ark does **not** guarantee the same image from the same
> prompt+seed, so the ledger is **provenance, not a reproduction recipe**. `assets/` is git-ignored.
> The moment you `Read` a result worth keeping, run `imagine gallery --add <png> --slug S` — that
> curated copy is the only reliable way to still have it next month.

## One-time setup

```dotenv
# project-root .env  (git-ignored)
ARK_API_KEY=<from console.volcengine.com/ark → API Key 管理 → 创建>
```

Full walkthrough (real-name verification, per-model activation in 模型广场, masked verification via
`doctor`): [references/backends.md](references/backends.md#getting-an-ark_api_key).

Optional, for the local bitmap operators that need OpenCV / scikit-image:

```powershell
uv pip install -e ".[imaging]"
```

## Capability boundary (read this first)

**Tier 0 — Qoder built-in `ImageGen` (available now, zero install):**
- The Agent calls the `ImageGen` tool directly for text-to-image (returns a `file_path`).
- Great for quick concept sketches, covers, and as a fallback when no `ARK_API_KEY` is configured.
- No input image, no model/size control. Then `imagine ingest` to file + ledger it.

**Tier 1 — Ark direct (needs `ARK_API_KEY` only):**
- Text-to-image, image-to-image, multi-reference, marker-driven interactive editing, layer
  decomposition. On **this account** the live tiers are 5.0-flash (default) and 5.0-pro; coherent
  groups and direct PNG need tiers that are delisted / closing here, so those are contract-supported
  but currently unreachable.
- `imagine models` prints the capability matrix (per-tier activation status as last verified);
  `imagine models --live` lists the catalog your account sees — **catalog ≠ activation**, a delisted
  ID still shows up there. `imagine doctor` reports the resolved endpoint and whether a key is
  present (masked). Without a key, Tier 1 fails fast with a setup hint and Tier 0 still works.

**Always available (pure Python, offline, no key):** `doctor` / `models` / `list` / `ledger` /
`gallery` / `ingest` / `mark` / `adjust` / `sheet` / `palette` / `bridge` / `img split` /
`img composite` — Pillow and the ledger need no network and no extra.

**Needs the `imaging` extra:** the rest of `img` (`fuse` / `inpaint` / `mask` / `morph` / `warp` /
`measure` / `align`). Without it these print the exact install command instead of a traceback.

## Commands at a glance

| Command | Use when | Key output |
|---|---|---|
| `doctor` | Session start / anything broken | config + libs + endpoint + key presence + cost guard + Tier 0 |
| `models [--live] [--all]` | Model choice unclear (**never guess capabilities**) | capability matrix; `--live` reads the IDs your account can actually see (free) |
| `list [--research R]` | Inventory of assets/gallery/recipes (or a research line's artwork) | file listing |
| `ledger [--backend --model --contains --limit] [--stats] [--render]` | Recalling how an image was made | provenance table (+ `LEDGER.md`) |
| `gallery [--add I --slug S] [--sheet]` | **Retaining** a winner / curating aesthetic references | gallery copy / contact sheet |
| `ingest --src I [--slug S] [--research R] [--gallery] [--prompt --seed --model --backend --recipe --notes]` | Filing an already-generated image (ImageGen / external) | copied into `assets/` (or `article/artwork/`) + ledger entry |
| `gen --prompt P [--size --width --height --model --seed --n] [--group --max-images] [--research --slug --recipe --notes] [--dry-run]` | Text-to-image / coherent group | image(s) + PNG metadata + ledger + `runs/` snapshot |
| `i2i --image I --prompt P […]` | Image-to-image / multi-reference (repeat `--image`; local files auto-base64'd) | image(s) + ledger |
| `edit --image MARKED --prompt P […]` | Marker-driven local editing (auto-switches to 5.0-pro) | edited image + ledger |
| `layers --image I --prompt P […]` | Layer decomposition (1 base + ≤16 alpha layers, 5.0-pro) | layers + ledger |
| `mark --src I [--rect x,y,w,h]… [--arrow x1,y1,x2,y2]… [--point x,y]… [--text x,y,内容]… [--color --width --out]` | Annotating regions **before** `edit` (pure Pillow) | marked image + auto A/B/C labels + a prompt skeleton |
| `adjust SRC --out O [--crop --resize --rotate --mode --pad --pad-color --format --quality]` | Simple edits (crop/resize/rotate/convert/pad) | adjusted image |
| `sheet IMG… [--out O] [--cols --thumb --pad --bg --no-label]` | Seeing many images at once | contact-sheet PNG |
| `palette --src I [--n 6]` | Extracting a reference's dominant colors | hex palette (for the bridge) |
| `bridge --ref I --research R --slug S [--style --width --template --n --palette-name --no-copy-ref --overwrite]` | Turning an aesthetic ref into a real figure | scaffolded pipeline + `design_spec.md` |
| `img split SRC [--dest]` | Separating an RGBA layer into `_rgb.png` + `_alpha.png` | two files (pure PIL) |
| `img composite --base B --layer L… [--opacity …] [--pos x,y …] [--out O]` | Re-stacking layers (consumes `layers` output) | composited image (pure PIL) |
| `img fuse SRC --base B [--mask --center --mode]` | Seamlessly blending an element into a photo / field plot | Poisson-blended image (OpenCV) |
| `img inpaint SRC --mask M [--radius --method telea\|ns]` | Removing a defect / watermark residue (no GPU, no weights) | repaired image (OpenCV) |
| `img mask SRC [--method otsu\|manual\|canny\|grabcut --thresh --invert --blur --grabcut-rect --iterations]` | Building the mask `fuse`/`inpaint` need | binary mask (OpenCV) |
| `img morph SRC [--op-name open\|close\|erode\|dilate --radius --iterations]` | Removing specks / filling holes | cleaned image (scikit-image) |
| `img warp SRC [--src-pts --dst-pts --size --rotate]` | Rectifying a tilted photo, or a pure rotation | corrected image |
| `img measure SRC [--min-area --thresh --mask]` | Quantifying blobs (area / centroid / axes / solidity) | measurement table |
| `img align SRC --ref R [--upsample --no-apply]` | Sub-pixel registration of two same-size images | shift `(y,x)`+`(x,y)`, optional aligned image |

Run `imagine <command> -h` for the full option list.

## Workflow A — aesthetic reference → data-accurate reproduction (the killer feature)

AI images are *beautiful but their data/labels are nonsense*. Use them as **design templates**, then
reproduce with real data via `scientific-plotting`:

```
- [ ] 1. imagine gen --prompt '<aesthetic cover/figure concept>' --size 2K   # or use Tier 0 ImageGen
- [ ] 2. Read the generated PNG                                              # judge composition/color
- [ ] 3. imagine gallery --add <png> --slug ref_cover                        # RETAIN it (see Retention rule)
- [ ] 4. imagine bridge --ref <png> --research gain_ep --slug fig1_cover     # extract palette + layout
- [ ] 5. Edit the scaffolded pipeline: fill in REAL data (bridge wrote design_spec.md with hexes/layout)
- [ ] 6. figures build 'data/research/1_gain_ep/article/figures/fig1_cover'
- [ ] 7. Read the _preview.png and compare against the reference; iterate
```

`bridge` uses Pillow color quantization to pull the dominant hexes, notes the aspect ratio /
composition, calls `scaffold_figure(...)`, and writes a `design_spec.md` (embedded reference + hex
palette + layout hints) into the figure dir. Optionally it registers the palette into `STYLE.yaml`
via scientific_plotting's `register_palette` (colorblind-safety still audited). Details:
[references/bridge.md](references/bridge.md).

## Workflow B — direct artwork production (cover / graphical abstract / schematic)

For purely aesthetic deliverables (no data to be accurate):

```
- [ ] 1. imagine doctor                                       # Tier 1 key present? else use Tier 0
- [ ] 2. imagine gen --prompt '<…>' --size 2K --research gain_ep --slug cover --dry-run   # free check
- [ ] 3. same command without --dry-run
       (Tier 0: call ImageGen, then imagine ingest --src <path> --slug cover --research gain_ep)
- [ ] 4. Read the PNG                                         # VISUAL CHECK
- [ ] 5. imagine gallery --add <png> --slug cover             # retain the winner NOW
- [ ] 6. imagine adjust <png> --out <png> --resize 1600 0     # crop/resize/convert as needed
- [ ] 7. imagine ledger --contains '<kw>'                     # recall prompt/model/usage later
```

## Workflow C — marker-driven local editing (`mark` → `edit`)

Ark has **no per-region parameter**; the region↔instruction binding lives entirely in the prompt
text, and `mark` is what makes that binding unambiguous.

```
- [ ] 1. imagine mark --src <png> --rect 120,80,400,260 --arrow 900,700,760,560
            → prints "A: 框选区域 @ (120,80)" / "B: 箭头所指位置 @ (760,560)"
            → and a ready-to-paste prompt skeleton mentioning A and B
- [ ] 2. Read the marked PNG                                  # confirm the boxes cover what you meant
- [ ] 3. imagine edit --image <marked.png> --prompt '根据图中标记进行修改：将框选区域 A …；
            在箭头 B 所指位置添加 …；保持其余部分、整体透视、光影与画风完全不变。'
- [ ] 4. Read the result; if a region was missed, re-mark and re-edit (don't re-roll blindly)
```

Labels are assigned automatically in `--rect` → `--arrow` → `--point` order (A, B, C …). One marker
per instruction clause, same order. Prompt patterns:
[references/prompting.md](references/prompting.md).

## Workflow D — layer decomposition → recomposite

```
- [ ] 1. imagine layers --image <poster.png> --prompt '将这张海报拆分为四个透明图层：1) 主标题；
            2) 说明小字；3) 中央主体；4) 纯色背景。保持原始尺寸、位置、光影不变，不要重绘。'
- [ ] 2. imagine sheet <all outputs> ; Read it                # did it actually separate?
- [ ] 3. imagine img split <layerN.png> --dest <dir>          # RGBA → _rgb.png + _alpha.png
- [ ] 4. imagine img composite --base <base.png> --layer L1 --layer L2 --opacity 1 --opacity 0.6
            --pos 0,0 --pos 40,30 --out <dir>/recomposed.png
- [ ] 5. Read, then iterate opacities/positions (free — steps 3–5 are all local)
```

Steps 3–5 cost nothing, which is the point: experiment locally, spend only on step 1.

## Workflow E — local cleanup, blending and measurement (no cloud, no cost)

For fixing up an existing image — AI-generated, photographed, or a COMSOL/实验 export:

```
- [ ] imagine img mask    <src> --method grabcut --grabcut-rect 100 80 600 700   # or otsu/canny
- [ ] imagine img inpaint <src> --mask <m> --radius 3                            # kill a defect
- [ ] imagine img fuse    <element.png> --base <photo.png> --mask <m>            # blend seamlessly
- [ ] imagine img morph   <src> --op-name open --radius 2                        # drop specks
- [ ] imagine img warp    <src> --src-pts '…;…;…;…' --size 1200 900              # rectify a tilt
- [ ] imagine img measure <src> --min-area 50                                    # area/centroid/axes
- [ ] imagine img align   <src> --ref <ref> --upsample 10                        # sub-pixel shift
```

> **`fuse` gotcha:** Poisson blending transfers **gradients, not absolute color**. A flat-colored
> element blended into a flat background **vanishes** (the solution collapses to the boundary
> value). To *paste* something, use `img composite`. `--mode mixed` lets the base texture show
> through; `--mode normal` smooths it away — they differ **only** when the base has texture.

`measure` / `align` are the scientific-quantification entry points: use them to read geometry off a
microscope/实验 image, or to register two frames before overlaying them in a figure.

## Visual closed loop

Every generation/edit prints a `✓ 视觉校验：Read '<path>'` line. **Read the PNG** with the Read tool
to actually *see* it, then iterate on the prompt (Tier 1) or the adjust/img params. Never ask the
user to eyeball intermediate images — look yourself. This is the same render→Read pattern as the
comsol / figures / document skills. Use `sheet` / `gallery --sheet` to compare many candidates with
a single `Read`.

## Output locations

| Path | Contents |
|---|---|
| `data/skills/ai_drawing/assets/` | generated images (git-ignored: large, **not** regenerable from the ledger) |
| `data/skills/ai_drawing/gallery/` | curated aesthetic references (tracked, keep < 2MB each) |
| `data/skills/ai_drawing/recipes/` | pipeline knowledge cards (`.md`, same convention as the other skills) |
| `data/skills/ai_drawing/prompts/` | reusable scientific-aesthetic prompt recipes |
| `data/skills/ai_drawing/LEDGER.md` + `manifest.jsonl` | generation ledger (human + machine); provenance only |
| `data/skills/ai_drawing/cache/` `runs/` | transient previews / redacted request snapshots (git-ignored) |
| `data/research/<n>_<name>/article/artwork/` | a research line's finished artwork (cover / abstract / schematic) |

Provenance is stored **twice**: in the ledger, and as PNG `tEXt` chunks inside every product written
by `gen` / `i2i` / `edit` / `layers` / `ingest` (`pysci:prompt`, `pysci:model`, `pysci:backend`, …)
— so a stray image found outside the ledger can still be traced. Non-PNG products are skipped rather
than re-compressed. `imagine ledger --render` rebuilds `LEDGER.md` from `manifest.jsonl` at any time.

## When something breaks

1. `imagine doctor` — config, libs (PIL / requests / OpenCV / scikit-image), endpoint, key presence
   (masked), cost guard, Tier 0 reminder.
2. Tier 1 fails with "未配置 ARK_API_KEY": add it to the project-root `.env`, then re-run `doctor`.
3. HTTP 401 / 403 with a valid key: the **model isn't activated**. Ark activates each Seedream tier
   individually in 模型广场 — an un-activated ID is an authorization error, not a bad key. And note
   `models --live` shows the **catalog**, not your activations: a rejected 1K probe is the cheapest
   activation test, because only successful images are billed.
4. HTTP 400 on a flag combination: run the same command with `--dry-run` and read the request body
   against `imagine models`. Usual culprits: `--output-format png` on a non-5.0 model,
   `--group` on 5.0-pro, too many `--image` refs, `--seed` out of `[0, 65535]`.
5. `img <op>` prints an install command: `uv pip install -e ".[imaging]"`.
6. Wasted spend / can't get a good image back: it is **not** in the ledger's power to restore it —
   Ark doesn't guarantee prompt/seed determinism. `imagine ledger --contains '<kw>'` recovers the
   prompt so you can re-roll deliberately; `gallery/` is where survivors should already be.
7. Anything unexplained: the redacted request + response snapshot is in `data/skills/ai_drawing/runs/`.

## Reference files

- [references/backends.md](references/backends.md) — Tier 0 vs Tier 1 matrix, the community libraries
  we reuse for local bitmap work, **why ComfyUI was removed and when to bring an orchestrator back**,
  `ARK_API_KEY` walkthrough, `.env` keys, cost discipline.
- [references/prompting.md](references/prompting.md) — scientific-aesthetic prompt recipes (cover /
  graphical abstract / schematic / bridge reference), negative constraints (there is no `--negative`
  flag), prompting for `edit` and `layers`, seed/size/model truth, cost discipline.
- [references/bridge.md](references/bridge.md) — the aesthetic-reference → data-reproduction playbook.
