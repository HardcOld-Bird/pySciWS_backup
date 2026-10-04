# Workflows B–E — artwork production, marker editing, layers, local cleanup

Step-by-step checklists for everything that is *not* the aesthetics→data bridge:
direct artwork production (B), marker-driven local editing (C), layer decomposition and
recomposite (D), and the no-cloud / no-cost local bitmap operators (E). `SKILL.md` keeps
Workflow A (the killer feature) plus a one-line pointer to each of these.

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
[prompting.md](prompting.md).

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
