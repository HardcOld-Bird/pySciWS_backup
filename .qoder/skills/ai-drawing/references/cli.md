# The `imagine` CLI — every flag, provenance, and troubleshooting

The full option list for all 24 `imagine` commands, how generation provenance is stored
twice (ledger + PNG `tEXt` chunks), and the long form of each failure mode. `SKILL.md`
keeps only the grouped command table, the output paths, and the first-line triage.

## Full command table

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

## Output paths, and provenance stored twice

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

## Troubleshooting, in full

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
