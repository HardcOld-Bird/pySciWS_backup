---
name: ai-drawing
description: Generate and edit images with cloud image models, plus Pillow post-processing, a reproducible generation ledger, and an aesthetics→data bridge into scientific_plotting. Two tiers — Tier 0 is Qoder's built-in ImageGen (zero-install quick sketches and fallback); Tier 1 is a headless ComfyUI used purely as a cloud-API workflow orchestrator that calls Volcano Ark / Jimeng Seedream with your own API key (text-to-image, image-to-image, multi-reference). Drives a unified `imagine` CLI. Use when the user asks to generate or draw an image, create a journal cover / graphical abstract / schematic, produce an aesthetic reference to later reproduce with real data, edit/crop/resize/convert a generated image, extract a color palette from a reference image, or bridge an AI artwork into a data-accurate figure.
---

# AI Drawing

One CLI drives the whole loop: **generate → ingest → adjust → visually verify**, plus an
**aesthetics → data bridge** that turns a beautiful-but-fake AI reference into a real, data-accurate
figure pipeline.

The backend lives in `src/pysci/skills/ai_drawing/` (tools: config / postprocess / ledger /
comfy_client / comfy_session / workflows / providers / bridge + an `imagine` facade). **Treat it as
a black box** and drive everything through the `imagine` CLI below. Only open the code when
maintaining it.

**The key design idea:** this skill does **not** run diffusion locally (the box has no CUDA GPU).
Instead ComfyUI is used purely as a **cloud-API workflow orchestrator** — it runs headless
(`--cpu`) on `127.0.0.1:8188` and, via the community `ComfyUI-Jimeng-API` node, calls **Volcano
Ark / Jimeng Seedream** with *your own* API key. pysci stays a thin HTTP client (`requests` +
optional `websocket-client`) + Pillow post-processing + a generation ledger. ComfyUI + torch + the
custom node are installed by `comfy-cli` in a **separate environment** and never enter pysci's
dependencies.

## Invocation

Run from the **project root**. The skill installs a console script `pysci-imagine`:

```
uv run pysci-imagine <command> [options]
```

Below, `imagine …` is shorthand for `uv run pysci-imagine …`. (Fallback if the script isn't
installed / offline: `uv run --no-sync python -m pysci.skills.ai_drawing.tools.imagine …`.)

> **PowerShell rule (critical):** wrap multi-word arguments (prompts!) in **single quotes**; use `;`
> (never `&&`) to chain commands.

> **Cost rule:** every Tier 1 `gen` / `i2i` / `run` spends real money (~¥0.2 / image on Ark).
> Default to **one** image; pass `--n` explicitly for more. Prefer the free quota first, keep the
> Jimeng **Quota guard node** in the workflow, and always `Read` the result before spending again
> to iterate. Record the `seed` so a good result is reproducible without re-rolling.

## Capability boundary (read this first)

**Tier 0 — Qoder built-in `ImageGen` (available now, zero install):**
- The Agent calls the `ImageGen` tool directly for text-to-image (returns a `file_path`).
- Great for quick concept sketches, covers, and as a fallback when ComfyUI/Ark isn't set up.
- No image-to-image input, no seed/model control. Then `imagine ingest` to file + ledger it.

**Tier 1 — ComfyUI orchestrator → cloud Ark/Jimeng (needs Phase 0 setup):**
- Text-to-image, image-to-image, multi-reference / group images via Seedream 3/4/5.
- Full control: model, size, seed, negative prompt, quota guard.
- Requires: a `comfy-cli` install of ComfyUI + the `ComfyUI-Jimeng-API` node + an `ARK_API_KEY`
  (in the node's `api_keys.json`). See [references/comfyui.md](references/comfyui.md).
- Until that setup exists, `imagine doctor` reports `server_reachable: False` and Tier 1 commands
  fail fast with a setup hint — Tier 0 still works.

**Always available (pure Python, offline):** `adjust` / `sheet` / `gallery` / `ledger` / `list` /
`palette` — Pillow post-processing and the ledger need no ComfyUI, no key, no network.

## Commands at a glance

| Command | Use when | Key output |
|---|---|---|
| `doctor` | Session start / anything broken | config + libs + ComfyUI reachability + provider + Tier 0 |
| `ingest --src I --slug S [--research R] [--gallery]` | Filing an already-generated image (ImageGen / external) | copied into `assets/` (or research `article/artwork/`) + ledger entry |
| `adjust SRC --out O [--crop --resize --rotate --mode --pad --format]` | Simple edits (crop/resize/rotate/convert/pad) | adjusted image |
| `sheet IMG… --out O [--cols --thumb]` | Seeing many images at once | contact-sheet PNG |
| `gallery [--add I --slug S] [--sheet]` | Curating aesthetic references | gallery listing / contact sheet |
| `ledger [--backend --model --contains --limit] [--stats] [--render]` | Recalling how an image was made | reproducible records table |
| `list [--research R]` | Inventory of assets/gallery/workflows (or a research's artwork) | file listing |
| `comfy doctor` | Checking the Tier 1 stack | `/system_stats` + Jimeng nodes + key presence |
| `comfy nodes [CLASS]` | Node schema unknown (**never guess**) | `/object_info` ground truth |
| `comfy server start/stop/status [--cpu] [--port]` | Managing the headless orchestrator | persistent ComfyUI + `runs/comfy_server.json` |
| `gen --prompt P [--model --size --seed --n --out] [--research --slug]` | Text-to-image | generated PNG(s) in `assets/` + ledger |
| `i2i --image I --prompt P [--seed --out]` | Image-to-image / editing (repeat `--image` for multi-reference) | generated PNG + ledger |
| `run --workflow W.json [--args JSON] [--out]` | Reusing a saved API-format workflow | generated PNG(s) + ledger |
| `workflows` | Listing saved workflow recipes | recipe inventory |
| `palette --src I [--n 6]` | Extracting a reference's dominant colors | hex palette (for the bridge) |
| `bridge --ref I --research R --slug S [--style --template]` | Turning an aesthetic ref into a real figure | scaffolded pipeline + `design_spec.md` |

Run `imagine <command> -h` for the full option list.

## Workflow A — aesthetic reference → data-accurate reproduction (the killer feature)

AI images are *beautiful but their data/labels are nonsense*. Use them as **design templates**, then
reproduce with real data via `scientific-plotting`:

```
- [ ] 1. imagine gen --prompt '<aesthetic cover/figure concept>' --seed 42   # or use Tier 0 ImageGen
- [ ] 2. Read the generated PNG                                             # judge composition/color
- [ ] 3. imagine gallery --add <png> --slug ref_cover                       # curate if reusable
- [ ] 4. imagine bridge --ref <png> --research gain_ep --slug fig1_cover    # extract palette + layout
- [ ] 5. Edit the scaffolded pipeline: fill in REAL data (bridge wrote design_spec.md with hexes/layout)
- [ ] 6. figures build 'data/research/1_gain_ep/article/figures/fig1_cover'
- [ ] 7. Read the _preview.png and compare against the reference; iterate
```

`bridge` uses Pillow color quantization to pull the dominant hexes, notes the aspect ratio /
composition, calls `scaffold_figure(...)`, and writes a `design_spec.md` (embedded reference + hex
palette + layout hints) into the figure dir. Optionally it registers the palette into `STYLE.yaml`
via scientific_plotting's `register_palette` (colorblind-safety still audited).

## Workflow B — direct artwork production (cover / graphical abstract / schematic)

For purely aesthetic deliverables (no data to be accurate):

```
- [ ] 1. imagine doctor                                    # Tier 1 reachable? else use Tier 0
- [ ] 2. imagine gen --prompt '<...>' --size 2K --seed 7 --research gain_ep --slug cover
       (Tier 0: call ImageGen, then imagine ingest --src <path> --slug cover --research gain_ep)
- [ ] 3. Read the PNG                                      # VISUAL CHECK
- [ ] 4. imagine adjust <png> --out <png> --resize 1600 0  # crop/resize/convert as needed
- [ ] 5. imagine ledger --research … (or --contains)       # recall prompt/seed to reproduce
```

## Visual closed loop

Every generation/edit prints a `✓ 视觉校验：Read '<path>'` line. **Read the PNG** with the Read tool
to actually *see* it, then iterate on the prompt/seed (Tier 1) or the adjust params. Never ask the
user to eyeball intermediate images — look yourself. This is the same render→Read pattern as the
comsol / figures / document skills. Use `sheet` / `gallery --sheet` to compare many candidates at once.

## The ComfyUI orchestrator (Tier 1)

ComfyUI is an **external** tool, discovered cheaply via `.env` (`COMFY_ROOT` / `COMFY_SERVER_URL`) —
pysci never imports it. Install it once with `comfy-cli` in its own venv, add the
`ComfyUI-Jimeng-API` node, put your `ARK_API_KEY` in the node's `api_keys.json`, then
`comfy launch --background -- --cpu`. `imagine comfy server start/stop/status` manages a persistent
headless server (state in `runs/comfy_server.json`, mirrors the comsol `server` pattern). The GUI on
the same `:8188` can be opened for live collaboration while the CLI drives it.

**Node schema is ground truth — never guess `class_type` strings or input names.** Run
`imagine comfy nodes [CLASS]` (hits `/object_info`) exactly like comsol's `inspect node` rule.

## Output locations

| Path | Contents |
|---|---|
| `data/skills/ai_drawing/assets/` | generated images (git-ignored: large, reproducible via ledger) |
| `data/skills/ai_drawing/gallery/` | curated aesthetic references (tracked, keep < 2MB each) |
| `data/skills/ai_drawing/workflows/` | saved ComfyUI API-format workflow recipes (`.json`) |
| `data/skills/ai_drawing/prompts/` | reusable scientific-aesthetic prompt recipes |
| `data/skills/ai_drawing/LEDGER.md` + `manifest.jsonl` | generation ledger (human + machine) |
| `data/skills/ai_drawing/cache/` `runs/` | transient previews / server state (git-ignored) |
| `data/research/<n>_<name>/article/artwork/` | a research line's finished artwork (cover / abstract / schematic) |

## When something breaks

1. `imagine doctor` — reports config, libs (PIL/requests/websocket), ComfyUI reachability, provider
   key presence, and the Tier 0 reminder.
2. Tier 1 command fails with "server not reachable": start it (`imagine comfy server start --cpu`) or
   check `.env`'s `COMFY_SERVER_URL`; confirm the Jimeng node + `api_keys.json` via `comfy doctor`.
3. Unknown node / input name: **never guess** — `imagine comfy nodes [CLASS]` for `/object_info` truth.
4. Wasted spend / can't reproduce a good image: `imagine ledger --contains '<kw>'` to recover the
   exact prompt + seed + model, then re-`gen` with `--seed`.
5. Blank/broken generation: check the Ark quota (free-tier exhaustion), the model ID, and the size
   string; the Quota guard node caps runaway batches.

## Reference files

- [references/backends.md](references/backends.md) — the backend decision matrix (Tier 0 ImageGen vs
  Tier 1 ComfyUI→Ark/Jimeng), why no local diffusion, why not comfy.org cloud nodes, provider options.
- [references/prompting.md](references/prompting.md) — scientific-aesthetic prompt recipes (cover /
  graphical abstract / schematic), negative prompts, seed/size/model guidance, cost discipline.
- [references/comfyui.md](references/comfyui.md) — ComfyUI API-format authoring, the Jimeng node
  contract, i2i / multi-reference / quota recipes, `comfy-cli` install, GUI collaboration.
- [references/bridge.md](references/bridge.md) — the aesthetic-reference → data-reproduction playbook.
