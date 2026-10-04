---
name: ai-drawing
description: Generate and edit images with Volcano Ark / Jimeng Seedream cloud models, plus local bitmap processing (Pillow; scikit-image + OpenCV via the optional `imaging` extra), a generation ledger with PNG-embedded provenance, and an aesthetics→data bridge into scientific_plotting. Two tiers — Tier 0 Qoder's built-in ImageGen; Tier 1 a direct Ark HTTP client (text-to-image, image-to-image, multi-reference, marker-driven editing, layer decomposition). No local diffusion, no orchestrator. Drives a unified `imagine` CLI. Use when the user asks to generate or draw an image, create a journal cover / graphical abstract / schematic, produce an aesthetic reference to later reproduce with real data, edit or retouch a generated image, mark up regions and edit only there, split artwork into layers and recomposite it, seamlessly blend a generated element into a photo or field plot, remove a defect or watermark, measure / align / morphologically clean a bitmap, extract a color palette from a reference image, or bridge an AI artwork into a data-accurate figure.
---

# AI Drawing

One CLI drives the whole loop: **generate → look → curate → adjust → verify**, plus an
**aesthetics → data bridge** turning a beautiful-but-fake AI reference into a real, data-accurate
figure pipeline.

**Key design idea:** no local diffusion (no CUDA GPU) and no orchestrator. Generative work is one
plain `requests` POST to **Volcano Ark / Jimeng Seedream**; everything non-generative runs locally on
**Pillow / OpenCV / scikit-image** — no node graph, no server, no second venv, no `torch`. Why, and when
to bring an orchestrator back: [backends.md](references/backends.md). The backend in
`src/pysci/skills/ai_drawing/` is a **black box**: drive it through the CLI.

## Invocation

From the **project root**: `uv run pysci-imagine <command> [options]`, below `imagine …` (fallback:
`uv run --no-sync python -m pysci.skills.ai_drawing.tools.imagine …`). This CLI prints Chinese — apply
`.qoder/rules/basic.md` §3 or you read mojibake. Setup: `ARK_API_KEY=<key>` in the project-root `.env`
([walkthrough](references/backends.md#getting-an-ark_api_key)); OpenCV / scikit-image operators also
need `uv pip install -e ".[imaging]"`.

> **Cost rule:** every Tier 1 `gen` / `i2i` / `edit` / `layers` call spends real money (~¥0.2/image).
> Default to **one** image; `--n` is explicit and clamped. Preview new flags with `--dry-run` — free,
> keyless, prints the exact request body. `Read` the result before spending again.

> **Retention rule:** Ark does **not** guarantee the same image from the same prompt+seed, so the ledger
> is **provenance, not a reproduction recipe**. The moment you `Read` a result worth keeping, run
> `imagine gallery --add <png> --slug S` — that curated copy is the only reliable way to still have it.

## Capability boundary

- **Tier 0 — Qoder built-in `ImageGen`** (zero install, no key): call the tool directly for
  text-to-image, get a `file_path`. Concept sketches, covers, and the fallback with no key. No input
  image, no model/size control. Then `imagine ingest` to file + ledger it.
- **Tier 1 — Ark direct** (needs only `ARK_API_KEY`): text-to-image, image-to-image, multi-reference,
  marker-driven editing, layer decomposition. On **this account** only 5.0-flash (default) and 5.0-pro
  are live; coherent groups and direct PNG need delisted tiers. **Never guess capabilities:**
  `imagine models` prints the matrix, `models --live` the catalog (**catalog ≠ activation**).
- Everything else is **pure Python, offline, keyless** except these seven, which need the `imaging`
  extra: `fuse` `inpaint` `mask` `morph` `warp` `measure` `align`.

## Commands

| Group | Commands | For |
|---|---|---|
| health | `doctor` · `models [--live]` | session start / anything broken; capability matrix |
| inventory | `list` · `ledger` · `gallery [--add I --slug S] [--sheet]` | what exists; provenance; **retaining** a winner |
| generate | `gen --prompt P` · `i2i --image I --prompt P` | text-to-image / coherent group; image-to-image / multi-reference |
| edit | `mark --src I` · `edit --image MARKED --prompt P` · `layers --image I --prompt P` | annotate **before** `edit` (auto A/B/C labels); marker-driven edit; ≤16 alpha layers |
| local bitmap | `adjust` · `sheet` · `palette` · `img split｜composite｜fuse｜inpaint｜mask｜morph｜warp｜measure｜align` | offline crop/resize/rotate, contact sheet, palette, and the seven `img` ops |
| file | `ingest --src I [--slug S] [--research R]` | file an ImageGen / external image + ledger it |
| bridge | `bridge --ref I --research R --slug S` | aesthetic ref → real figure pipeline |

Full flags for all 24: [cli.md](references/cli.md#full-command-table), or `imagine <cmd> -h`.

## Workflow A — aesthetic reference → data-accurate reproduction (the killer feature)

AI images are *beautiful but their data/labels are nonsense*. Use them as **design templates**, then
reproduce with real data via `scientific-plotting`:

```
- [ ] 1. imagine gen --prompt '<aesthetic cover/figure concept>' --size 2K   # or use Tier 0 ImageGen
- [ ] 2. Read the generated PNG; then imagine gallery --add <png> --slug ref_cover   # judge, RETAIN
- [ ] 3. imagine bridge --ref <png> --research gain_ep --slug fig1_cover     # palette + layout + scaffold
- [ ] 4. Fill the scaffolded pipeline with REAL data (bridge wrote design_spec.md: hexes + layout)
- [ ] 5. figures build 'data/research/1_gain_ep/article/figures/fig1_cover'
- [ ] 6. Read the _preview.png, compare against the reference, iterate
```

Step 3's mechanics + worked example: [bridge.md](references/bridge.md). **Workflows B–E:**
[workflows.md](references/workflows.md) — **B** direct artwork · **C** `mark` → `edit` (Ark has **no
per-region parameter** — the binding lives in the prompt text) · **D** layers → recomposite (only
step 1 costs) · **E** local cleanup / blend / measure (free). Gotcha: `fuse` transfers **gradients, not
absolute color** — a flat element on a flat background **vanishes**; to *paste*, use `img composite`.

## Visual closed loop

Every generation/edit prints a `✓ 视觉校验：Read '<path>'` line. **Read the PNG** to actually *see* it,
then iterate on the prompt (Tier 1) or the adjust/img params — never ask the user to eyeball
intermediates. `sheet` / `gallery --sheet` compare many in one `Read`.

## Output locations

Under `data/skills/ai_drawing/`, what matters is `assets/` (git-ignored, **not** regenerable from the
ledger) versus `gallery/` (the tracked curated copy). Finished artwork for a research line goes to
`data/research/<n>_<name>/article/artwork/`. Path table and the double provenance record:
[cli.md](references/cli.md#output-paths-and-provenance-stored-twice).

## When something breaks

`imagine doctor` first. The three that matter: **"未配置 ARK_API_KEY"** → add it to `.env` ·
**401 / 403 with a valid key** → the *model isn't activated* (Ark activates each Seedream tier
individually in 模型广场; a rejected 1K probe is the cheapest test — only successes are billed) ·
**400 on flags** → re-run with `--dry-run`, read the body against `imagine models`. Everything else:
[cli.md](references/cli.md#troubleshooting-in-full).

## Reference files

- [cli.md](references/cli.md) — every flag for all 24 commands, output paths, provenance, troubleshooting.
- [workflows.md](references/workflows.md) — checklists B (artwork), C (`mark`→`edit`), D (layers), E (local bitmap).
- [backends.md](references/backends.md) — Tier 0 vs Tier 1 matrix, **why ComfyUI was removed and when to bring an orchestrator back**, `ARK_API_KEY` walkthrough, cost.
- [prompting.md](references/prompting.md) — prompt recipes (cover / graphical abstract / schematic / bridge ref), negative constraints (no `--negative` flag), prompting for `edit` / `layers`, seed/size/model truth.
- [bridge.md](references/bridge.md) — the aesthetic-reference → data-reproduction playbook and its deliberate limits.
