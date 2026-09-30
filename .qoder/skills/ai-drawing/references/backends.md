# Backends — decision matrix

This skill deliberately does **not** run image diffusion locally. The machine has an Intel Iris Xe
iGPU (no CUDA), so local Stable-Diffusion/Flux would be slow to the point of unusable, and pulling
`torch` + model weights into pysci would bloat a scientific computing environment for one skill.
Instead we orchestrate **cloud** image models and keep pysci a thin HTTP + Pillow client.

## The two tiers

| | **Tier 0 — Qoder `ImageGen`** | **Tier 1 — ComfyUI → Ark/Jimeng** |
|---|---|---|
| What it is | Qoder Agent's built-in text-to-image tool | Headless ComfyUI as a *cloud-API workflow orchestrator*, calling Volcano Ark / Jimeng Seedream via the `ComfyUI-Jimeng-API` node |
| Install cost | **Zero** (already in the Agent) | `comfy-cli` install of ComfyUI + node + an `ARK_API_KEY` (separate venv, never in pysci) |
| Text-to-image | ✅ | ✅ (Seedream 3/4/5) |
| Image-to-image / edit | ❌ (no input-image parameter) | ✅ (feed an image input → auto i2i; multi-reference / group images) |
| Seed / model / size control | ❌ | ✅ |
| Negative prompt / quota guard | ❌ | ✅ (Jimeng Quota node) |
| Workflow reuse (API-format graph) | ❌ | ✅ (`workflows/` recipes, `run --workflow`) |
| Cost | bundled with the Agent | ~¥0.2 / image on Ark; free quota (≈200) first |
| Network | Agent-side | Ark is **mainland-China reachable directly** (no proxy) |
| Best for | quick concept sketches, covers, fallback | controlled, reproducible, editable, batchable production |

**Default path:** use Tier 0 for a fast first look / when Tier 1 isn't set up; use Tier 1 whenever
you need reproducibility (seed), image-to-image, multi-reference, or batch control. `imagine doctor`
tells you which tier is live right now.

## Why ComfyUI at all (if generation is in the cloud)?

ComfyUI is used **only** for its node-graph workflow engine + ecosystem, not its local samplers:

- A rich, maintained **node ecosystem** — the community `ComfyUI-Jimeng-API` node already implements
  the Ark/Jimeng HTTP contract, key management (`api_keys.json`), multi-reference images, group
  images, and a **quota guard**. We reuse it instead of re-implementing an Ark client.
- A stable, well-documented **REST/WS API** (`POST /prompt`, `GET /history`, `GET /view`,
  `POST /upload/image`, `GET /object_info`, `/ws`) — pysci drives it with plain `requests`.
- **Workflows as data**: an API-format graph (`{node_id: {class_type, inputs}}`) is a JSON recipe we
  can save, version, and re-parameterize (`run --workflow W.json --args '{...}'`).
- A **GUI on the same `:8188`** the user can open to watch/collaborate while the CLI drives the run.

## Why *not* comfy.org cloud nodes

ComfyUI ships official cloud/org nodes, but they route through comfy.org infrastructure that is
**blocked / needs a proxy** in the user's network environment. We instead use nodes that accept
**your own API key** and call providers reachable directly from mainland China (Volcano Ark first).

## Provider options (pluggable)

`providers.py` abstracts the image backend so new providers slot in without touching the CLI:

- **Volcano Ark / Jimeng Seedream** (default, first-class): `POST
  https://ark.cn-beijing.volces.com/api/v3/images/generations`, `Authorization: Bearer $ARK_API_KEY`,
  body `{model, prompt, size, output_format, watermark, response_format, image? (i2i)}`. Model IDs
  like `doubao-seedream-4-0-250828`. ~¥0.2/image, free quota, IPM 500. Requires real-name
  verification + model activation.
- **Reserved (not yet wired):** Alibaba Tongyi Wanxiang (DashScope), Zhipu CogView — same shape
  (key + REST), add a provider class when needed.

The key lives in the Jimeng node's `api_keys.json` (in the external comfy workspace). pysci's `.env`
only carries `ARK_API_KEY` for **existence probing / masked display** — never logged in clear.

## Configuration keys (`.env`, all optional with defaults)

| Key | Default | Meaning |
|---|---|---|
| `AI_DRAWING_BACKEND` | `comfyui` | default backend label recorded in the ledger |
| `COMFY_ROOT` | — | external ComfyUI workspace root (for `comfy server start`) |
| `COMFY_SERVER_URL` | `http://127.0.0.1:8188` | orchestrator URL (can point at a remote/cloud ComfyUI) |
| `COMFY_CLI` | — | explicit `comfy` executable path if not on `PATH` (for `server start/stop`) |
| `COMFY_JIMENG_KEY_NAME` | `Custom` | default `JimengAPIClient.key_name` for `gen`/`i2i` (matches an `api_keys.json` `customName`) |
| `COMFY_DEFAULT_MODEL` | `doubao-seedream-4-0-250828` | default image model ID |
| `AI_DRAWING_DEFAULT_SIZE` | `2K` | default Ark `size` (1K/2K/4K or `WxH`) |
| `ARK_API_KEY` | — | existence probe only (real key in node `api_keys.json`) |
| `AI_DRAWING_MAX_IMAGES` | `4` | cost guard: cap on images per generation |

## Cost discipline

- Default to **one** image; `--n` is explicit and capped by `AI_DRAWING_MAX_IMAGES`.
- Keep the **Quota guard node** in Tier 1 workflows to hard-cap runaway batches.
- Burn the **free quota** first; every spend is recorded in the ledger (`prompt`/`seed`/`model`) so a
  good result is **reproducible without re-rolling** — reuse the seed instead of paying again.
