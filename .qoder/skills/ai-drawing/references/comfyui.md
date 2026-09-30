# ComfyUI orchestrator (Tier 1)

Tier 1 uses a **headless ComfyUI** purely as a *cloud-API workflow orchestrator*. It runs on
`127.0.0.1:8188` with `--cpu` (no local diffusion — the box has no CUDA GPU), and via the community
**`ComfyUI-Jimeng-API`** node it calls **Volcano Ark / Jimeng Seedream** with *your own* API key.
pysci talks to it over plain HTTP (`requests`) + optional WebSocket (`websocket-client`), and never
imports ComfyUI, torch, or the node.

This file is the operational contract: how to set it up, the REST/WS API pysci drives, the
**verified** Jimeng node schema, and ready-to-use recipes.

---

## Phase 0 setup (one-time, in a *separate* environment)

> Everything below lives **outside** pysci's venv. ComfyUI + torch + the custom node are managed by
> `comfy-cli` in their own workspace so a heavy image stack never bloats the scientific environment.

### 1. Install `comfy-cli` in its own venv

```powershell
python -m venv D:\comfy-venv ; D:\comfy-venv\Scripts\Activate.ps1
pip install comfy-cli
comfy install --cpu        # CPU-only torch; installs ComfyUI into a workspace (default ~\comfy)
comfy which                # print the workspace path  → this is COMFY_ROOT
```

### 2. Install the `ComfyUI-Jimeng-API` node

```powershell
comfy install-node https://github.com/kithara/ComfyUI-Jimeng-API    # or via `comfy manager` in the GUI
```

(Confirm the exact repo/menu name at install time — the node appears under the **`JimengAI`** menu.)

### 3. Configure `api_keys.json` (the node reads your Ark key here)

In the node's directory (`<COMFY_ROOT>/custom_nodes/ComfyUI-Jimeng-API/`), copy
`api_keys.json.example` → `api_keys.json`:

```json
[
  { "customName": "ark-main", "apiKey": "<YOUR_ARK_API_KEY>" }
]
```

- `customName` is what you select via `JimengAPIClient.key_name` (and pass with `--key-name`).
- The key **never** enters pysci or the repo. `api_keys.json` is git-ignored by the skill's
  `.gitignore` and lives in the external workspace anyway.
- Get the Ark key from Volcano Engine → 火山方舟 → API Key (needs real-name verification + Seedream
  model activation; there is a free quota ≈ 200 images).

### 4. Launch headless

```powershell
comfy launch --background -- --cpu     # starts on :8188; GUI available at the same URL
comfy stop                             # when done (frees RAM)
```

Or let pysci manage it: `imagine comfy server start --cpu` / `stop` / `status` (needs `comfy` on
`PATH`, or set `COMFY_CLI=<path to comfy exe>`). State is tracked in
`data/skills/ai_drawing/runs/comfy_server.json` so consecutive CLI calls reuse one server.

### 5. Wire pysci (`.env` at the project root — all optional, sane defaults)

```dotenv
COMFY_ROOT=D:\comfy                       # workspace root (from `comfy which`); enables `server start`
COMFY_SERVER_URL=http://127.0.0.1:8188    # orchestrator URL (can be a remote/cloud ComfyUI)
COMFY_CLI=D:\comfy-venv\Scripts\comfy.exe # optional: explicit comfy-cli path if not on PATH
COMFY_JIMENG_KEY_NAME=ark-main            # optional: default JimengAPIClient.key_name for gen/i2i
ARK_API_KEY=<optional>                    # existence probe / masked display ONLY (real key in api_keys.json)
AI_DRAWING_MAX_IMAGES=4                   # cost guard: cap images per generation
```

### 6. Verify

```powershell
uv run pysci-imagine comfy doctor
```

Expect: `port_alive: True`, `comfyui_version`, the device list, all `Jimeng*` nodes `OK`, and
`api_keys.json 配置 : 已配置 N 个` (listing your `customName`s). If `key_name` shows only `Custom`,
`api_keys.json` isn't picked up — recheck step 3.

---

## The ComfyUI HTTP/WS API (what pysci drives)

| Endpoint | Purpose |
|---|---|
| `GET /system_stats` | version + devices; cheap reachability probe (`comfy doctor`) |
| `GET /object_info[/{class_type}]` | **node schema ground truth** — never guess `class_type`/inputs |
| `POST /upload/image` | multipart upload of an i2i source image → `{name, subfolder, type}` |
| `POST /prompt` `{"prompt": graph, "client_id": cid}` | submit an API-format graph → `prompt_id` |
| `GET /history/{prompt_id}` | execution result (`status` + `outputs`) |
| `GET /view?filename=&subfolder=&type=` | fetch a produced image's bytes |
| `GET /queue` / `POST /interrupt` | queue status / cancel current run |
| `WS /ws?clientId=cid` | live progress (`executing`/`progress`/`executed`/`status`) |

pysci's `comfy_client.ComfyClient` wraps these. It prefers the WebSocket for progress and
**automatically falls back to polling `/history`** when `websocket-client` isn't installed (install
the extra with `uv sync --extra comfy` for streaming progress — functionality is identical either
way). `POST /prompt` validates the graph server-side; on failure it returns `node_errors`, which
pysci surfaces verbatim (this is the first-place diagnostic for a wrong `class_type`/input name).

---

## API-format workflows (the graph pysci submits)

ComfyUI has two serializations: the **UI format** (what the GUI saves — `nodes`/`links` + canvas
coords) and the **API format** (what `POST /prompt` eats). pysci builds and submits **API format**:

```json
{
  "1": { "class_type": "JimengAPIClient",
         "inputs": { "key_name": "ark-main", "new_api_key": "", "new_key_name": "" } },
  "2": { "class_type": "JimengSeedream4",
         "inputs": { "client": ["1", 0], "model_version": "doubao-seedream-4.0",
                     "prompt": "a journal cover …", "size": "2K (adaptive)",
                     "width": 2048, "height": 2048, "seed": 42,
                     "enable_group_generation": false, "max_images": 1,
                     "generation_count": 1, "thinking": true, "watermark": false } },
  "3": { "class_type": "SaveImage",
         "inputs": { "images": ["2", 0], "filename_prefix": "Jimeng/Image/Seedream4" } }
}
```

**Link encoding:** a connection is `["<source_node_id>", <output_slot_index>]`. Above, `"client":
["1", 0]` wires node 1's output slot 0 (the `client`) into node 2's `client` input; `"images":
["2", 0]` wires the generated images into `SaveImage`. A `SaveImage` (an OUTPUT_NODE) is what makes
the graph actually execute.

`imagine gen/i2i` build this programmatically (`workflows.py`); `--dry-run` prints it without
submitting; `--save-workflow NAME` writes it to `data/skills/ai_drawing/workflows/NAME.json`.

---

## The Jimeng node contract (verified — but `/object_info` is the ultimate truth)

Values below were read from the `ComfyUI-Jimeng-API` source (menu `JimengAI`). **The server's
`GET /object_info` always wins** — confirm with `imagine comfy nodes <CLASS>` before relying on an
exotic input, especially after a node update.

### `JimengAPIClient` (every workflow's entry point)
- Inputs/widgets: `key_name` (COMBO; default `"Custom"`; options = the `customName`s in
  `api_keys.json`), `new_api_key` (STRING; for ad-hoc keys via the GUI), `new_key_name` (STRING).
- Output: `client` (`JIMENG_CLIENT`, slot 0).
- pysci passes `key_name` only — **never** a raw key.

### `JimengSeedream4` (text-to-image; add an image input → auto image-to-image)
- Inputs: `client` (link), `model_version` (COMBO), `prompt` (STRING), `size` (COMBO),
  `width`/`height` (INT, default 2048 — **only used when `size == "Custom"`**), `seed` (INT; `0`
  default, `-1` → node picks random), `enable_group_generation` (BOOL), `max_images` (INT 1–15),
  `generation_count` (INT), `thinking` (BOOL; prompt-optimization, **4.0 only**), `watermark` (BOOL).
- Optional autogrow input `images` (`image_1 … image_14`) — feed reference images for i2i / multi-ref.
- Outputs: `images` (slot 0), `response` (slot 1, raw JSON).
- Per-image seed: the node uses `current_seed = seed + index` within a batch — pysci records each
  image's `seed+i` in the ledger so any single result is reproducible.

### Model versions ↔ Ark API IDs
| `model_version` (node COMBO) | Ark API ID | Node |
|---|---|---|
| `doubao-seedream-4.0` | `doubao-seedream-4-0-250828` | `JimengSeedream4` (default) |
| `doubao-seedream-4.5` | `doubao-seedream-4-5-251128` | `JimengSeedream4` |
| `doubao-seedream-5.0-pro` | `doubao-seedream-5-0-pro-260628` | `JimengSeedream5` |
| `doubao-seedream-5.0-lite` | `doubao-seedream-5-0-260128` | `JimengSeedream5` |
| `doubao-seedream-3.0-t2i` | `doubao-seedream-3-0-t2i-250415` | `JimengSeedream3` |

`imagine gen --model` accepts the UI version, the API ID, or a shorthand (`4.0`, `4`, `seedream-4`).
**Only `JimengSeedream4` is wired** in pysci's builders right now (`providers.py`); Seedream 3/5 use
a different schema (3 has `guidance_scale`, no `model_version`/`thinking`; 5 uses a `DynamicCombo`)
— for those, author the graph in the GUI and drive it with `run --workflow` (below).

### `size` options (`RECOMMENDED_SIZES_V4`)
`"2K (adaptive)"`, `"4K (adaptive)"`, `"2048x2048 (1:1)"`, `"2304x1728 (4:3)"`, `"1728x2304 (3:4)"`,
`"2848x1600 (16:9)"`, `"1600x2848 (9:16)"`, `"2496x1664 (3:2)"`, `"1664x2496 (2:3)"`,
`"3136x1344 (21:9)"`, `"4096x4096 (1:1)"`, `"Custom"`. The node executes with `size.split(" ")[0]`
(`"2K (adaptive)"` → Ark `size:"2K"`). `imagine gen --size` normalizes prefixes (`2K` → `2K (adaptive)`).

### `JimengQuotaSettings` (cost guard)
- Inputs: `client` (link), `image_model` (COMBO), `image_limit` (INT; `0` = no cap), `video_model`,
  `video_limit`. Output: `status` (STRING).
- It writes limits into a **process-level singleton** (`QuotaManager`, keyed by api_key); generation
  nodes call `check_quota` before spending.
- ⚠️ Its `status` output, if not wired to an OUTPUT_NODE, may be **pruned** and not execute within a
  given graph. So pysci's **hard** guard is `AI_DRAWING_MAX_IMAGES` + the default of one image; the
  quota node is an optional server-side second lock — run it once on its own to register the cap.

---

## Recipes

### Text-to-image (`gen`)
```powershell
uv run pysci-imagine gen --prompt 'a journal cover: non-Hermitian skin effect, glowing chiral arcs, dark navy, minimalist' --size '2K (adaptive)' --seed 42 --research gain_ep --slug cover
```
Preview the graph for free first with `--dry-run`; save it as a recipe with `--save-workflow cover_v1`.

### Image-to-image / multi-reference (`i2i`)
```powershell
uv run pysci-imagine i2i --image data/skills/ai_drawing/gallery/ref_a.png --image ref_b.png --prompt 'keep composition, recolor to a cool palette' --seed 7
```
Each `--image` is uploaded via `POST /upload/image`, then wired to `image_1…image_N` on the
Seedream node (reference-based editing — Seedream i2i has **no denoise/`--strength`** knob).

> **Autogrow `images` serialization caveat:** pysci emits the nested form `"images": {"image_1":
> ["2", 0], …}`. If a future ComfyUI version rejects it (`/prompt` returns `node_errors`), confirm
> the real shape with `imagine comfy nodes JimengSeedream4`, or export the graph from the GUI
> (**Save (API Format)**) and run it with `run --workflow`.

### Group images (a coherent set in one call)
```powershell
uv run pysci-imagine gen --prompt 'four matching panel illustrations of the same scene, consistent style' --group --max-images 4
```

### Reusing a GUI-exported workflow (`run --workflow`) — the universal escape hatch
Author any graph in the GUI (Seedream 3/5, extra nodes, exotic wiring), **Save (API Format)** into
`data/skills/ai_drawing/workflows/`, then re-parameterize and run headless:
```powershell
uv run pysci-imagine run --workflow my_s5_graph --args '{"prompt":"new idea","seed":99}' --research gain_ep
uv run pysci-imagine run --workflow my_s5_graph --args '{"3.seed":99,"3.size":"4K (adaptive)"}'
```
`--args` accepts **friendly keys** (`prompt`/`seed`/`size`/`width`/`height`/`model`/`n`/`watermark`/
`thinking`, auto-applied to the detected Seedream node) or **dot keys** (`"<node_id>.<input>"`) for
precise targeting.

> **PowerShell `--args` quoting (important):** PowerShell 5.1 mangles embedded double quotes in
> native-command args, so `--args '{"seed":99}'` often arrives as `{seed:99}` (invalid JSON).
> Workarounds: (a) use the stop-parsing token — `--% --args "{\"seed\":99}"`; (b) run under
> PowerShell 7+; or (c) keep the graph in a workflow file and edit values there. Verify with
> `--dry-run` (prints the final graph without spending).

---

## Node introspection rule (never guess)

Exactly like comsol's `inspect node`: **`class_type` strings and input names are ground truth from
the server, not memory.** Before building an unfamiliar node:
```powershell
uv run pysci-imagine comfy nodes                 # list all Jimeng* class_types
uv run pysci-imagine comfy nodes JimengSeedream4 # dump its required/optional inputs + specs
```

---

## GUI collaboration

The headless server exposes the **same GUI on `:8188`**. Open `http://127.0.0.1:8188` in a browser
to watch the queue, inspect nodes, tweak a graph visually, or **Save (API Format)** a recipe for
`run`. The CLI and GUI share one server, so you can drive a run from the terminal while the user
watches it render live.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `comfy doctor` → `port_alive: False` | Launch it (`comfy launch --background -- --cpu` or `imagine comfy server start --cpu`); check `COMFY_SERVER_URL`. |
| `Jimeng*` nodes `MISSING` | The node isn't installed in this workspace — redo step 2, restart the server. |
| `key_name` shows only `Custom` | `api_keys.json` not found/parsed — redo step 3 (correct dir, valid JSON), restart. |
| `/prompt` → `node_errors` | A `class_type`/input name is wrong or the i2i `images` shape changed — `imagine comfy nodes <CLASS>`, or switch to `run --workflow` with a GUI export. |
| Blank/failed image | Ark free-quota exhausted, wrong model ID, or bad `size` string; check the Quota node cap and `imagine ledger` for the last spend. |
| No streaming progress | `websocket-client` missing — `uv sync --extra comfy` (or ignore; polling works). |
| Wasted spend | Recover the exact `prompt`/`seed`/`model` with `imagine ledger --contains '<kw>'`, then re-`gen --seed`. |
