# Backends — decision matrix

This skill deliberately does **not** run image diffusion locally. The machine has an Intel Iris Xe
iGPU (1 GB shared VRAM, no CUDA), so local Stable-Diffusion/Flux would be slow to the point of
unusable, and pulling `torch` + model weights into pysci would bloat a scientific computing
environment for one skill. Instead we call **cloud** image models and keep pysci a thin client.

## The two tiers

| | **Tier 0 — Qoder `ImageGen`** | **Tier 1 — Volcano Ark direct** |
|---|---|---|
| What it is | Qoder Agent's built-in text-to-image tool | `ark_client.py`: a plain `requests` POST to Ark's OpenAI-compatible `/images/generations` |
| Install cost | **Zero** (already in the Agent) | **Zero** — `requests` is already a pysci core dependency; only an `ARK_API_KEY` in `.env` |
| Text-to-image | ✅ | ✅ (Seedream 3.0 / 4.0 / 4.5 / 5.0 / 5.0-pro) |
| Image-to-image / multi-reference | ❌ (no input-image parameter) | ✅ (local files auto-base64'd; ≤10 refs on 5.0-pro, ≤14 elsewhere) |
| Group (coherent multi-image) | ❌ | ✅ `--group` (`sequential_image_generation=auto`) |
| Marker-driven interactive editing | ❌ | ✅ `mark` → `edit` |
| Layer decomposition (base + alpha layers) | ❌ | ✅ `layers` (5.0-pro, ≤16 layers) |
| Model / size / seed / format control | ❌ | ✅ (seed is weak — see prompting.md) |
| Cost | bundled with the Agent | ~¥0.2 / image; burn the free quota first |
| Network | Agent-side | Ark is **mainland-China reachable directly** (no proxy) |
| Best for | quick concept sketches, covers, fallback | controlled, editable, batchable production |

**Default path:** Tier 0 for a fast first look or when no key is configured; Tier 1 whenever you need
reference images, editing, or model/size control. `imagine doctor` tells you which is live.

**Catalog ≠ activation.** `GET /models` (`imagine models --live`) lists the IDs the endpoint knows —
including **delisted** ones — so “visible in /models” never proves “my account may run it”.
Activation is per-model in the console 「开通管理」 and is observable elsewhere only by attempting a
call (a rejection at the authorization stage costs nothing; only successful images are billed). On
this account the live tiers are `doubao-seedream-5-0-flash-260915` (default) and
`doubao-seedream-5-0-pro-260628` (edit/layers); 4.0 is delisted and 5.0-lite is closing, which is
why group generation and direct PNG output are currently out of reach here.

## Local bitmap work — community libraries, not self-built

Everything that does **not** need a generative model stays local and offline:

| Need | Library | Install |
|---|---|---|
| crop / resize / rotate / pad / convert, contact sheets, palette extraction, marker overlay | **Pillow** | core dependency, always available |
| Poisson blending, algorithmic inpaint, mask generation (otsu/canny/grabCut), perspective warp | **OpenCV** | optional extra `imaging` |
| morphology, connected-component measurement, sub-pixel phase-correlation registration | **scikit-image** | optional extra `imaging` |

```powershell
uv pip install -e ".[imaging]"
```

The extra is **lazy-imported**: without it every `imagine img <op>` that needs OpenCV/scikit-image
prints the exact install command instead of a traceback, and the pure-Pillow paths (`img split`,
`img composite`, `adjust`, `sheet`, `mark`, `palette`) keep working. Tests are guarded per-library,
so installing only one of the two still runs that half.

This is the "prefer community infrastructure over self-built" principle applied: we wrote **no**
image-processing algorithms, only thin path/argument/CLI glue around `cv2` and `skimage`.

## Why not ComfyUI at all

An earlier revision of this skill ran a headless ComfyUI on `127.0.0.1:8188` purely as a
**cloud-API workflow orchestrator**, driving Ark through the community `ComfyUI-Jimeng-API` node.
That whole layer was **removed**. Reasons, in order of weight:

1. **The node graph bought nothing we actually used.** Every workflow we ever needed was a single
   HTTP POST: prompt (+ optional reference images) in, image bytes out. There was no branching, no
   intermediate tensor, no sampler chain — the "graph" was one node wide. We were paying an
   orchestration layer's complexity for a straight-line call.
2. **It cost a whole second environment.** `comfy-cli`, ComfyUI itself, `torch` (CPU), and a custom
   node — all outside pysci's venv, all with their own upgrade cadence, all of which had to be
   installed *and kept working* before a single image could be generated. A broken ComfyUI install
   made Tier 1 unusable even though the Ark key was fine.
3. **It moved the secret out of `.env`.** The Jimeng node keeps keys in its own `api_keys.json`, so
   `ARK_API_KEY` in `.env` was reduced to an existence probe while the *real* credential lived in a
   directory this repo doesn't manage. One key, one place (`.env`) is easier to reason about.
4. **Two failure surfaces instead of one.** A generation could now fail in pysci, in the ComfyUI
   server, in the node, or at Ark — and only the last one produced an Ark error code. Debugging
   meant `comfy doctor` + `/object_info` spelunking before you could even see the HTTP status.
5. **The genuine gap is real but narrower than "we need a node graph".** What the graph *would*
   have provided beyond a single call is (a) pipeline composition and (b) bitmap processing. (a) is
   covered by `recipes/*.md` knowledge cards + the Agent sequencing CLI calls; (b) is covered by
   OpenCV/scikit-image above — which do it **without a GPU, without model weights, and
   deterministically**, unlike a diffusion node.
6. **Direct Ark is not a downgrade in capability.** Interactive editing, layer decomposition,
   multi-reference and group generation are all first-class **Ark API** features (5.0-pro /
   5.0). The Jimeng node was itself only a wrapper over the same HTTP contract.

Net effect: ~4 source modules + ~4 test modules deleted, one new `requests`-based module (~560
lines) that owns the whole Ark contract including its guardrails, and no new core dependency.

## When to bring a node graph back

This is a **reversible** decision, not an ideology. Re-introduce an orchestrator (ComfyUI, or
something lighter) only when a concrete need appears that a linear CLI sequence cannot express:

- A step whose output must be **routed conditionally** into different downstream nodes (e.g.
  "measure the region; if area < threshold re-inpaint with a larger radius, else composite") that
  you want to run **unattended, repeatedly, at scale** — not once via the Agent.
- A pipeline that needs **local diffusion weights** (LoRA / ControlNet / img2img samplers) — which
  is also the point at which the hardware argument returns, so it implies a different machine.
- A need to **share a runnable graph** with a collaborator who has no pysci checkout.
- Ark introduces a feature that genuinely requires **multiple coordinated requests with
  intermediate state** (e.g. server-side session/canvas state across calls).

Until one of those is real, a linear CLI + `recipes/*.md` is cheaper to maintain. If you do add it
back, keep it exactly as before: **external process, separate venv, discovered via `.env`, never
imported by pysci**.

## Provider options (pluggable)

`ark_client.py` is the only provider module. It is deliberately **not** a registry:

- **Volcano Ark / Jimeng Seedream** (default, first-class):
  `POST {ARK_BASE_URL}/images/generations`, `Authorization: Bearer $ARK_API_KEY`,
  body `{model, prompt, size|width|height, image?, seed?, watermark, response_format,
  output_format?, sequential_image_generation?, max_images?, tools?}`.
  Model IDs are passed through **verbatim** — the Ark console 「模型列表」 is the single source of
  truth. This repo keeps no local model registry, because a registry rots every time Ark ships a
  new tier. `imagine models` prints a *capability* matrix (human guidance), not an allow-list.
- **Other OpenAI-compatible image providers** (Alibaba Tongyi Wanxiang / DashScope, Zhipu CogView):
  same shape (key + REST). If one is ever needed, point `ARK_BASE_URL` at it first and see whether
  the contract matches; only add a second module if the request/response shape genuinely differs.

## Getting an `ARK_API_KEY`

Tier 1 needs exactly one secret. To obtain it:

1. Register / sign in at the **Volcano Engine (火山引擎)** console — <https://console.volcengine.com/>.
2. Complete **real-name verification (实名认证)**. Ark will not serve image models without it; this
   is the step that most often blocks a first attempt, and it can take a little while.
3. Open **方舟大模型服务 (Ark)** — <https://console.volcengine.com/ark> — and activate it.
4. In **模型广场 / Model Square**, find the **即梦 Seedream (doubao-seedream-\*)** image models and
   **开通 (activate)** the tier you want. Image generation models are activated *individually*;
   an un-activated model ID returns an authorization error even with a valid key.
   Note the exact **Model ID** string shown there — that is what `--model` / `AI_DRAWING_MODEL` takes.
5. In **API Key 管理**, click **创建 API Key**. The secret is shown **once** — copy it immediately.
6. Put it in the **project-root `.env`** (never in `data/`, never in the repo):

   ```dotenv
   ARK_API_KEY=<your-key>
   ```

7. Verify without spending anything:

   ```powershell
   uv run pysci-imagine doctor
   uv run pysci-imagine gen --prompt 'test' --dry-run
   ```

   `doctor` prints the key **masked** (`abc***xyz`) and the resolved endpoint; `--dry-run` prints the
   full request body and sends nothing. The key is never written to the ledger, to `runs/`
   snapshots, or to stdout.

Billing: Ark image generation is metered per output image (~¥0.2/image at the time of writing) with
a free quota on activation. Check the console 「费用中心」 for current prices — do not trust this
number, it drifts.

## Configuration keys (`.env`)

Only `ARK_API_KEY` is required; everything else has a working default.

| Key | Default | Meaning |
|---|---|---|
| `ARK_API_KEY` | — | **The** credential; required for Tier 1 only. `.env` is git-ignored |
| `ARK_BASE_URL` | `https://ark.cn-beijing.volces.com/api/v3` | OpenAI-compatible base; `/images/generations` is appended |
| `AI_DRAWING_BACKEND` | `ark` | default backend label recorded in the ledger (`ark` \| `imagegen`) |
| `AI_DRAWING_MODEL` | `doubao-seedream-5-0-flash-260915` | default Model ID (pass-through string) |
| `AI_DRAWING_DEFAULT_SIZE` | `2K` | default `size` (`1K`/`2K`/`3K`/`4K` or `WxH`) |
| `AI_DRAWING_MAX_IMAGES` | `4` | **cost guard**: cap on images per generation / on `--n` fan-out |

Removed with the ComfyUI layer (do not re-add unless the layer comes back): `COMFY_ROOT`,
`COMFY_SERVER_URL`, `COMFY_CLI`, `COMFY_JIMENG_KEY_NAME`, `COMFY_DEFAULT_MODEL`.

## Cost discipline

- Default to **one** image. `--n` is explicit and clamped by `AI_DRAWING_MAX_IMAGES`; `--dry-run`
  reports the clamp before you pay.
- `--dry-run` is free and keyless — use it for every new flag combination.
- `--response-format b64_json` is the default on purpose: Ark `url` results **expire after 24 h**,
  so a ledger entry pointing at a URL rots. Bytes come home immediately.
- `watermark: false` is sent explicitly (Ark's default is `true`).
- Every spend is recorded in the ledger with prompt / model / usage / a redacted request snapshot —
  but that is **provenance, not reproduction**. Ark does not guarantee the same image from the same
  prompt+seed, so `gallery --add` the winner; the ledger cannot conjure it back.
