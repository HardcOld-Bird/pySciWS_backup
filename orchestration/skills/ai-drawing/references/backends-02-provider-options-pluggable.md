# backends 分册 2/2

> 含小节：Provider options (pluggable)；Getting an `ARK_API_KEY`；Configuration keys (`.env`)；Cost discipline
> 原 `backends.md` 超 8KB 预算拆分（2026-10-10，backlog 20261010-refs-split-8kb）；分册索引见同目录 `backends.md`，按需只读所需分册。

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
