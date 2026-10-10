# prompting 分册 2/2

> 含小节：Seed / size / model guidance；Cost discipline while prompting
> 原 `prompting.md` 超 8KB 预算拆分（2026-10-10，backlog 20261010-refs-split-8kb）；分册索引见同目录 `prompting.md`，按需只读所需分册。

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
