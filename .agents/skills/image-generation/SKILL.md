---
name: image-generation
description: ⑦AI 生图能力域操作面：纯云生成链（prompt 配方→方舟 Seedream 主/硅基流动副→出图护栏→Pillow/rembg 后处理→账本溯源 PNG tEXt）工作流；5.5 透明底直出、5.0-pro 图层拆分、组图/参考图风格一致、连通域量化与亚像素配准。需要生成科研示意图/概念图/艺术配图/封面/graphical abstract、抠透明底、批量出图记账审计、量化或配准生成图时使用。数据图（曲线/场图/统计图）归③科研绘图，不走本域。
---

# Image Generation（⑦AI 生图）

AI 生图能力域的操作面：**纯云生成**（本机仅 CPU ~5GB 内存，本地扩散不可行）——
方舟 Seedream **主通道** + 硅基流动**副通道**，本地只做轻后处理（Pillow + rembg）。
选型与用户裁决见 `sub_root/others/community_research/2026-11-17/domain-07-imagegen-selection.md`；
本技能只放"怎么用"。薄胶水在 `sub_root/src/pysci/skills/image_generation/`
（providers / guardrails / ledger / imaging / postprocess，纯 API 无 CLI 壳）。

> **API 状态注记（2026-11-17）**：客户端按两家**官方文档**实现（方舟图片生成教程 +
> 硅基流动 OpenAPI schema），**真实出图行为未实测**——`ARK_API_KEY` 已过期，
> 真实冒烟挂起待 key 更新（脚本就绪：`tests/skills/image_generation/smoke_real.py`）。
> 方舟 5.5 参数/价目以控制台为唯一真值源。

## 域边界（先明确）

- **⑦ = 生成式图像**：示意图 / 概念图 / 艺术配图 / 封面 / graphical abstract / 图层素材。
- **③ = 数据图**（曲线/场图/统计图）走 `scientific-plotting` 出版链；**AI 图不作数据证据**，
  只作示意/美学用途（红线延续旧账本原则）。
- 与⑤衔接：产物落研究线 `data/research/<n>_<name>/article/artwork/`，文档/幻灯直接插图。

## 能力分层（谁负责什么）

1. **方舟 Seedream（主）**：文生/图生/组图；**5.5 `background:"transparent"` 透明底直出**；
   **5.0-pro 图层拆分**（1 底图 + ≤16 alpha 层）与交互编辑（mark 画框/箭头/标签）；
   组图 `sequential_image_generation`（4.0/4.5/5.0 主档，参考+产出 ≤15）；
   size 1K/2K/3K/4K；联网增强 tools=web_search（5.0）。
2. **硅基流动（副）**：Qwen-Image / Qwen-Image-Edit、FLUX.2-pro/flex、FLUX.1-Kontext 系
   （改图不崩族）、FLUX-1.1-pro(-Ultra)、Z-Image-Turbo、MOSS-Image（文字渲染/复杂布局）；
   **seed 全档生效（科研可复现优势）**；batch_size ≤4；无透明底参数。
3. **本地轻后处理**：Pillow（裁剪/缩放）+ **rembg**（26k★ 任意图抠图/透明底兜底，
   u2net 权重已 vendor 至 `deps_management/deps/cache/rembg`）。
4. **薄胶水** `pysci.skills.image_generation`：双云统一客户端、出图护栏、生成账本
   （manifest.jsonl + PNG tEXt 溯源）、imaging（measure/align）、postprocess。

## 生成链工作流（六步）

| 步 | 做什么 | 入口 |
|---|---|---|
| 1 | 配 prompt（主体/风格/构图/负面） | 配方库 → [references/prompt-recipes.md](references/prompt-recipes.md) |
| 2 | 选通道/模型 | 示意/概念/组图→方舟 5.5/5.0-pro；文字重版式→MOSS-Image；改图→Kontext/Qwen-Edit |
| 3 | 护栏判定 | `guardrails.check_seed/check_images_budget`（细则 → [references/generation-guardrails.md](references/generation-guardrails.md)） |
| 4 | 出图 | `providers.generate(provider, prompt, ...)` → `save_images`（url 立即下载） |
| 5 | 后处理 | 透明底路线 → [references/transparent-background.md](references/transparent-background.md)；`postprocess.crop_resize` |
| 6 | 账本溯源 | `ledger.record`（manifest.jsonl）+ `ledger.embed_metadata`（PNG tEXt `pysci:*`）→ **read_image 目检** |

**Always read_image 目检产物再交付**（视觉策略 2.4）——生成图的语义错误（文字、比例、
结构）只有看图能发现；不达标改 prompt 重生成，别只靠后处理硬修。

## 薄胶水 API 速查

```python
from pysci.skills.image_generation import providers, guardrails, ledger, postprocess, imaging

resp = providers.generate("ark", "a minimal schematic of ..., flat vector style",
                          model="doubao-seedream-5-5", size="2K", transparent=False)
paths = providers.save_images(resp, "data/research/1_demo/article/artwork", stem="fig0")
ledger.record(prompt="...", model=resp.model, provider="ark", out=paths[0],
              extra={"url_ttl": guardrails.url_ttl_note("ark")})
ledger.embed_metadata(paths[0], {"prompt": "...", "model": resp.model,
                                 "provider": "ark", "ts": "..."})
postprocess.remove_background("art.png", "art_cut.png")   # rembg 兜底
postprocess.verify_alpha("art_cut.png")                   # alpha max 必须 255
imaging.measure_regions("fig.png", min_area=50)           # 连通域量化
imaging.align("layer.png", "base.png", dest="layer_fix.png")  # 亚像素配准
```

- key 读项目根 `.env`：`ARK_API_KEY` / `SILICONFLOW_API_KEY` / `SILICONFLOW_BASE_URL`。
- **401（key 过期/无效）→ `providers.ProviderError` 优雅报错**（带换 key 提示），不裸崩。

## 出图护栏速查（详见 references/generation-guardrails.md）

- **watermark 恒显式 false**——方舟默认加水印；要水印显式 `allow_watermark=True`。
- **seed 兼容档**：方舟 [0,65535] 仅 3.0-t2i 生效且官方声明非确定→只记账不作复现依据；
  硅基 [0,9999999999] 全档生效→可复现。
- **b64 规避 URL 失效**：方舟默认 `b64_json`（url 24h 失效）；硅基仅出 url（**1h 失效**）→ 立即下载。
- **成本护栏**：单次张数 clamp（默认 4，`IMAGEGEN_MAX_IMAGES` 覆盖）；方舟组图参考+产出 ≤15。
- 单张失败走 `data[i].error`（不整请求失败）；模型 ID 字符串直传，控制台为真值源。

## References（渐进加载，只读需要的那一册）

- [references/prompt-recipes.md](references/prompt-recipes.md) —
  prompt 配方库：科研示意图结构、场景配方（封面/graphical abstract/图标贴纸/组图）、风格一致性三招。
- [references/generation-guardrails.md](references/generation-guardrails.md) —
  出图护栏清单：watermark/seed 兼容档/b64 与 URL 时效/成本上限/参数映射对照表/踩坑实录。
- [references/transparent-background.md](references/transparent-background.md) —
  透明底策略：云直出优先（5.5 background）→ 5.0 图层拆分 → rembg 兜底 → alpha 验证口径。
- [references/imaging.md](references/imaging.md) —
  measure 连通域量化 / align 相位相关亚像素配准：字段口径、用途、限制与实测精度。
