# 出图护栏清单（⑦域）

> 薄胶水实现：`pysci.skills.image_generation.guardrails`（providers.build_payload
> 内部已接线）。每条护栏对应一个真实事故面；**API 契约按官方文档实现，真实行为
> 未实测**（key 过期，冒烟挂起）——5.5/价目以方舟控制台为唯一真值源。

## 1. watermark 显式 false（防"AI 生成"水印）

- **方舟默认 `watermark=true`**，出图右下角带"AI 生成"水印；科研交付图不能带。
- 护栏：请求恒**显式** `watermark=false`（`guardrails.apply_watermark`）；
  确需可溯源草稿水印时显式 `allow_watermark=True`。
- 硅基流动 schema 无 watermark 字段（不加水印），但统一写 false 无害。

## 2. seed 兼容档（谁能复现，谁不能）

| 通道 | 合法范围 | 生效性 | 科研口径 |
|---|---|---|---|
| 方舟 | [0, 65535] | **仅 3.0-t2i 生效**；官方声明"类似结果不保证一致" | **不可作复现依据**，只记账 |
| 硅基流动 | [0, 9999999999] | **全档生效** | 可复现（固定 seed 重跑） |

- 护栏：`guardrails.check_seed(seed, provider, model=...)`——越界**报错不静默截断**；
- 账本一律记 seed（有就记）；方舟档注记"非确定"，复现需求优先走硅基通道。

## 3. b64 规避 URL 失效（产出必须立即落盘）

| 通道 | 取图方式 | URL 时效 | 护栏 |
|---|---|---|---|
| 方舟 | `response_format=b64_json`（默认）\| url | url **24h 失效** | 默认 b64；用 url 则 `save_images` 立即下载 |
| 硅基流动 | 仅 `images[].url`（schema 无 response_format） | **1h 失效** | 收到响应**立即下载**；传 b64 参数直接报错 |

- `GeneratedImage.to_bytes()` / `providers.save_images()` 落盘后缀按**真实字节**判定
  （Pillow 探测，不信 output_format 假设）；
- 请求快照里的 data URI 会截断（防数 MB 参考图进账本）；`runs` 快照剥离 b64。

## 4. 成本护栏（张数上限 + 组图硬顶）

- 单次请求张数 clamp：默认 **4** 张（`IMAGEGEN_MAX_IMAGES` 环境变量覆盖），
  `guardrails.clamp_count/check_images_budget`；
- 方舟组图硬顶：**参考图数 + 产出数 ≤ 15**（超出报错）；
- 成本心法：一次 `n=4` 选 1 张 < 单张重掷 4 次；失败张走 `data[i].error`
  （仅成功张计费），别整请求重试；
- 价格锚点（2026-11 调研，实施前控制台核价）：5.0-flash 分销 ~0.12 元/张；
  5.5 第三方 1K ~$0.018；硅基模型广场实时价，Kolors/schnell 历史有免费档。

## 5. 契约硬约束速查（build_payload 已拦）

| 约束 | 方舟 | 硅基流动 |
|---|---|---|
| 文生图 | 全档 | 全档 |
| 图生图参考图 | 5.0-pro/flash ≤10，其余 ≤14 | 单图：`image`（Qwen-Image-Edit/Kontext-dev）或 `input_image`（Kontext-max/pro） |
| 组图 | `sequential_image_generation=auto`，4.0/4.5/5.0 主档 | 无组图；`batch_size` ≤4 |
| 尺寸 | `size` 1K/2K/3K/4K 或 WxH | `image_size` "宽x高"（模型各有推荐档） |
| 格式 | `output_format` png\|jpeg（png 仅 5.0 主档） | `output_format` png\|jpeg（Kontext） |
| 透明底 | 5.5 `background=transparent`+png；5.0 仅 RGBA 图生图保透明 | 无（rembg 兜底） |
| seed | [0,65535]，仅 3.0-t2i | [0,9999999999] 全档 |
| 单张失败 | `data[i].error`，其余张照常 | 同（响应归一到 `failed_images`） |
| 联网增强 | `tools=[{"type":"web_search"}]`（5.0） | — |

- 模型 ID **字符串直传**（控制台为真值源）；档位能力表会滞后，
  `extra={...}` 可透传任意上游参数绕过本地检查（pro/flash 判定按 ID 含 `pro`/`flash`）。

## 6. 401/403 优雅报错（key 过期是常态）

- `providers.ProviderError`：带 `status_code/code/body` + `hint()` 可执行提示
  （去哪换 key、模型未开通、额度/限流、服务过载）；
- key 未配置 → 提前 401 口径报错，**不发请求**；
- 本项目对 401 的口径：**优雅报错不裸崩**——冒烟脚本、批量出图遇 401 停下换 key，
  不重试烧钱。
