# ⑦ AI 生图能力域 · 新需求三段式「选型」调研（2026-11-17）

> 状态：**待用户裁决**。工作 Agent 调研产物；输入 = `00-synthesis.md`（总调研 + §7 硬件约束）+
> `per-domain-flagships.md`（⑦初核）+ 旧 pySciWS ai_drawing 技能（仅对照能力，未复制代码）。
> 硬件约束（用户口述）：**仅 CPU、无独显/显存、可用内存 ~5GB → 本地扩散推理不可行，云 API 优先**。
> 判定口径：① star ≥ 100；② 2026-05~11 逐月提交；③ 引入可减少自研 + 无可替代性分级豁免。
> 域边界：⑦管**生成式图像**（示意图/概念图/艺术配图/封面/graphical abstract）；数据图归③域。

---

## 0. 结论速览

**建议组合：云双通道 + 本地轻后处理，扩散引擎不留本地**——

| 角色 | 选型 | 判定 |
|---|---|---|
| 云主通道 | **火山方舟 Seedream**（doubao-seedream-5.x/4.x，OpenAI 兼容 images API） | key 已有、能力最全（图层拆分/交互编辑/组图/4K/5.5 原生透明底），薄胶水重建 |
| 云副通道 | **硅基流动**（Qwen-Image/FLUX.2/FLUX.1-Kontext/Z-Image/MOSS-Image…） | key 已有、seed 可复现、有免费/低价档、编辑模型族全 |
| 本地后处理 | **Pillow + scikit-image + opencv-python-headless**（可选 extra）+ **rembg**（26k★，抠图/透明底兜底） | 纯 CPU、成熟库、承接旧 postprocess/imaging 能力 |
| 本地扩散兜底 | 默认**纯云**；diffusers+sd-turbo 可选 extra / FastSD CPU 二选一（待裁决） | CPU 分钟级/张、内存贴线，性价比低 |
| 拒绝 | ComfyUI（GPU 优先）、InvokeAI（Py3.13 不兼容）、A1111/Fooocus（停更）、全部现成生图 MCP（均差口径①②）、Qoder 内置 ImageGen（IDE 特有不可迁移） | — |

---

## 1. 任务一：旧 pySciWS AI 生图能力清单（仅对照）

旧域 = `pySciWS/src/pysci/skills/ai_drawing/`（CLI 门面 `imagine.py`）+ `data/skills/ai_drawing/` 资产区。
定位（旧 docstring）：科研**效果图**（封面/graphical abstract/示意图，美学为主）+ **视觉审美桥**（AI 范本 → 真实数据复现）。

| # | 能力项 | 旧实现（形态，不复制代码） | 保留判定 |
|---|---|---|---|
| 1 | 云端出图（文生图/图生图/组图） | `ark_client`：方舟 OpenAI 兼容 `POST /api/v3/images/generations`，requests 零新增依赖；i2i 多参考（5.0-pro≤10/其余≤14）；组图 sequential_image_generation（4.0/4.5/5.0 主档，参考+产出≤15） | **保留**（薄胶水重建，扩 SiliconFlow 副通道） |
| 2 | 图层拆分 + 交互编辑 | 5.0-pro/flash：1 底图 + ≤16 带 alpha 图层；`mark` 画框/箭头/标签 → `edit` 局部编辑 | **保留**（科研示意图改版刚需） |
| 3 | 出图护栏/契约 | watermark 显式 false（方舟默认加水印）；seed 仅 3.0-t2i 生效（官方声明非确定→账本不做复现依据）；b64_json 默认规避 URL 24h 失效；size 1K/2K/3K/4K 或 WxH；png 仅 5.0 主档；单图失败走 `data[i].error`；max_images 成本护栏 | **保留**（护栏知识进技能文档 + 薄胶水） |
| 4 | 本地位图后处理 | `postprocess`（Pillow）：crop/resize/rotate/convert/pad/加边/contact sheet/主色板抽取 | **保留**（Pillow 唯一核心新增依赖） |
| 5 | 进阶位图算子 | `imaging`（scikit-image+OpenCV optional extra）：alpha split/composite、泊松融合、算法 inpaint（Telea/NS）、mask（otsu/canny/grabcut）、形态学、透视拉正、连通域测量、相位配准 | **保留但收缩**：split/composite/mask/warp 留⑦；measure/align 与③域重叠，建议移交③ |
| 6 | 生成账本/溯源 | `ledger`：manifest.jsonl + LEDGER.md 双写、PNG tEXt 内嵌 `pysci:*` 溯源（与 ComfyUI/A1111 tEXt 约定共存）、runs/*.json 请求快照、成本统计 | **保留**（云计费场景溯源+审计必要） |
| 7 | 审美范本→数据复现桥 | `bridge`：palette 抽取 + scaffold_figure（脚手架③域管线）+ design_spec.md + 配色注册③域调色板 | **保留**（⑦→③衔接的差异化能力） |
| 8 | 统一 CLI 门面 | `imagine`：doctor/models/gen/i2i/edit/layers/mark/adjust/sheet/ingest/gallery/ledger/palette/bridge | **形态保留**（技能 CLI + SKILL.md，重建） |
| 9 | 技能资产区 | assets/gallery/prompts/recipes/runs + LEDGER（审美画廊、prompt 配方、请求快照） | **保留**（目录约定重建） |
| 10 | Tier 0：Qoder 内置 ImageGen | IDE 内置生图兜底、产物 `ingest` 入库 | **不保留**（DSH 无对应物，环境特有） |
| 11 | ComfyUI 客户端 | 已删除，仅 `__pycache__` 残留（comfy_client/comfy_session/providers/workflows）；旧设计明确否决 ComfyUI（无 GPU + 第三方节点收窄方舟能力面） | **不保留**（路线已否决，维持） |

注：devops 域 `keyscan`（SiliconFlow/方舟 key 脱敏扫描）属密钥治理，不归⑦域。

---

## 2. 任务二a：云 API 通道调研

### 2.1 API 对比表（2026-11-17 采样）

| 维度 | 火山方舟 ARK（主通道） | 硅基流动 SiliconFlow（副通道） |
|---|---|---|
| 端点 | `POST https://ark.cn-beijing.volces.com/api/v3/images/generations`（OpenAI 兼容） | `POST https://api.siliconflow.cn/v1/images/generations`（OpenAI 兼容 `images.generate`） |
| key | `ARK_API_KEY`（旧 .env 已有，另有 `ARK_API_KEY_2`） | `SILICONFLOW_API_KEY` + `SILICONFLOW_BASE_URL`（旧 .env 已有） |
| 模型清单 | doubao-seedream-**5-5**（11-09 发布，官方教程页已上线）、5-0-pro-260628、5-0-flash-260915、4-5-251128、4-0-250828、3-0-t2i-250415（控制台模型列表为唯一真值源） | Qwen/Qwen-Image、Qwen-Image-Edit、FLUX.2-pro/flex、FLUX.1-schnell/dev、FLUX.1-Kontext-dev/max/pro、FLUX-1.1-pro/Ultra、Tongyi-MAI/Z-Image-Turbo；模型广场另有 Kwai-Kolors/Kolors、**MOSS-Image**（11-11 上新，5 折至 11-25，主打精准文字渲染/复杂布局/4K token 提示词）、ERNIE-Image-Turbo |
| 文生图 | ✅ 全档 | ✅ 全档 |
| 图生图/编辑 | ✅ 多参考（5.0-pro≤10、其余≤14）；5.0 交互编辑（带 mark 标记图局部改）；图层拆分（1 底图+≤16 alpha 层） | ✅ Qwen-Image-Edit、FLUX.1-Kontext 系列（input_image base64）、FLUX-1.1-pro image_prompt（Kontext 即"改图不崩"族） |
| 组图/连续生成 | ✅ sequential_image_generation=auto（4.0/4.5/5.0 主档，≤15 张） | ❌ 无组图参数；batch_size ≤4 |
| 分辨率 | **1K/2K/3K/4K** 或 width/height 自定义 | image_size 分档 512²~1664²（Qwen-Image 推荐 1328²~1664²）；FLUX-1.1-pro-Ultra 最大 2359296 px；aspect_ratio 21:9~9:21（Kontext） |
| **透明底** | 5.0 API `background:"transparent"` 仅支持"恰好 1 张 RGBA PNG 输入的图生图保透明"，文生图直出 400（11-12 实测）；**Seedream 5.5 原生直出透明 PNG**（background:transparent + output_format:png，玻璃类仍常不透光，需下载后验 alpha） | **无 background 参数**（OpenAPI schema 无）→ 无原生透明直出 |
| seed | 仅 3.0-t2i 生效，官方声明"类似结果不保证一致"→ 不可作复现依据 | ✅ seed 0..9999999999 全档生效（**科研可复现优势**） |
| 输出 | response_format url（**24h 失效**）\| b64_json；output_format png（仅 5.0 主档）\|jpeg；watermark **默认 true 须显式 false** | images[].url（**1 小时失效**）；output_format png/jpeg（Kontext） |
| 联网增强 | tools=[{"type":"web_search"}]（5.0） | — |
| 价格锚点 | 官方价目表 JS 渲染未逐项核验；锚点：5.0-flash 分销 **0.12 元/张**（七牛 2026-11）、5.5 第三方 1K **$0.018** 起 / 2K $0.0315–0.037（Kie/WaveSpeed/Segmind，11-12 核）；实施前控制台核对 | 按张计费，模型广场实时价（生图段 JS 未抓到）；Kolors/FLUX.1-schnell 历史有免费档；MOSS-Image 当前 5 折 → 实施前模型广场核对 |
| 封装薄胶水量 | 小：OpenAI 兼容 + 参数映射表（size 档位/组图/图层拆分/mark）；旧 ark_client 经验约数百行含护栏 | 小：标准 `images.generate` + extra_body（steps/cfg） |

**透明底补强路线**：①方舟 5.5 `background:transparent` 直出（首选，核验项：alpha 最高值应为 255，防 251–254"差一点不透明"）；
②方舟 5.0 图层拆分拿 alpha 层；③本地 **rembg** 抠图兜底；④旧 imaging mask/grabcut 手动抠。①②云端、③④ CPU 可行。

### 2.2 薄胶水形态

两通道同为 OpenAI 兼容 `images/generations` → **一个 requests 客户端 + 双 provider 参数映射 + 护栏/记账**（沿旧 ark_client 结构重建，
不复制代码）。.env key 从旧项目迁入工作区配置。产物落 `data/…/artwork/`，b64/立即下载规避 URL 时效。

---

## 3. 任务二b：本地轻量兜底

| 候选 | ★/活跃 | CPU 适配（5GB 可用内存） | 三口径 | 判定 |
|---|---|---|---|---|
| **diffusers**（huggingface/diffusers，Apache-2.0） | 34.9k★ ✅（旗舰核已验 3/3） | SD1.5/sd-turbo 可跑但**分钟级/张**、fp32 峰值 ~4-8GB 贴线；SDXL 级不可行；torch CPU wheel 重 | 3/3 ✅ | **可选兜底**（默认不装，`[local-gen]` extra） |
| **ComfyUI**（GPL-3.0，源码安装） | 147.2k★ ✅ | GPU 优先；`--cpu` 仅演示级（分钟级/图）；节点图本地优势在本机无从发挥 | 3/3 ✅ 但硬件不适配 | **拒绝**（纯云路线下无必要；将来有 GPU 再议） |
| **FastSD CPU**（rupeshs/fastsdcpu，LCM-LoRA+OpenVINO/TAESD） | 2.2k★，11 月有提交 | CPU 专用加速，README 称 512² ~10s 级（未实测）；GUI/CLI 形态，agent 可编程性弱于 diffusers API | ①✅②⚠️逐月待细核③✅ | **候选兜底**（若要本地，比裸 diffusers 合适） |
| **rembg**（MIT，ONNX U2Net 族） | 26k★，11 月有提交 | 抠图/透明底秒级、纯 CPU | ①✅②⚠️bursty 待细核③✅ | **建议引入**（可选 extra，补透明底缺口） |
| Pillow / scikit-image / opencv-python-headless | 成熟库 | ✅ 纯 CPU | ✅ | **引入**（位图后处理层） |
| InvokeAI / A1111 / Fooocus | — | — | — | **拒绝**（旗舰核结论：Py3.13 不兼容/停更） |

---

## 4. 任务二c：Agent 可编程性（HTTP 薄胶水 vs MCP）

**现成生图 MCP 检索（GitHub API，2026-11-17）**：

| MCP 候选 | ★ | 许可 | 活跃 | 三口径判定 |
|---|---|---|---|---|
| artokun/comfyui-mcp（178 工具/36 skills，ComfyUI 控制面） | 932 | MIT | 当天推送 | ①✅②✅③❌（绑本地 ComfyUI 实例，纯云无用）→ **拒绝**（形态参考） |
| shinpr/mcp-image（Gemini/GPT-Image/**BytePlus Seedream**，prompt 自动优化） | 192 | MIT | 11-13 推送 | ①✅②⚠️③⚠️ → **备选/借鉴**（唯一口径可信的云生图 MCP） |
| ATH-MaaS/Pixelle-MCP（ComfyUI+MCP+LLM） | 1,131 | MIT | 2025-12 后停 | ②❌ → **拒绝** |
| Seedream 专用 MCP：tengmmvp/Seedream_MCP 26★、joeseesun 19★、wearzdk/seedream-image-mcp 18★、mingforpc 8★、nsmao-com/seedream-mcp 3★（3 月后无提交） | <30 | MIT 等 | 多数停 | ①❌②❌ → **全拒** |
| PyPI-only：mcp-seedream-pro、volcengine-ark-mcp、SiliconFlow 图片生成 MCP（mcpworld 登记） | 无 GitHub | — | — | ①❌ → **拒绝** |

**结论：MCP 层本轮不引入**——云生图 MCP 全是小项目（差口径①②），唯一体面的 shinpr/mcp-image 面向通用网页配图，
不承载记账/溯源/③⑤衔接；而两通道同为 OpenAI 兼容 HTTP，agent 用 pwsh/curl/python 直接可调，
**HTTP 薄胶水 + 技能文档（SKILL.md）** 就是自研面最小解（与①域 paper-search-mcp"唯一聚合通道"不同：生图无聚合需求）。

---

## 5. 三口径 + 硬件适配汇总判定

| 候选 | ① star≥100 | ② 2026-05~11 逐月 | ③ 减自研 | 硬件 | 判定 |
|---|---|---|---|---|---|
| diffusers | ✅ 34.9k | ✅ | ✅ | ⚠️ CPU 慢/内存贴线 | **豁免硬件不适配，作可选兜底**（或不装，待裁决） |
| ComfyUI | ✅ 147.2k | ✅ | ✅ | ❌ GPU 优先 | **拒绝**（硬件不适配且纯云够用） |
| FastSD CPU | ✅ 2.2k | ⚠️ 11 月活跃，逐月待细核 | ✅ | ✅ CPU 专用 | **候选**（兜底方案二选一） |
| rembg | ✅ 26k | ⚠️ bursty 待细核 | ✅ | ✅ | **引入（可选 extra）** |
| Pillow/skimage/opencv-headless | ✅ 成熟库 | ✅ | ✅ | ✅ | **引入** |
| shinpr/mcp-image | ✅ 192 | ⚠️ | ⚠️ | ✅ | 备选/借鉴 |
| 其余生图 MCP | ❌ | ❌ | ❌ | — | 拒绝 |
| ARK/硅基流动（服务） | 服务非库 | — | ✅ OpenAI 兼容薄封装 | ✅ 云端 | **主/副通道** |

---

## 6. 科研场景需求映射

| 需求 | 方案 |
|---|---|
| 示意图/概念图生成 | ARK Seedream 5.5/5.0-pro（组图+参考图保风格）；文字渲染重的版式试 SiliconFlow MOSS-Image |
| 风格一致性 | ①组图 sequential_image_generation；②参考图 i2i（同风格锚）；③prompt 配方库（recipes）+ 审美画廊（gallery）固化风格词 |
| 透明底（图标/贴纸/独立元素） | 首选 Seedream 5.5 background:transparent（验 alpha）；备选 5.0 图层拆分；兜底 rembg |
| 批量 | batch_size（SF ≤4）/组图（ARK ≤15）+ 成本护栏 max_images + ledger 记账审计 |
| 与③域衔接 | bridge：审美范本 → palette 注册③域调色板 + scaffold_figure 脚手架 + design_spec.md（AI 图好看出骨架、真实数据出成品） |
| 与⑤域衔接 | 产物统一落研究线 `article/artwork/`（含 PNG tEXt 溯源），⑤文档/幻灯直接插图；contact sheet 供 agent 一眼全览 |
| 科研红线 | AI 图仅示意图/概念/美学用途，**不作数据证据**（旧 ledger 原则延续）；数据图走③域 |

---

## 7. 建议组合与待裁决点

**建议组合**
1. **云主通道 = 火山方舟 Seedream**（薄胶水重建：requests 客户端 + 护栏 + 记账 + 图层/交互编辑/mark）。
2. **云副通道 = 硅基流动**（同客户端双 provider 映射；seed 可复现 + 免费/低价档 + Kontext/Qwen-Edit 编辑族 + MOSS-Image 文字渲染）。
3. **本地 = 后处理而非推理**：Pillow + scikit-image/opencv（extra）+ rembg（透明底兜底）。
4. **拒绝**：ComfyUI、InvokeAI、A1111、Fooocus、全部现成生图 MCP、Qoder ImageGen。
5. 旧能力保留 1/2/3/4/6/7/8/9 项（§1 表），5 收缩、10/11 不保留。

**待用户裁决**
1. **本地扩散兜底取舍**：A. 纯云（推荐，维护面最小）｜B. diffusers+sd-turbo 可选 extra｜C. FastSD CPU（CPU 专用优化）。
2. **透明底路线**：等方舟 5.5 官方 API 参数核验后云直出（推荐）｜现在即配 rembg 兜底（可并行，建议保留）。
3. **MCP 形态**：建议不引入（shinpr/mcp-image 仅借鉴 prompt 优化思路）；如需 MCP 面请明示。
4. **imaging 收缩边界**：measure（连通域测量）/align（配准）是否移交③域。
5. **实施前核验项**：方舟 5.5 模型 ID + background 参数 + 官方价目（控制台）；硅基流动模型广场当日价与免费档；两 key 迁入工作区配置。
6. **⑦③边界确认**：生成式图（⑦）vs 数据图（③）分界按 §6 红线执行。

---

## 8. 来源 URL

- 总调研/硬件约束：本目录 `00-synthesis.md`、`per-domain-flagships.md`
- 旧项目能力对照：`D:\XXXIIIGGG\projects\pySci\pySciWS\src\pysci\skills\ai_drawing\`（docstring/CLI 契约）、`data\skills\ai_drawing\`
- 硅基流动 API：https://docs.siliconflow.com/en/api-reference/images/images-generations.md ｜ https://docs.siliconflow.com/cn/userguide/capabilities/images.md ｜ https://docs.siliconflow.com/llms.txt ｜ https://www.siliconflow.cn/pricing ｜ https://www.siliconflow.cn/ （MOSS-Image 5 折横幅）
- 火山方舟/Seedream：https://docs.volcengine.com/docs/ark/seedream-5-5?lang=zh ｜ https://docs.volcengine.com/docs/ark/image-generation-tutorial?lang=zh ｜ https://docs.byteplus.com/en/docs/modelark/seedream-5-0-pro ｜ https://docs.byteplus.com/en/docs/modelark/model-pricing
- 透明底实测（laozhang，2026-11-12）：https://blog.laozhang.ai/zh/posts/seedream-5-5-transparent-background
- 价格锚点：https://www.qiniu.com/ai/model/bytedance/doubao-seedream-5.0-flash （5.0-flash 0.12 元/张）｜ https://infron.ai/model-family/Seedream
- 本地兜底：https://github.com/huggingface/diffusers ｜ https://github.com/Comfy-Org/ComfyUI ｜ https://github.com/rupeshs/fastsdcpu （2.2k★，11 月提交）｜ https://github.com/danielgatis/rembg （26k★，11 月提交）
- MCP 检索：https://api.github.com/search/repositories?q=comfyui+mcp / ?q=seedream+mcp（2026-11-17）｜ https://github.com/artokun/comfyui-mcp （932★）｜ https://github.com/shinpr/mcp-image （192★）｜ https://github.com/ATH-MaaS/Pixelle-MCP （1,131★，2025-12 停）｜ https://github.com/tengmmvp/Seedream_MCP ｜ https://github.com/wearzdk/seedream-image-mcp ｜ https://github.com/nsmao-com/seedream-mcp ｜ https://pypi.org/project/mcp-seedream-pro/
- 星数/活跃核验：https://img.shields.io/github/stars|last-commit/*.json（2026-11-17 实测）

## 用户裁决记录（2026-11-17）
- **本地兜底=A 纯云**（不装本地扩散；Pillow/rembg 后处理本地无压力）；**透明底=并行保留 rembg**（云直出核验通过则云优先，rembg 作任意图抠图/兜底）。
- **measure/align 留⑦域 imaging 后处理**（不移交③——mpl 数据图用不上像素测量/配准；align 服务图层拆分/组图配准，measure 服务生成图/实验图量化）。
- **方舟 ARK_API_KEY 目前已过期**——实施仅做客户端/护栏/账本实现与静态测试，**真实出图冒烟留待 key 更新后补做**；硅基流动同暂不实测。
- ⑦③边界确认：⑦=生成式图像（示意/概念/艺术/图层），③=数据图。
