# Prompt 配方库（⑦域）

> 目标：科研生成图的 prompt **可复用、可迭代**。配方固化后存研究线
> `data/research/<n>_<name>/article/artwork/` 的 notes 或技能资产区，
> 满意的 prompt 与产物一起进账本（`ledger.record(prompt=...)`）。

## 一、科研示意图的 prompt 结构（四段式）

```
[主体] + [风格] + [构图/视角] + [负面/排除]
```

- **主体**：说清"画什么"——物理对象 + 关键特征，别堆术语（模型不是同行评审）。
  - ✅ `a lithium-ion battery cell cutaway showing anode, separator, cathode layers`
  - ❌ `electrochemical system`（太抽象，模型自由发挥）
- **风格**：科研插图常用风格词——`flat vector illustration / minimal schematic /
  technical diagram / isometric 3D render / clean line art / scientific illustration`。
  风格词决定"像论文插图"还是"像营销海报"，宁可具体。
- **构图/视角**：`centered composition, white background, generous margins, side view,
  cross-section, exploded view`。白底 + 居中 + 留边 = 后续裁剪/抠图友好。
- **负面/排除**：方舟/硅基部分模型认 negative prompt（硅基 `negative_prompt` 字段；
  方舟写进 prompt 尾部 `no text, no watermark, no labels`）。
  惯用排除：`text, letters, watermark, logo, captions, photorealistic background`。

**科研红线**：AI 生成图只作示意/概念/美学用途——prompt 里画的"比例、结构、数量"
不作数据证据；要真数据走③域。

## 二、场景配方

### 封面 / graphical abstract
```
a striking scientific illustration of <主题隐喻>, <风格>, bold shapes,
limited color palette (2-3 colors), high contrast, centered composition,
white background, no text
```
- 产出后 `postprocess.crop_resize` 到期刊封面尺寸；配色抽出来反哺③域
  （旧 bridge 能力，palette → ③域调色板）。

### 示意图（装置/流程/机理）
```
a minimal flat vector schematic of <对象>, <并列要素 A, B, C>, arrows showing <流程>,
labeled-free layout, white background, thin consistent line weight, no text
```
- **文字渲染是弱项**：图内尽量 no text，标签留给③域 matplotlib 叠加（可控、可改）。
  确需图内文字（海报/幻灯）→ 硅基 **MOSS-Image**（主打精准文字渲染/复杂布局）。

### 图标 / 贴纸 / 独立元素（透明底）
```
a simple flat icon of <对象>, isolated on transparent background,
clean edges, no shadow, full object visible with margin
```
- 走 `transparent=True`（方舟 5.5 云直出）；核验 alpha 见 transparent-background.md。

### 组图 / 风格一致性（多图同风格）
- **首选①组图**：方舟 `sequential_image_generation="auto"`（4.0/4.5/5.0 主档），
  一个 prompt 出 ≤15 张连贯分镜；
- **次选②参考图 i2i**：第一张定稿后作 `images=[...]` 参考（5.0-pro ≤10 / 其余 ≤14），
  prompt 描述"同风格的下一幕"；
- **兜底③配方固化**：同风格词 + 同 seed（硅基档）+ 同构图模板逐张出。

## 三、模型选择速查

| 需求 | 通道/模型 | 备注 |
|---|---|---|
| 通用示意/概念/封面 | 方舟 doubao-seedream-5-5 / 5-0-pro | 主力；5.5 另有透明底/OCR 强项 |
| 图层拆分/交互编辑 | 方舟 5.0-pro/flash | 1 底图 + ≤16 alpha 层；mark 画框/箭头/标签 |
| 组图分镜 | 方舟 4.0/4.5/5.0 主档 | sequential_image_generation |
| 文字渲染/复杂布局 | 硅基 MOSS-Image | 4K token 提示词 |
| 改图不崩（i2i 编辑） | 硅基 FLUX.1-Kontext 系 / Qwen-Image-Edit | input_image/image 字段 |
| 可复现出图 | 硅基任意档（seed 全档生效） | 方舟 seed 不可作复现依据 |
| 免费/低价试错 | 硅基 FLUX.1-schnell / Kolors 历史免费档 | 实施前模型广场核价 |

模型 ID 一律**字符串直传**（控制台/模型广场为唯一真值源），本地不维护注册表。

## 四、迭代心法

1. **一次只改一个变量**（风格/构图/主体分开调），prompt 版本写进账本 notes；
2. 出 4 张选 1 张（`n=4` + 成本护栏），别单张反复重掷烧钱；
3. 好结果**立即归档**（生成非确定，重跑拿不回同一张）；
4. prompt + 产物 + 参数一起走 `ledger.record` + `embed_metadata`——图拷走了也能溯源。
