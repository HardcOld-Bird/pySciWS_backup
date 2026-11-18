# 透明底策略（⑦域）

> 需求场景：图标/贴纸/独立元素（叠进③域图、⑤域幻灯）、图层素材（合成海报）。
> 裁决（2026-11-17）：**云直出核验通过则云优先，rembg 并行保留作任意图抠图/兜底**。
> 薄胶水实现：`providers.build_payload(transparent=True)` + `postprocess.remove_background`
> + `postprocess.verify_alpha`。

## 路线优先级（四选一，按序降级）

### ① 方舟 Seedream 5.5 云直出（首选）

```python
resp = providers.generate("ark", prompt, model="doubao-seedream-5-5", transparent=True)
# → payload: background="transparent" + output_format="png"
```

- **必须验 alpha**（`postprocess.verify_alpha`）：`alpha_max == 255` 且有透明像素。
  常见坑："假透明"产物 alpha 最高值 251–254，叠深色底露白边；
- 已知边界（2026-11 实测记录）：玻璃/半透明类材质常不透光，需下载后目检 + 验 alpha；
- 5.0 API `background:"transparent"` **仅**"恰好 1 张 RGBA PNG 输入的图生图保透明"，
  文生图直出 400（11-12 实测）——别在 5.0 文生图上试。

### ② 方舟 5.0-pro 图层拆分拿 alpha 层

- 1 底图 + ≤16 带 alpha 的 PNG 图层（编辑/拆分能力，走 `extra` 透传 + mark 标记）；
- 拆出的单层即透明素材；后续 `imaging` 可做层间配准（align）。

### ③ rembg 本地抠图兜底（任意图）

```python
postprocess.remove_background("in.png", "out.png", model="u2net")   # 或 isnet-general-use
report = postprocess.verify_alpha("out.png")
```

- 适用：副通道（硅基无透明参数）出图抠底、实拍图抠主体、云直出不合格时兜底；
- 纯 CPU 秒级；模型权重 vendor 在 `deps_management/deps/cache/rembg/models/`
  （`U2NET_HOME` 由 postprocess 在 import rembg 前自动设好，用户环境不落盘）；
- 边缘毛糙 → `alpha_matting=True`（慢）或换 isnet 模型。

### ④ 手动 mask（最后手段）

- 旧 imaging 的 mask/grabcut 手抠工具已收缩（未重建）；确实需要时用 rembg 输出
  的 alpha 当 mask 起点，或请③域画掩膜。

## alpha 合规验证口径（verify_alpha）

| 判定 | 条件 | 处置 |
|---|---|---|
| OK | RGBA + `alpha_max==255` + 有透明像素 | 交付 |
| 假透明 | `alpha_max` ∈ [1,254] | 重抠/重生成；或 `alpha.point(v -> 255 if v>127 else 0)` 硬化（损失边缘） |
| 全不透明 | 透明像素 = 0 | 抠图失败或本就不透明，重跑 rembg/换模型 |
| 无 alpha | RGB/L 等 | 不是透明底产物；重走 ①–③ |

## 合成注意事项

- 透明 PNG 叠到深色底验白边、叠浅色底验黑边（各看一眼）；
- JPEG 无 alpha——透明产物一律 PNG 容器（`crop_resize` 保存 JPEG 时会自动转 RGB 丢 alpha，
  别用 JPEG 存透明素材）；
- 与③域合成时先 `imaging.align` 对齐再叠（亚像素配准，见 imaging.md）。
