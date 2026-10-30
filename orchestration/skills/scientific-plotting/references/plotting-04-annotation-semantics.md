# plotting 分册 4/4 — 标注几何语义（ANNOTATIONS.yaml + gid 约定）

> 机械判据层：把任务书里的**可计数语义模型**（如「每侧 2 个框、外框大于内框、虚线相连」）
> 变成 `pysci-figures audit` 能核验的声明。适用范围：场图/示意图上的框、多边形、连接线。
> 通用实现，不含任何具体形状或物理尺寸；尺寸/边距/比例全部参数化声明。

## 1. 两步落地

1. **图目录写 `ANNOTATIONS.yaml`**（同名 `.json` 也可；与 `STYLE.yaml` 同级），声明语义模型；
2. **管线给标注原语打 `gid`**，audit 从**实际绘制的 Artist** 抽取几何（不是读声明里的数值）：
   - 原语：`art.set_gid("annot:<id>")`；可选元数据 `;role=<role>`、`;view=<view>`、
     `;kind=<box|polygon|line>`（语义是「框」但用 `Polygon` 绘制时写 `;kind=box`，
     计数按声明的语义 kind 归类）。
   - 视图：`ax.set_gid("view:<view>")`；未打时按轴序号 `ax0/ax1/…` 分组。
   - 支持：`Rectangle`、`Polygon`、`PathPatch`（闭合 = box/polygon）、`Line2D`（= line）。
     连接线请用 `ax.plot(...)` / `ax.add_line(...)` 绘制（`ConnectionPatch`/箭头补丁不抽取）。

   ```python
   field = Rectangle((-1, 0), 2, 1, fill=False, ls="-")
   field.set_gid("annot:L_field;role=field")
   ax.add_patch(field)

   mark = Rectangle((-0.2, 0.2), 0.4, 0.3, fill=False, ls="-")
   mark.set_gid("annot:L_mark;role=mark")
   ax.add_patch(mark)

   ax.plot([0.2, 0.5], [0.35, 0.5], ls="--").set_gid("annot:L_conn")
   ax.set_gid("view:L")
   ```

## 2. 声明 schema（全部可选；只声明你要核验的项）

```yaml
# 同一实体多视图的计数一致性：按 view 分组数原语
counts:
  - view: L
    kind: box            # box | polygon | line | any（默认 box）
    role: mark           # 可选：只数该 role
    expected: 2          # 任务书语义模型（如「每侧 2 个框」）

# 区域内标注：内原语完整位于外原语内 + 留边距
contain:
  - inner: L_mark
    outer: L_field
    margin_ratio: 0.02   # 边距 / 外框 bbox 短边；可选 margin_px 绝对像素（默认 0.02）

# 成对框相对大小（按闭合多边形面积；line 退化为 bbox 面积）
sizes:
  - outer: L_scan
    inner: L_mark
    min_ratio: 1.05      # 面积比须严格 > min_ratio（默认 1.0，相等即不通过）

# 连接线存在性：某条 line 的两端点分别落在 source/target 上或附近
connectors:
  - source: L_mark
    target: L_scan
    tolerance_ratio: 0.05   # 相对两元素中较小 bbox 短边；可选 tolerance_px

# 必须存在的原语 id（缺即报）
require: [L_field, L_mark, L_scan]
```

## 3. 判据与口径

- **坐标**：所有原语统一换算到**显示像素**后比较；混用 data/axes 坐标也能核验。
- **包含**：内原语所有顶点在外原语内、且内外边界无相交；凹多边形按边相交复核。
- **边距**：内边界到外边界的最短距离；贴边（距离 0）必然不通过默认 `margin_ratio`。
- **相对大小**：闭合原语用鞋带公式面积；不是 bbox 短边。大小倒置即报。
- **连接线**：只认 `kind=line` 原语的首/末顶点；端点须落在元素内或距边界 ≤ 容差。
- **计数**：按 `view × kind × role` 精确相等；多画一个框即 ERROR（报实际 id 清单）。

## 4. audit 接入与跳过

`pysci-figures audit <figdir>` 自动读取图目录 `ANNOTATIONS.yaml|json`：

- 声明存在 → 执行上述核验，违规以 `[ERROR] annot-*` 记入审计报告（FAIL）；
- 图中存在 `annot:*` 原语但无声明 → `WARN annotations`（提示补声明）；
- 无声明且无标注原语 → 静默跳过（不影响普通图）；
- 声明引用的 id 缺原语 → `ERROR annot-missing`；`counts` 声明视图缺原语 → 计数不符。

失败码：`annot-count` / `annot-missing` / `annot-contain` / `annot-size` / `annot-connect`。
审查量表侧对应 H6/D3（`pysci-figures audit` 为指定工具）。
