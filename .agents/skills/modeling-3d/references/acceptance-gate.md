# 验收门红线说明（⑥域）

**验收门是唯一交付路径**：打印件 STL 不过门不得切片，渲染件不过门不得入图。
"看着行就交"没有例外——门失败的正确处置永远是**回脚本改几何/参数重跑**，
不是修网格、不是降低门槛、不是跳过。

## 判据表（打印件与渲染件分开判据）

| 判据 | 打印件（`judge_print`） | 渲染件（`judge_render`） | 度量来源 |
|---|---|---|---|
| 尺寸 | **必判**：`expect_bbox_mm` 排序逐维 ±0.1mm；体积 ±1%（可选） | 仅 sanity 警告（±0.5mm，不判负） | `stl_metrics` |
| 流形/闭合 | **必判（红线）**：`watertight` 每条无向边恰 2 面共享 | 不判（开面合法） | `stl_metrics` |
| 法线一致 | **必判（红线）**：`winding_consistent` 每条有向边恰 1 次 | 不判（渲染可双面） | `stl_metrics` |
| 面数预算 | 判：默认 40 万（切片负担） | **必判**：默认 20 万（Cycles CPU 性能红线） | `stl_metrics` / 标记行申报 |
| 退化面片 | **判（红线）**：零面积/重复顶点三角 | 仅警告（黑斑/漏光瑕疵） | `stl_metrics` |
| 体积 | 判：必须为正（负=内外翻转） | 不判 | `stl_metrics` |
| 度量缺失 | **判失败**（红线不许空过） | 不判（渲染件走标记行申报） | — |

判定函数（`pysci.skills.modeling_3d.acceptance`）：

```python
from pysci.skills.modeling_3d import stl_metrics, judge_print, judge_render

report = judge_print(stl_metrics("part.stl"), expect_bbox_mm=(40.0, 20.0, 8.0))
if not report.ok:
    print(report.format())   # FAIL 逐项列出；退出非零

# 渲染件（GLB/OBJ）：面数由导出脚本标记行申报
render_report = judge_render(metrics_from_markers(parse_markers(stdout)))
```

为什么分开判：打印件的失败是**物理失败**（切片废件、装配不上、内外反），渲染件的
失败是**性能/观感失败**（卡死、黑斑）。把渲染件按打印判据判会误杀合法开面；把打印件
按渲染判据判会放过非流形废件。这就是"打印渲染不混"红线的判据层含义
（全口径见 [print-vs-render.md](print-vs-render.md)）。

## 门语义（红线的三条推论）

1. **FAIL = 回脚本**：非流形→布尔残留/自交，改 bpy 脚本重导出；尺寸不符→参数/单位
   错，改参数区并复核几何断言区。**禁止**手改网格蒙混（对账会抓到，见 ledger 红线）。
2. **容差语义**：bbox 排序后逐维比较（STL 无方向语义，不假定轴向对应）；体积 rtol 1%
   吸收三角化误差；渲染件 bbox 容差放宽到 0.5mm 且只警告。
3. **度量缺失不算过**：打印判据里 `watertight=None`（未实测）按失败论。门的默认方向
   是"拒"，不是"放"。

## triage 速查

| 症状 | 原因 | 处置 |
|---|---|---|
| watertight=False | 布尔残留、自交、未合并顶点 | 脚本里布尔后清理/重算；重导出再过门 |
| winding=False | 建模时翻面、镜像未修正 | `bpy.ops.object.mode_set` 后 recalc normals（`bpy.ops.mesh.normals_make_consistent(inside=False)`），重导出 |
| 体积 ≤ 0 | 整体法线内翻 | 同上 recalc；或镜像变换后未翻转 |
| bbox 超差 | 单位错（mm/BU 混）、参数错 | 核对 `scale_length=0.001` 与参数区；复核断言区 |
| 面数超预算 | 细分过头/布尔产生碎片 | 降细分、decimate（打印件慎用）、重建低模 |
| 退化面片 | 共面焊接/布尔残片 | 脚本里 merge by distance / 重建该处几何 |

## 与"交付冻结"红线的关系

过门 ≠ 可交付；交付 = 过门 + `ledger.freeze` 冻结（产物↔脚本指纹对账）。冻结后产物
再被手改 → `reconcile` 报 `drift`，交付作废回脚本。两道红线串起来才闭环：
**门保证"这一次对"，冻结保证"每一次都可复现地对"**。
