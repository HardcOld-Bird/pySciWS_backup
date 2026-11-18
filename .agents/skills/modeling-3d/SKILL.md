---
name: modeling-3d
description: ⑥3D 建模能力域操作面：bpy 无头全链（脚本即模型→参数化建模→Cycles CPU 渲染→GLB/STL/OBJ 导出→验收门红线→交付物冻结回脚本）工作流；打印件与渲染件判据不混；Tripo 云生成只做原型/示意/参考，精密件一律 bpy。需要参数化建模打印件/实验装置、出概念渲染图、导出 GLB/STL/OBJ、验网格（尺寸/流形/法线/面数）、冻结交付物防手改丢失复现、或做 Blender/Cycles/内存自检时使用。
---

# 3D 建模（⑥ · modeling-3d）

⑥域操作面：**bpy 无头全链**是唯一主力。选型与用户裁决（2026-11-17，**B 纯 Blender**）见
`sub_root/others/community_research/2026-11-17/domain-06-3d-selection.md` 末尾裁决记录；
实施计划见 `sub_root/others/plans/domains-06-07-3d-imagegen.md`。本技能只收"怎么用"。

## 能力边界注记（裁决即边界）

- **STEP 不做**：纯 Blender 裁决下无 B-rep 通道；CAD→STEP→COMSOL（②域）需求出现时
  **再立项**（届时评估 build123d），勿在本域临时补 CAD。
- **渲染边界**：概念渲染管线（Blender 无头→PNG）**留⑥**；③科研绘图域只收 PNG 媒体做
  排版（外部媒体面板），出图尺寸/DPI/背景契约归③。pyvista 快速几何预览归③。
- **Tripo 云生成**：任意尺度三角网格（GLB），只做**原型/示意/参考**；精密件（打印/仿真/
  有尺寸要求）一律 bpy 参数化脚本。详见 [references/tripo-cloud.md](references/tripo-cloud.md)。

## 四条红线

1. **验收门红线**：交付物必须过验收门——打印件判尺寸/流形/法线/面数预算（严），渲染件判
   面数预算（宽，开面合法）。**门是唯一交付路径**，无"看着行就交"。
   判据细则：[references/acceptance-gate.md](references/acceptance-gate.md)。
2. **打印件与渲染件不混**：两套判据、两种失败语义；云网格/渲染件**不直接打印**，
   打印件不为渲染妥协几何。详见 [references/print-vs-render.md](references/print-vs-render.md)。
3. **交付物冻结回脚本**：交付 = 版本化 bpy 脚本重跑的产物 + 冻结账本（产物↔脚本指纹
   对账）。手改模型/手改渲染图 = 复现链断裂，对账即暴露；blender-mcp 交互探索的成果
   必须先冻结回脚本。
4. **渲染纪律（内存红线）**：Cycles CPU、单帧低采样、**禁并行多实例**（~5GB 预算）；
   EEVEE 需 GPU 本机不可用。详见 [references/cycles-cpu.md](references/cycles-cpu.md)。

## 工作流（建模→渲染→导出→验收→冻结）

| 步 | 做什么 | 入口 |
|---|---|---|
| 1 | 自检 | `doctor()`：Blender 路径/版本、Cycles CPU 口径、内存预算、单实例 |
| 2 | 脚手架 | `scaffold.ensure_model_session(name, slug)` 建会话目录；`render_scaffold("part"\|"scene", ...)` 生成 bpy 脚本 |
| 3 | 建模 | 改脚本参数区（mm，逐项注来源）→ 建模区改实际几何 → §3 几何断言跟着复核（脚本即模型） |
| 4 | 渲染 | `scene` 脚本：Cycles CPU 低采样单帧出 PNG（VLM 目检后再交付，视觉策略 2.4） |
| 5 | 导出 | `part` 脚本导出 STL/OBJ/GLB，打 `[model3d]` 标记行（协议勿改格式） |
| 6 | 验收门 | 打印件 `judge_print(stl_metrics(stl), expect_bbox_mm=...)`；渲染件 `judge_render(...)`；FAIL 即回第 3 步改脚本 |
| 7 | 冻结 | `ledger.freeze(script, artifacts, out)` → 交付；日后 `ledger.reconcile(record, base_dir=...)` 对账 |

运行形态：`blender -b -P <脚本>.py`（Blender 自带 Python，脚本自包含、不 import pysci）；
薄胶水 `pysci.skills.modeling_3d` 在项目 venv 侧做生成/判定/对账/自检。命令速查：

```powershell
uv run python -c "from pysci.skills.modeling_3d import doctor; print(doctor().format())"
& <blender.exe> -b -P data\research\<n>_<name>\models\<slug>\scripts\<slug>.py
```

## Tripo 云生成（口径摘要）

图/文→GLB 走官方 `tripo3d` SDK（v3；v2 已关停勿用）。产物**只做原型/示意/参考**
（进③概念图、当重拓扑视觉参照、教学演示）；打印/仿真件拿 GLB 当参考**回 bpy 重建成
精确几何**。真实出图冒烟挂起（key 待核），接口与口径见
[references/tripo-cloud.md](references/tripo-cloud.md)。

## 薄胶水 API（`pysci.skills.modeling_3d`）

| 模块 | 职责 |
|---|---|
| `scaffold.py` | bpy 脚手架模板生成（part 四段式 / scene 影棚骨架）+ 会话目录 |
| `acceptance.py` | 验收门：`stl_metrics`（纯 Python STL 度量）+ `judge_print`/`judge_render` 分开判据 |
| `ledger.py` | 交付冻结：`freeze`/`reconcile` 指纹对账 + `[model3d]` 标记行解析 |
| `doctor.py` | `find_blender`/`ensure_single_instance`/`available_memory_gb`/`doctor` 自检 |

## 何时坏掉

先 `doctor().format()`：**Blender NOT FOUND** → 等 vendor 落盘
`deps_management/deps/blender/<版本>/` 或设 `BLENDER_EXE`（环境契约：薄胶水集中读取，
blender-mcp 走 `BLENDER_HOST`/`BLENDER_PORT`=9876）；**已有实例** → 收尾其一再渲
（内存禁并行）；**验收门非流形** → 在 bpy 脚本里修几何重导出，勿手改网格；
**对账 drift** → 回脚本重跑重新冻结，勿交付手改件。bpy 算子改名坑/引擎 ID 坑：
[references/bpy-recipes.md](references/bpy-recipes.md)。**起 MCP/子进程要自定义 env 时
显式传 env**（mcp stdio_client 不传会剥掉自定义变量）。

## Reference files

- [references/bpy-recipes.md](references/bpy-recipes.md) — bpy 无头配方与坑点（四段式脚本形态、单位、导入/导出算子改名、标记行协议）。
- [references/cycles-cpu.md](references/cycles-cpu.md) — Cycles CPU 须知（EEVEE 不可用、低采样、内存 ~5GB 纪律）。
- [references/acceptance-gate.md](references/acceptance-gate.md) — 验收门红线说明（判据表、triage、门语义）。
- [references/print-vs-render.md](references/print-vs-render.md) — 打印件与渲染件不混原则。
- [references/tripo-cloud.md](references/tripo-cloud.md) — Tripo 云生成口径（原型/示意/参考）。
