---
name: comsol-simulation
description: ②COMSOL 仿真域操作面：mph/COMSOL-MCP 双入口工作流——建模→求解→后处理→出版级出图（尺寸/DPI/背景/品牌剥离 logo*d=off）→sidecar（data↔pixel 变换元数据）→验证（守恒/收敛/domain-check）；单证会话约定与 Java/mph 坑点速查。需要建/改 COMSOL 模型、跑求解/参数扫掠、出出版级结果图、产 sidecar 供③域叠加、做物理验证或查 COMSOL API 坑点时使用。
---

# COMSOL Simulation（②仿真）

COMSOL 仿真能力域的操作面：**仿真本体是 COMSOL 6.4**（外部商业软件，项目内自包含安装
`sub_root/deps_management/deps/comsol/6.4/base`，注册表 COMSOLROOT 自发现）；
Python 通道是 **mph==1.4.0 + JPype1==1.7.1**（豁免 pin，官方无 Python API 的唯一现实
通道），agent 通用驱动是 **COMSOL-MCP**（可选层，pin commit，另一代理接入 DSH）。
选型与用户裁决见 `sub_root/others/community_research/2026-11-17/domain-02-comsol-selection.md`；
本技能只放"怎么用"。薄胶水在 `sub_root/src/pysci/skills/comsol_simulation/`。

## 能力分层（谁负责什么）

1. **COMSOL-MCP**（agent 层，可选）：通用驱动 101 工具——建模/参数/几何/物理/网格/
   求解(async+progress)/结果导出；**手册知识走其 `pdf_search` RAG**（手册 PDF 喂入
   `sub_root/data/skills/comsol_simulation/manual/`，由接入代理维护）。空闲态不占证，
   显式 `comsol_start` 才占。
2. **mph**（Python 通道）：脚本化驱动的底座——`client.load/create`、`model.solve/export/
   evaluate/inner/outer`、`node.property()` 写属性、`node.java` 全量 Java API 逃生通道。
3. **薄胶水** `pysci.skills.comsol_simulation`（差异化保留，≈1100 行）：
   - `export` —— 出版级出图：像素级参数拼装（尺寸/DPI/背景/抗锯齿）、品牌剥离
     `logo*d=off`、clean、空白自检；
   - `sidecar` —— 栅格 sidecar 生产端（data↔pixel 变换元数据 JSON）+ 消费端
     （matplotlib/ultraplot 数据坐标叠加，衔接③域 truthful colorbar）；
   - `features` —— feature-types 试建探测（COMSOL 无枚举 API）+ node set 提取/批量写属性；
   - `plotgroup` —— 求解后绘图组数据集自动重绑（空白 PNG 防复发）；
   - `validate` —— 验证函数族（能量守恒/收敛阶/参考比对，纯函数）；
   - `session` —— 会话/许可证探测 + domain-check 机械检查。

**能力边界（裁决注记）**：imgcheck 像素擦除**不建**（品牌剥离走 COMSOL 原生
`logo*d=off`）；pyvista 3D/场渲染归③域；手册知识检索走 COMSOL-MCP RAG（另一代理）；
export polar（参数化弧线扫角）远场特有、按需在研究树实现；④域 S 矩阵验证以
`validate.compare_to_reference` 同入同出比对形态衔接。**CMT 理论↔仿真 S 矩阵对账属
研究线专用**（2026-11-18 用户裁决：仅适用 gain_ep 研究）——代码与冻结判据在
`src/pysci/research/gain_ep/reconcile.py`，不在本技能内。

## 工作流（六步）

| 步 | 做什么 | 入口 |
|---|---|---|
| 1 | 建模（双入口） | COMSOL-MCP geometry/physics/mesh 工具 **或** mph 脚本（`model.create(path, 类型)` + `node.property()`；配方 → [references/modeling-recipes.md](references/modeling-recipes.md)） |
| 2 | 求解 | `model.solve('<study 名>')`（按**名**不按 tag）或 COMSOL-MCP `study_solve(_async)` |
| 3 | 后处理 + 重绑 | `model.evaluate(...)` / `model.export('Data')`；**solve 后必 `plotgroup.rebind_plot_groups(model)`** |
| 4 | 出版级出图 | `export.export_image(model, '<绘图组名>', png, width_mm=…, clean=True, geom_bbox=…)`；契约 → [references/export-contract.md](references/export-contract.md) |
| 5 | sidecar | 随第 4 步自动产出 `<png>.sidecar.json`；下游 `sidecar.imshow_sidecar(ax, sc)` 叠加 |
| 6 | 验证 | `validate.check_energy_conservation / check_convergence / compare_to_reference`；开工/交付 `session.check_declaration`（四问）→ [references/domain-declaration.md](references/domain-declaration.md) |

**交付前必读图**：`export_image` 的产物 PNG 用 read_image 目检（视觉策略 2.4）——
「能导出 ≠ 是对的」；空白自检只兜「只剩轴框」一类硬伤。

## 会话/许可证约定（单证机器，必读）

- **显式 start 才占证**：mph `session.start_client()` / COMSOL-MCP `comsol_start`；
  空闲态不占证，MCP server 可常驻。
- **进程退出才放证**：JPype JVM 启动后不可卸载，`client.disconnect()` 只清会话
  **不放许可证**——批任务按「进程级」编排，一次 Python 进程一个 Client 单例。
- **占证探测看 `jvm.dll`**（JVM 真启动）；`_jpype.pyd` 只是 import 过 jpype，是误报源。
  `session.probe_license_holders()` 列 jvm 进程；hardware 冒烟只拦 `comsol*` 进程。
- **本机会话模式用 stand-alone**：mph ≥1.3 默认 client–server，但 `comsolmphserver
  -port 0` 在本机起服务即炸（`Invalid port: 65536`）——`session.start_client()` 默认
  stand-alone（详见 [references/java-mph-pitfalls.md](references/java-mph-pitfalls.md)）。

## 机械断言

`sub_root/tests/skills/comsol_simulation/`：mock mph 纯函数单测 109 项**恒跑**
（sidecar 变换数学/出图参数拼装/domain-check 判定/feature 探测解析/验证函数）；
CMT 对账测试随代码移入 `tests/research/gain_ep/`（研究线不入库）；
真实 COMSOL 冒烟（建模→solve→出图→sidecar→domain-check）标 `-m hardware`
（`tests.cmd hardware`；跑前守卫：无 `comsol*` 进程、许可证可得，否则 skip 不红）。

## 不做 / 拒绝清单提醒

- **不做**（裁决）：imgcheck、pyvista 渲染（③域）、自写 JPype 层（=自研 mph）、
  FEniCSx（换范式）、LiveLink MATLAB（商业绑定）。
- 薄胶水只做差异化接缝；**通用建模/求解/导出优先走 COMSOL-MCP 工具或 mph 原生 API**，
  不要往胶水里堆通用功能。

## References（渐进加载，只读需要的那一册）

- [references/modeling-recipes.md](references/modeling-recipes.md) —
  建模配方：mph 双入口 API 速查、类型名/属性名真源表、声学最小模型配方、材料符号口径。
- [references/java-mph-pitfalls.md](references/java-mph-pitfalls.md) —
  Java/mph 坑点册：版本串短名、JVM 不卸载、disconnect 不放证、会话管理约定、
  静默空解、本地化 label、2D 轴限不暴露等实测坑。
- [references/domain-declaration.md](references/domain-declaration.md) —
  域几何声明模板：强制四问 + 激励口径工作点-EP 偏移子问 + 机械检查判定。
- [references/export-contract.md](references/export-contract.md) —
  出版级出图契约：尺寸/DPI/字体/背景/品牌剥离、Image 导出属性真源表、
  sidecar 契约与消费端用法（与③域导出契约风格一致）。
