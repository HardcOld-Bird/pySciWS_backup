# ⑥ 3D 建模能力域 · 新需求三段式"选型"调研

> 状态：待用户审核。2026-11-17 由工作 Agent 采样核验（star 用 shields.io、逐月提交用 GitHub search API
> `committer-date` 逐月计数、PyPI 用 pypi.org JSON）。判定口径（用户已裁决）：① star ≥ 100；
> ② 2026-05 ~ 2026-11 逐月均有提交；③ 引入可减少自研代码量。差②按**无可替代性分级豁免**：
> 无可替代→豁免 pin + 原地 ADR；有替代→拒绝。
> **硬件约束**：仅 CPU、无独显、16GB 内存（可用约 5GB）——本地重型推理谨慎，云 API 优先。
> ③域裁决"pyvista 渲染归③"已定；②域"CAD→STEP→COMSOL 导入"衔接需求在本域评估。

## 0. 结论摘要

1. **无一站式 3D 套件**（与 C 线一致）；走"社区工具组合 + 薄胶水"。
2. **首选组合**：**build123d**（参数化 CAD，代码即模型）+ **trimesh**（交付验收门）+ **Blender
   无头渲染**（本机已装，外部安装件）+ **blender-mcp**（35k★，交互探索 + 云生成入口；差②
   豁免 pin）+ **Tripo 官方 SDK `tripo3d`**（云 3D 生成，v3 API，key 已在旧项目 .env 待迁）。
3. **次选/可选豁免 pin**：gmsh（网格剖分，按需；star 不可核验 + 发版驱动，按 paper-qa 先例 pin
   可选件 + ADR）；FreeCAD（GUI 后备/STEP 人工编辑，外部工具不进依赖）。
4. **拒绝**：CadQuery（与 build123d 二选一，败于内核/依赖面）、OpenSCAD（有替代）、meshio
   （0/7，维持③域判定）、tripo-mcp 社区封装（4★停更）。
5. **重要旧档发现**：旧 pySciWS 该域已实装 build123d + trimesh + Blender 无头 + Tripo 的双管线，
   且当年做过 build123d vs CadQuery 对决——本轮依赖树核验证实其取舍依据**仍然成立且更清晰**
   （build123d→`cadquery-ocp-novtk` OCCT 8.0 无 VTK；CadQuery 2.8→`cadquery-ocp` 7.9 +
   trame-vtk/VTK 全家桶，与③域 pyvista 的 VTK 依赖面冲突）。另：**旧 Tripo v2 客户端已随 v2 API
   2026-10-31 关停而失效**，云生成通道必须重写为 v3（官方 SDK）。

## 1. 旧能力清单（仅对照，不复制代码）

来源：`pySciWS/orchestration/skills/modeling3d/`（SKILL.md + cli.md + mcp.md）、
`src/pysci/skills/modeling3d/`（tools 5 模块 ≈57KB + blender_scripts/render_studio.py ≈10KB）、
`data/skills/modeling3d/templates/`（print_part/scene 两模板）。**无 tests**（tests/skills 无该域）；
recipes/cache 为空。

| # | 能力项 | 旧实现（形态） | 概要 | 保留判定 |
|---|---|---|---|---|
| 1 | 参数化 CAD 打印件建模 | `print_part.py.tmpl`（build123d 脚本模板） | mm 精确、OCCT B-rep、布尔/孔/圆角/阵列；脚本四段式（参数区含来源注记/建模区/几何断言区/导出区） | **保留（社区顶位）**：build123d 直接覆盖；模板内容作技能文档资产 |
| 2 | 几何断言（物理正确性守门） | 模板 §3（bbox/尺寸 assert） | 参数一改即复核，防"尺寸不符"静默出图 | **保内容不保代码**：并入模板/技能文档（零代码） |
| 3 | STL/STEP 双导出 + 标记行协议 | `model3d build` 解析 `[model3d] stl:`/`volume_mm3:` | 脚本↔CLI 的接口契约 | **保形态薄化**：协议保留于薄 CLI（胶水接口） |
| 4 | 网格验收门 | `check.py`（trimesh 5.7KB） | watertight/体积为正/体积断言（rtol 1%）/排序 bbox 断言（0.1mm）/Euler 数/连通体数；非闭合=不可切片 | **保留（trimesh 顶位 + 薄胶水 ≈100 行判据）** |
| 5 | Blender 无头渲染 | `render_studio.py` 10KB + `blender.py` 3.8KB | studio 灯光/材质/相机/引擎回退链（EEVEE→Cycles）；`blender -b -P` 驱动 | **保留（形态）**：Blender 已装机；**渲染归属待裁决**（见 §7-4） |
| 6 | Tripo 云生成（图/文→3D） | `tripo_client.py` 9.4KB（v2 REST，requests 零新增依赖） | 图/文→GLB、PBR/quad/face_limit、balance 查询、dry-run、任务轮询下载 | **保留能力、接口重写**：v2 API 已于 2026-10-31 关停 → 官方 `tripo3d` SDK（v3） |
| 7 | blender-mcp 交互通道 | `mcp.md` 约定（"探索用，交付物冻结回脚本"） | 交互建模/预览/云生成（Tripo 同 key）；**绑定规则**：交付物必须来自版本化脚本 | **保留（社区工具 + 文档约定）** |
| 8 | 脚手架/会话管理 | `model3d new`（模板渲染 + 目录 + `assert_within_data` 防散落） | research 会话目录 stl/step/renders/notes | **保形态（薄胶水 ≈100 行）** |
| 9 | doctor 自检 | `cmd_doctor` + `config.py` 探测 | Blender exe/version/addon、库版本、Tripo key 有效性 | **保形态（薄胶水 ≈50 行）** |
| 10 | inventory 清单 | `cmd list` | 会话+脚本列表 | **不保留**（低价值，可由 agent 文件工具替代） |
| 11 | 配置面 | `.env`：MODELING3D_BLENDER / TRIPO_API_KEY | 路径守卫 assert_within_data | **保形态**（项目内配置自包含） |
| 12 | CAD→STEP→COMSOL 导入（②域衔接） | 旧②域 `build.py import_external_geomesh`（STL/PLY/VTK/STEP/IGES→Import 特征） | 为 Blender/CAD 铺的接入点 | **②侧已由 COMSOL-MCP `geometry_import` 顶位**（②域已收）；⑥侧只需产出 STEP + 单位/坐标约定 |
| 13 | 测试 | 无 | — | **新增最小 smoke**（验收门判据单测 + 脚手架冒烟），进 QA |

旧 pyproject 的选型实测记录（**只取结论不取代码**）：build123d 与 CadQuery 2.7 对决（同测试件体积/
网格/包围盒一致），弃 CadQuery 原因=内核旧（OCCT 7.8 vs 8.0）+ cadquery-ocp 捆绑 VTK 与 pyvista
冲突 + build123d 为同作者后继项目。本轮 §2 依赖树核验对上述三条的更新：内核代差仍在（7.9 vs 8.0）、
VTK 冲突被精确化（build123d 改用 **cadquery-ocp-novtk**，彻底无 VTK）、后继关系属实。

## 2. 候选指标表（2026-11-17 采样）

| 候选 | 定位 | ★ | 口径②逐月提交（2026-05..11） | Py3.13/Win wheel | agent 可编程性 | 许可证 | CPU/内存友好 | 判定 |
|---|---|---|---|---|---|---|---|---|
| **build123d** (gumyr/build123d) | 参数化 CAD（特征/布尔/阵列/钣金/3MF） | 3.7k ✅ | **7/7**（65/68/70/117/155/173/44）✅ | ✅ py3-none-any；0.13.0 @2026-09-21 | 纯 Python（builder/algebra 双模式），脚本即模型 | Apache-2.0 | ✅ OCCT 纯 CPU | **引入（⑥代码级首选）** |
| CadQuery | 参数化 CAD | 6.1k ✅ | **7/7**（6/8/8/4/7/6/4）✅ | ✅ py3-none-any；2.8.0 @2026-06-21（dep cadquery-ocp 7.9.x 有 cp313 wheel） | 纯 Python | Apache-2.0 | ⚠️ 依赖面重：trame-vtk/**VTK**（与 pyvista 冲突）、numba、casadi、nlopt | **次选备位（不与 build123d 并引入）** |
| FreeCAD | GUI CAD + 格式转换 | 36k ✅ | **7/7**（442/408/380/508/621/348/163）✅ | 官方安装包（自带 Python，与项目 venv 解耦） | ⚠️ FreeCADCmd 无头 + 自带宏/Python API；非 pip、解释器独立 | LGPL-2.1 | ✅（无头转换轻） | **引入（外部工具：GUI 后备/STEP 人工编辑）** |
| OpenSCAD | CSG 布尔建模（DSL） | 11k ✅ | **7/7**（4/41/56/61/50/54/46）✅（**修正总调研 §4"差②"标注**） | CLI 安装包 | ✅ CLI + .scad DSL（非 Python） | GPL-2.0 系 | ✅ 轻量 | **拒绝（有替代：build123d 全覆盖 CSG 场景）**；备选记录 |
| **trimesh** | 网格验收/度量/格式读写 | 3.7k ✅ | **7/7**（39/21/35/31/32/75/33）✅ | ✅ py3-none-any；5.2.1 @2026-11-12 | 纯 Python | MIT | ✅ 轻量 | **引入（验收门）** |
| gmsh | 几何+网格剖分（OCC 内置） | 镜像 135★ ⚠️（官方 gitlab.onelab.info 自托管，star 不可核验） | 镜像 **7/7**（32/29/81/132/693/416/309）✅（上游代理）；PyPI 发版驱动：4.15.2 @2026-03-24（8 个月未更）⚠️ | ✅ `py2.py3-none-win_amd64`（捆绑 SDK 二进制） | ✅ gmsh.py API + CLI | GPL-2.0-or-later | ✅ 纯 CPU | **可选件豁免 pin + ADR**（差①+发版驱动；无替代=独立剖分器中唯一 Python API 全家桶；主链不依赖，COMSOL 自划网格兜底） |
| Blender/bpy | 场景/渲染/云生成宿主 | 22k（GitHub 镜像）✅ | 镜像 **7/7**（790/1307/806/711/780/1522/647）✅ | ⚠️ 两条路：本机已装 Blender（无头 `-b -P`，旧档路线）；pip `bpy` 5.3.0 @2026-11-10 **有 cp313 win wheel 但 340MB** | ✅ bpy 脚本/无头 | GPL-2.0-or-later | ⚠️ Cycles CPU 可用；**EEVEE 需 GPU（无独显不可用）**；渲染吃内存 | **引入（外部 Blender 无头 + blender-mcp）**；bpy pip 作可选替代 |
| blender-mcp（ahujasid，现更名 mcp-for-blender） | Blender 交互 MCP + 云 3D 生成入口（Tripo/Meshy/Hyper3D/Hunyuan3D） | 35k ✅ | **6/7**：2026-05=0 ❌，6~11 月均有（core API 实测：06-11/07-29/08-30/09-30/10-29/11-15 + Atom 三周 20 提交）⚠️ **差②** | MCP server（Python）+ Blender addon | ✅ MCP 工具面 | MIT | ✅（本地只跑 MCP） | **豁免 pin + ADR**（差②仅缺 2026-05；agent↔Blender 交互**唯一**通道，无可替代；仅探索辅助、不进交付主链，风险面小） |
| **tripo3d**（Tripo 官方 Python SDK） | 云 3D 生成 v3 API 客户端 | 61★ ❌① | 发版连续（0.3.x→1.0.0，2025-03~2026-10，1.0.0 @2026-10-14）；⚠️ | ✅ py3-none-any；MIT；requires-python≥3.8 | ✅ 官方 SDK（异步任务/轮询/下载封装） | MIT | ✅（云端生成，本地零算力） | **豁免引入候选**（差①；"厂商唯一官方 v3 通道"，替代=自写 REST≈200 行反增自研面）→ **待裁决 7-3** |
| tripo-mcp（lxy2109） | Tripo 社区 MCP 封装 | 4 ❌ | 停更 2025-05 ❌（PyPI 0.1.2 @2025-04） | — | — | — | — | **拒绝**（三口径全差） |
| meshio | 网格格式互转 | 2.3k ✅ | **0/7**（全月 0 提交；末版 5.3.5 @2024-01）❌ | ✅ | ✅ | MIT | ✅ | **拒绝（维持③域判定；本域无不可替代位，见 §4.4）** |

## 3. 三口径/豁免判定汇总

- **三口径全过 → 引入**：build123d、trimesh、FreeCAD（外部工具）、Blender 无头（外部工具）。
- **豁免 pin + ADR**：blender-mcp（差②缺 2026-05；交互唯一通道）；gmsh（可选件，paper-qa 先例）；
  tripo3d 官方 SDK（差①，待裁决，推荐豁免）。
- **有替代 → 拒绝**：CadQuery（build123d 替代）、OpenSCAD（build123d 替代）、meshio（gmsh/trimesh/
  pyvista/COMSOL 原生导入联合替代）、tripo-mcp（官方 SDK/直连替代）。

## 4. 重点评估

### 4.1 科研 3D 场景四问

| 场景 | 覆盖方案 | 说明 |
|---|---|---|
| 参数化 CAD（特征/布尔/阵列） | build123d | Box/Cylinder/extrude/revolve/sweep/loft/fillet/chamfer/Hole、Locations 阵列、三布尔模式；脚本可版本化、可 diff、agent 直接改参数 |
| STEP/STL/IGES 导出（衔接②） | build123d 导 STL/STEP(+3MF)；FreeCADCmd 转 IGES 等长尾格式 | STEP 为②域主通道（B-rep 精确、带 mm 单位）；STL 仅供网格导入/切片；IGES 用得少→FreeCAD 按需转 |
| 网格剖分（衔接②） | gmsh（可选）；COMSOL 自划网格（默认） | ②域已裁"COMSOL 亦可自划网格"——gmsh 仅当需要外部高质量体网格/预剖分时启用 |
| 示意图/渲染图 | Blender 无头（studio 模板）→ PNG → ③域面板（③ #14 已文档化"外部媒体"）；快速几何预览走③域 pyvista | 渲染归属边界见 §7-4；版本化 scene 脚本 + 冻结规则照旧 |

### 4.2 AI 3D 生成（Tripo 云 API）的科研定位

- **能力面**（v3 API，`openapi.tripo3d.ai/v3`）：text/image/multiview→3D（模型 2026-11 更新为
  **H3.1** 旗舰款 ~40s 高保真+PBR、**P2** 原生四边面 ≤5 万面）、贴图
  （HD/8K）、quad/低模/分件、格式转换（GLTF/FBX/USDZ/OBJ/**STL**/3MF，输入 GLB/GLTF/FBX/OBJ/STL）、
  重拓扑、分割、骨骼动画；异步任务（create→poll→download）；**国内账号走 tripo3d.com 域名**（
  blender-mcp 已支持 .ai/.com 选择）。计价：文生 3D $0.10~0.20、图生 3D $0.20~0.30/次，
  加项 $0.05~0.20——科研预算可控。
- **定位判断**：生成结果是**任意尺度三角网格（GLB）**，无参数/无约束/无精确尺寸——
  **只做"原型/示意/参考"，不做精密 CAD**。科研用法三类：①实验装置/复杂实物照片→示意模型
  （进③域概念图）；②复杂曲面参考体（GLB 当视觉参照，在 build123d 里重新按尺寸建模）；
  ③演示/教学素材。**打印/仿真件一律回 build123d**（旧档红线"云网格不直接打印"照旧：非流形、
  任意缩放风险）。开源 TripoSR（VAST-AI-Research）为本地替代线索，但推理吃 GPU，CPU 机型不可行。
- **与硬件约束契合**：云端生成零本地算力，是 16GB 内存 CPU 机型上唯一的"AI 3D"可行路线
  （blender-mcp 内 Hunyuan3D 本地模式需 GPU，不可用）。

### 4.3 GUI vs 代码即模型分工

- **代码即模型（build123d）= 唯一几何真源**：一切有尺寸要求的几何进版本化脚本；agent 可直接
  改参/重跑/断言。这是"交付物必须来自版本化脚本"绑定规则的延续。
- **FreeCAD GUI = 人工后备**：STEP 检查/修补/编辑、看图沟通；不进自动化链（其自带 Python 与
  项目 venv 解耦，接入成本高）。
- **Blender GUI（+blender-mcp）= 场景探索**：构图/材质/灯光试做；**交付物冻结回 scene 脚本重跑
  无头**。无头渲染用已装 Blender（`MODELING3D_BLENDER`），pip bpy 340MB 不默认安装。
- **OpenSCAD 不设位**：其 CSG 场景 build123d 全覆盖，避免双 DSL。

### 4.4 meshio 复核（③域已拒 0/7，本域是否有不可替代位？）

**无，维持拒绝**。其生态位"网格格式互转"在本域被分解覆盖：gmsh 写 MSH/INP/VTK/STL 等、
trimesh 读 STL/OBJ/PLY/GLB/GLTF、pyvista（③域）读 VTK 系、COMSOL 原生导入 CAD/网格、
Tripo convert 管云端转换。且 0/7 差②、有替代 → 按口径直接拒绝。

## 5. 衔接设计

| 衔接 | 方向 | 载体 | 约定 |
|---|---|---|---|
| ⑥→② 仿真 | CAD→STEP→COMSOL | build123d 导 STEP → COMSOL-MCP `geometry_import`（已验证存在） | **单位 mm、坐标系右手、关键面命名注记**写入脚本注释；导入后在②域做几何校验（bbox 对账） |
| ⑥→② 网格（可选） | gmsh MSH/INP→COMSOL 网格导入 | 仅当 COMSOL 自划不满足时 | 需要 Nastran/Abaqus 网格导入器；默认不走 |
| ⑥→③ 绘图 | 概念渲染 PNG / STL 预览 | Blender 无头出 PNG→③ "外部媒体面板"；STL→pyvista 快速预览 | 出图尺寸/DPI/背景在③域排版契约里定；⑥只交素材 |
| ④→⑥ 理论 | 尺寸参数→几何断言 | 模板 §3 断言区（参数来源注记） | 理论值改动→脚本参数+断言同步 |
| ⑥→⑤ 文档 | 经③排版 | — | 不直接产文档 |

## 6. 形态（技能文档 + 社区工具 + 薄胶水）

- **技能文档**：SKILL.md（双管线不混、验收门红线、冻结回脚本规则、云网格不直接打印）+
  references（build123d 速查/验收门语义/Blender 无头坑点与引擎回退/Tripo v3 用法/衔接清单）+
  模板资产（print_part/scene 两脚本模板，内容取旧档知识不取代码）。
- **社区工具**：build123d、trimesh、gmsh（可选）、Blender(+bpy 可选)、blender-mcp、tripo3d SDK、
  FreeCAD（外部）。
- **薄胶水（自研面估算 ≈300–500 行）**：薄 CLI（scaffold/build/check/render/tripo/doctor 路由）
  + 验收门判据 ≈100 行 + 标记行协议解析 + Tripo SDK 编排 ≈100 行 + 配置/路径守卫。
  旧域 ≈77KB 自研代码（tools 57KB + blender 脚本 10KB + CLI/文档）不迁移，按上表只取能力。
- **QA**：新增最小 smoke（验收门判据单测、脚手架模板可执行性、标记行协议）。

## 7. 待用户裁决点

1. **首选组合确认**：build123d + trimesh + Blender 无头 + blender-mcp（豁免 pin：差②缺 2026-05，
   交互唯一通道）+ tripo3d SDK；CadQuery 仅备选记录不引入（极简原则）？
2. **gmsh 豁免 pin 为可选件**（pin 版本 + 原地 ADR 记 GPL/发版风险；主链不依赖）？
3. **Tripo 通道二选一**：官方 `tripo3d` SDK 豁免引入（推荐，差① star=61，厂商唯一 v3 SDK，
   MIT）vs 自写 v3 REST 薄胶水（≈200 行，零依赖但自研面+1）？（旧 v2 胶水已死，两者都要重做）
4. **渲染归属边界**：pyvista 场渲染归③已定；**Blender 概念渲染管线**（无头 studio 模板）建议
   留⑥（3D 资产→出图连贯，③只收 PNG 媒体排版），或整体移交③域？请明确"渲染不在本域"的边界。
5. **OpenSCAD 拒绝清单确认**（总调研 §4 曾误标差②，实测 7/7 全过；因有替代仍建议拒绝、备选记录）；
   meshio 拒绝维持确认。
6. **FreeCAD 定位确认**：外部 GUI 后备工具（不进 pip 依赖、不进自动化链）？

## 8. 数据核验记录与来源

**核验方法**：star/last-commit 用 `img.shields.io/github/stars|last-commit/<o>/<r>.json`；
口径②用 GitHub search API `q=repo:<o>/<r> committer-date:<月初>..<月末>` 的 total_count（每仓库
每月 1 查，2026-11-17 20:40~21:00 经 127.0.0.1:12450 代理）；PyPI 用 `pypi.org/pypi/<pkg>/json`。
已知偏差与补核：blender-mcp 仓更名 `ahujasid/mcp-for-blender`，commit 搜索索引不含该仓（422），
逐月改用 GitHub core API `repos/.../commits?since&until&per_page=1`（2026-11-17 21:15 补核：
**2026-05=0、6~11 月均有 → 6/7 差②**），并以 Atom feed（10-28~11-15 共 20 提交）交叉印证。
gmsh 上游 gitlab.onelab.info 匿名不可读仓库，逐月以 GitHub 自动镜像 live-clones/gmsh 代理核验；
官方 star 不可核验（镜像 135★）。

**来源 URL**：
- 旧档：`pySciWS/orchestration/skills/modeling3d/`、`src/pysci/skills/modeling3d/`、`pyproject.toml`（build123d/CadQuery 实测注记）
- build123d：https://github.com/gumyr/build123d 、https://pypi.org/pypi/build123d/json（deps: cadquery-ocp-novtk>=8.0）
- CadQuery：https://github.com/CadQuery/cadquery 、https://pypi.org/pypi/cadquery/json（2.8.0 deps: cadquery-ocp<8.0、trame-vtk、numba、casadi）
- FreeCAD：https://github.com/FreeCAD/FreeCAD 、https://www.freecad.org
- OpenSCAD：https://github.com/openscad/openscad
- trimesh：https://github.com/mikedh/trimesh 、https://pypi.org/pypi/trimesh/json
- gmsh：https://gmsh.info 、https://gitlab.onelab.info/gmsh/gmsh（匿名限流）、https://pypi.org/pypi/gmsh/json（4.15.2 @2026-03-24）、镜像 https://github.com/live-clones/gmsh
- Blender/bpy：https://github.com/blender/blender（镜像）、https://pypi.org/pypi/bpy/json（5.3.0 cp313 win_amd64 340MB @2026-11-10）
- blender-mcp：https://github.com/ahujasid/mcp-for-blender（原 blender-mcp，MIT）、commits Atom feed（10-28~11-15）
- Tripo：https://developers.tripo3d.ai/en/docs/introduction （v3 API、任务生命周期）、/en/docs/sdk（官方 SDK 矩阵：PyPI `tripo3d==1.0.0`）、/en/docs/models-convert（格式转换）、/en/docs/pricing（计价）、https://github.com/Tripo-Developer/tripo-python-sdk（61★）、社区 tripo-mcp：https://github.com/lxy2109/tripo-mcp（4★停更）
- meshio：https://github.com/nschloe/meshio（逐月 0/7）
- 逐月计数（2026-05..11）：build123d 65/68/70/117/155/173/44；CadQuery 6/8/8/4/7/6/4；
  trimesh 39/21/35/31/32/75/33；FreeCAD 442/408/380/508/621/348/163；OpenSCAD 4/41/56/61/50/54/46；
  blender/blender 790/1307/806/711/780/1522/647；live-clones/gmsh 32/29/81/132/693/416/309；
  meshio 全 0（0/7）。

## 用户裁决记录（2026-11-17）
- **组合=B 纯 Blender**：bpy 无头全链主力（建模/Cycles CPU 渲染/GLB-STL-OBJ 导出）；build123d/trimesh 不引入（STEP 需求出现再立项）；CadQuery/OpenSCAD/meshio 拒绝清单认可。
- blender-mcp 豁免 pin（差②，agent↔Blender 唯一通道）；Tripo 官方 SDK tripo3d 豁免 pin（差①，云生成只做原型/示意）。
- 渲染边界=Blender 概念渲染留⑥（③只收 PNG 媒体）；gmsh/FreeCAD 在纯 Blender 裁决下无需求位，不引入。
- Tripo 真实出图冒烟同⑦：key 现状待核，冒烟挂起日后补。
