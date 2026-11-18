# 科学计算/工程仿真"整合套件"社区调研

- 调研日期：2026-11-17
- 调研角度：仿真（②）+ 3D 建模（⑥）+ 科研绘图（③）+ 文档输出（⑤）这条工程链上，是否存在社区"整合套件"级的一站式方案；以及 COMSOL（既定仿真核心）的 Python/自动化集成现状。
- 数据来源：GitHub 仓库页 / raw README、repos.ecosyste.ms 元数据 API（star、许可证、pushed_at、tag 时间线）、mph.readthedocs.io、web 搜索摘要。api.github.com 匿名限流，未依赖。
- 元数据口径：star 数与最近推送时间取自 repos.ecosyste.ms（同步于 2026-11 上旬）；"近 6 个月每月均有提交"以 tag 时间线 / pushed_at 推断，取不到逐月提交数据时标注"未能核实（按推送/发版推断）"。

## 0. 核心结论（TL;DR）

1. **没有一个开源套件能一站式吃掉 ②仿真+⑥3D+③绘图（+⑤文档）并满足 PySci 引入标准**。最接近的两个"套件"是 FreeCAD（⑥强 + ②弱）和 SALOME/salome-meca（⑥中 + ②中 + ③中，但 star 极低且 salome-meca 无 Windows）。pyiron 是"材料计算工作流 IDE"而非通用工程链。结论：**工程链应按"社区工具组合 + 薄胶水"构建，而不是找一个套件**。
2. **COMSOL 集成最佳路径 = 官方 Java API（Method 文件）打底 + 社区 `mph`（Python 薄封装，事实标准）+ 可选 COMSOL-MCP（wjc9011）给 LLM agent 用**。COMSOL 没有官方 Python API/LiveLink for Python（官方论坛确认）；LiveLink for MATLAB 是官方唯一"脚本语言级"集成，但绑定 MATLAB。
3. **最小组合确认**：`COMSOL + mph`（②仿真 + ③内置出图）+ `CadQuery 或 build123d`（⑥参数化 CAD，替代 FreeCAD 作为代码级建模；FreeCAD 保留为 GUI/STEP 编辑器）+ `matplotlib`（③论文图）+ 文档能力域（⑤，另行调研）。`gmsh` Python API 作为轻量几何/网格备选。更强整合选择只有"ParaView/trame 生态"（③极强，Kitware 仿真工具包 SMTK 可作远期观察项），但都不覆盖 ②⑥全链。

## 1. 核心问题逐条回答

### 1.1 有没有一站式开源套件吃掉 ②+⑥+③(+⑤)？

**没有。** 候选套件逐一排查：

| 套件 | 覆盖 | 为什么不够 |
|---|---|---|
| FreeCAD (+FEM 工作台) | ⑥强、②弱、③弱 | FEM 工作台是 CalculiX/Elmer/Mystran 的 GUI 包装，物理场少、无多物理场耦合；③只有 TechDraw 工程图，不是科研绘图；⑤无 |
| SALOME / salome-meca | ⑥中、②中、③中 | GEOM 参数化 CAD + SMESH 网格 + Code_Aster 求解（salome-meca）+ PARAVIS(=ParaView) 后处理，表面看最"全"；但：GitHub 各模块 star 都 <50（不达标）；salome-meca 官方二进制仅 Linux（Windows 需 WSL/Docker）；Python API 存在但学习成本高、胶水自研量大 |
| pyiron | ②偏（原子尺度） | 材料计算工作流 IDE（LAMMPS/VASP/Gaussian），无 CAD、无科研绘图、无文档输出；核心包 pyiron_atomistics 仅 55★（不达标），umbrella 包 pyiron 470★ 但 2025-10 后停更 |
| MOOSE / Elmer / FEniCSx / MFEM / CalculiX / OpenFOAM / GetDP | ② | 都是求解器/库，不是套件；CAD 与绘图需外部工具 |
| ParaView / VisIt | ③ | 只做后处理可视化 |
| Blender (bpy) | ⑥⑦ | 建模/渲染强，但不是参数化工程 CAD（无约束求解/BREP 精确几何），与 FEM 衔接弱 |

**反向验证**：真正"一站式"的恰是商业软件——COMSOL（②③强 + ⑥内建 CAD/导入）、Ansys Workbench（②⑥③）、Fusion 360（⑥+②基本仿真）。开源侧没有对等物，故 PySci 只能走组合路线。

### 1.2 COMSOL 的 Python/自动化集成现状

**官方面**：
- **COMSOL API（Java）**：核心自动化接口，Method 文件（Application Builder 内）/ 外部 Java 程序均可调用；`ModelUtil` 等类是底层。COMSOL 6.4（当前版本）文档含 "Getting Started with the API"（含 Model Manager API）。来源：https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/model_manager_ref_api.60.02.html 、https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/COMSOL_ProgrammingReferenceManual.pdf
- **LiveLink for MATLAB**：官方"脚本语言级"集成（MATLAB 语法调 COMSOL）；另有 LiveLink for Excel / Simulink。COMSOL 论坛 2019 年官方答复：**无原生 Python API**，截至 2026 年搜索仍未见 "LiveLink for Python" 产品。来源：https://www.comsol.com/forum/thread/241931/comsol-python
- **COMSOL Server**：多用户/集群部署，Java API 远程调用；COMSOL Compiler 可打包独立应用。
- 结论：**Python 集成全部走"Java 桥"**，社区以 JPype 包装 Java API 成就了 mph。

**社区面**：

| 项目 | Star | 许可 | 最近活动 | 说明 |
|---|---|---|---|---|
| [MPh-py/MPh](https://github.com/MPh-py/MPh)（PyPI: MPh） | **609** | MIT | v1.4.0 发布于 2026-08-30（支持 Model Manager 载入模型；修复 Windows 弹窗）；仓库最后推送 2026-08-30 | Pythonic COMSOL 接口的事实标准。JPype 桥接 Java API；Windows 一等公民（1.3.0 起 Windows 默认 client–server 模式）；模型可导出 Java/MATLAB/VBA 源码；文档完善。维护模式为"发版驱动"：2025-10（1.3.0）→2025-11（1.3.1）→2026-08（1.3.2、1.4.0），9–11 月无提交 |
| [wjc9011/COMSOL_Multiphysics_MCP](https://github.com/wjc9011/COMSOL_Multiphysics_MCP) | **944** | MIT | pushed 2026-10-31；创建于 2026-02 | COMSOL-MCP：**基于 mph** 封装的 MCP server，100+ 工具（session/model/geometry/physics/mesh/study/results/electrochemistry/PDF 知识检索），含论文（Neurocomputing 2026, DOI 10.1016/j.neucom.2026.134481）。覆盖 COMSOL 5.x/6.x，Windows 可用（Python 3.10+，需 Java）。是 LLM agent 驱动 COMSOL 的最强社区实现 |
| [juijunnarkar/COMSOL_Multiphysics_MCP](https://github.com/juijunnarkar/comsol_multiphysics_mcp) | 18 | 标注 "other"（非标准 SPDX，需核实） | pushed 2026-11-04 | "Physics-aware MCP server via Java API"；活跃但规模小 |
| [XiaoSe532/comsol-codex-mcp](https://github.com/XiaoSe532/comsol-codex-mcp) | 0 | MIT | 仅 2026-06-25 一次性提交 | 不达标 |
| [Zhangyoupeng1996/Codex_MCP_Comsol](https://github.com/Zhangyoupeng1996/Codex_MCP_Comsol) | 0 | MIT | 仅 2026-06 停更 | 不达标 |

**推荐集成路径**：
1. 仿真脚本层：**mph**（`pip install MPh`，MIT，609★）。它就是"薄胶水"本身——自研只需写模型脚本。
2. Agent 驱动层（可选）：**wjc9011 COMSOL-MCP**（944★，MIT）。它也是社区薄胶水（对 mph 的 MCP 包装），符合"社区优先"；但注意其 Phase 6 集成测试仍在完善，工具覆盖以热/流/静电/声学/电化学为主。
3. 兜底/审计：mph 的 `model.save()` 可导出 Java 代码 → 与 Method 文件互转，保留官方 Java API 逃生通道。

### 1.3 "COMSOL/mph + FreeCAD/CadQuery + 薄胶水"是最小组合吗？有无更强选择？

**基本确认，但 ⑥ 的首选建议从 FreeCAD 调整为 CadQuery/build123d**（纯 Python 库、pip 安装、脚本即文档），FreeCAD 降级为 GUI 后备。加强版组合：

- **⑥ 3D 建模**：`CadQuery`（6105★，OCCT 内核，导出 STEP/STL → COMSOL `geometry_import`）或 `build123d`（3686★，更 Pythonic 的后继者）。二者都满足引入标准。FreeCAD 仍保留（35392★，最活跃）用于交互编辑/格式转换。
- **网格补充**：`gmsh`（GPL-2.0-or-later，Windows 版可用，Python API `gmsh.py`）——需要高质量网格或"几何+网格合一"时替代 FreeCAD FEM 的网格环节。
- **③ 绘图**：COMSOL 自带出图（mph 可编程导出）负责仿真云图/曲线；`matplotlib`（论文图，属科研绘图能力域）负责其余。`ParaView`（1729★，BSD-3）在需要大型结果后处理时加入（vtk/trame 生态是"更强整合"的方向，但仍不覆盖 ②⑥）。
- **⑤ 文档**：本调研范围内没有工具能覆盖；交文档能力域（Word/Typst 技能），仿真图与表格经 matplotlib/导出文件流入。

## 2. 开源候选逐项评估

评估维度：能力域（①-⑦，重点 ②③⑥⑤）/ Python 可编程性 / Windows / star·活跃度·许可 / LLM agent 可驱动性。判定口径：star ≥ 100；近 6 个月（2026-05—2026-11）每月均有提交；能减少自研代码量。

| 项目 | 能力域 | Python/无头 | Windows | Star | 近 6 月活跃 | 许可 | Agent 可驱动 | 判定 |
|---|---|---|---|---|---|---|---|---|
| [FreeCAD](https://github.com/FreeCAD/FreeCAD) | ⑥强 ②弱（FEM: CalculiX/Elmer/Mystran）③弱(TechDraw) | FreeCADCmd 无头 + Python API | ✅ 官方安装包 | 35,392 | ✅ weekly tag 至 2026-10-21，1.1.4 发版 | LGPL-2.1 | ✅（FreeCADCmd + Python） | **引入**（⑥主力/GUI 后备） |
| [CadQuery](https://github.com/CadQuery/cadquery) | ⑥强（参数化 BREP） | 纯 Python API | ✅ pip(windows wheel) | 6,105 | ✅ push 2026-11-09（逐月提交未能核实） | Apache-2.0（元数据解析为 other，以官方声明为准） | ✅ | **引入候选**（⑥代码级首选） |
| [build123d](https://github.com/gumyr/build123d) | ⑥强（CadQuery 后继） | 纯 Python API | ✅ pip | 3,686 | ✅ push 2026-11-10 | Apache-2.0 | ✅ | **引入候选**（与 CadQuery 二选一） |
| [OpenSCAD](https://github.com/openscad/openscad) | ⑥中（CSG 布尔建模） | CLI + .scad DSL（非 Python） | ✅ | 10,638 | ✅ push 2026-11-12 | GPL-2.0 系（COPYING 非标准 SPDX 解析） | ✅ CLI | 备选（CSG 场景） |
| [gmsh](https://gmsh.info) | ⑥中（几何+网格）②网格 | ✅ gmsh.py / CLI | ✅ | 未能核实（官方仓库在 gitlab.onelab.info 自托管，GitHub 仅第三方镜像） | ✅（4.x 持续发版，需核实） | GPL-2.0-or-later | ✅ | **能力型引入**（网格刚需） |
| [Salome / salome-meca](https://github.com/SalomePlatform) | ⑥中(GEOM) ②中(SMESH+Code_Aster) ③中(PARAVIS) | ✅ SALOME Python API（复杂） | ⚠️ SALOME 本体有 Windows 二进制；**salome-meca 仅 Linux**（WSL/Docker） | 各模块 0–46（smesh 46 / shaper 30 / medcoupling 26） | 部分模块 1–2 月内有推送 | LGPL-2.1 系 | ⚠️ 需大量胶水 | **不引入**（star 不达标、Windows 差、胶水重） |
| [Code_Aster](https://code-aster.org) | ②强（结构/热，salome-meca 的求解器） | ✅ `.comm` 文件 + Python（Eficas） | ❌ 原生仅 Linux | 未能核实（官方仓在自有 GitLab；GitHub 仅第三方镜像 ralic/Code_Aster 等） | 活跃（论坛 2026 仍在讨论开发） | GPL-2.0 系 | ⚠️ | **不引入**（Windows 不可用；作为 Linux 侧可选求解器记录） |
| [Elmer FEM](https://github.com/ElmerCSC/elmerfem) | ②强（多物理场）③中（ElmerPost/ParaView） | ✅ 输入文件 + Python 脚本 | ⚠️ 可 MinGW 编译（CI 有 build-windows-mingw），无官方安装包 → 体验一般 | 1,756 | ✅ push 2026-11-06 | GPL 系（LICENSE.md 非标准 SPDX，elmerfem.org/blog/license） | ✅ CLI | 备选求解器（FreeCAD FEM 后端） |
| [CalculiX](https://github.com/Dhondtguido/CalculiX) | ②中（结构/热，作者官方仓） | ✅ `.inp` 文件（Abaqus 兼容） | ⚠️ 社区 Windows 构建 | 234 | ✅ push 2026-11-10 | GPL-2.0 | ✅ CLI | 备选（FreeCAD FEM 默认后端） |
| [FEniCSx (dolfinx)](https://github.com/FEniCS/dolfinx) | ②强（PDE 框架） | ✅ Pythonic（UFL） | ⚠️ 主要 Linux/容器（conda-forge 有 Win 构建，未验证） | 1,250 | ✅ push 2026-11 中 | LGPL-3.0 系（需核实） | ✅ | 备选（自定义 PDE 场景） |
| [MOOSE](https://github.com/idaholab/moose) | ②强（多物理场框架） | ⚠️ Python 测试/Peacock GUI，核心 C++ | ⚠️ 官方推荐 WSL | 2,405 | ✅ push 2026-11-11 | LGPL-2.1 | ⚠️ | 不引入（Windows 差、学习曲线陡） |
| [OpenFOAM](https://github.com/OpenFOAM/OpenFOAM-13) | ②强（CFD） | ⚠️ 配置文件 + Python 后处理 | ❌ 原生无（WSL/Docker） | 版本仓 85★（Foundation 按版本分仓，star 稀释；ESI 版在 GitLab） | v13 仓 push 2026-05-26（版本仓节奏） | GPL-3.0 系 | ⚠️ | 不引入（Windows 不可用；CFD 需求再评估） |
| [MFEM](https://github.com/mfem/mfem)（+PyMFEM 301★） | ②中（FEM 库） | ✅ PyMFEM | ⚠️ 可编译 | 2,285 | ✅ push 约 2026-11-06 | BSD/LGPL 系（需核实 LICENSE） | ✅ | 不引入（库级，非工具链） |
| [GetDP](https://getdp.info) | ②中（电磁/热，gmsh 同门） | ✅ `.pro` 文件 + gmsh 工作流 | ✅ 有 Win 构建 | 未能核实（官方仓在 onelab 自托管 GitLab；GitHub getdp 组织与本项目无关） | 慢 | GPL-2.0 系 | ✅ | 不引入（star 无法核实、社区小） |
| [pyiron](https://github.com/pyiron/pyiron) / [pyiron_atomistics](https://github.com/pyiron/pyiron_atomistics) | ②偏原子尺度，工作流 IDE | ✅ Jupyter/Python | ✅ | 470（umbrella，2025-10 后停推）/ 55（atomistics，push 2026-11-16） | atomistics ✅ | BSD-3 | ✅ | 不引入（能力域不匹配；若做材料计算再看） |
| [AiiDA](https://github.com/aiidateam/aiida-core) | ①文献弱 ②工作流/数据溯源 | ✅ Python | ⚠️ 需 PostgreSQL+Linux 优先 | 612 | ✅ push 2026-11-13 | LICENSE.txt 非标准 SPDX（需核实） | ✅ | 不引入（工作流引擎，非必需） |
| [ParaView](https://github.com/Kitware/ParaView) | ③强（科学可视化） | ✅ pvpython + Python 过滤器 | ✅ | 1,729 | ✅ push 2026-11-13 | BSD-3 | ✅（pvpython/CLI） | **引入候选**（大型结果后处理时） |
| [VisIt](https://github.com/visit-dav/visit) | ③强 | ✅ Python CLI | ✅ | 539 | ✅ push 2026-11-09 | BSD-3 系 | ✅ | 备选（与 ParaView 二选一，选 ParaView） |
| [OpenSCAD→Blender(bpy)](https://github.com/blender/blender) | ⑥中 ⑦强（渲染） | ✅ bpy (headless `blender -b -P`) | ✅ | 21,906（官方镜像；上游 projects.blender.org） | ✅ 镜像 push 2026-11-12 | GPL-2.0-or-later | ✅ | 备选（渲染/示意图；非工程 CAD） |
| [VTK](https://github.com/Kitware/VTK) / [trame](https://github.com/Kitware/trame) | ③强（可视化库/Web） | ✅ | ✅ | 3,234 / 712 | ✅ | BSD-3 | ✅ | 远期观察（trame=Python Web 可视化胶水） |
| [SMTK](https://github.com/Kitware/SMTK) | ②⑥集成工具包（Kitware 仿真建模工具链） | ✅ Python 绑定 | ⚠️ | 60 | ✅ push 2026-11-10 | BSD-3 系 | ✅ | 观察项（star 不达标；"仿真集成套件"的另一种思路） |

注：能力域 ①④⑦ 不在本角度重点，表中仅顺带标注。"能减少自研代码量"一项：凡判定"引入"者均可直接 pip/apt 安装并以薄胶水调用，显著减少自研。

## 3. 商业工具对照（许可/价格模式）

| 产品 | 覆盖 | 价格模式（约价，具体以官方/经销商询价为准） | 对 PySci 的意义 |
|---|---|---|---|
| COMSOL Multiphysics | ②③强 ⑥中 | 永久许可 + 年度服务费，或年度订阅；按"核心 + 模块"计价、按席位/网络浮动许可；学术价显著低；**询价制**（高校 WebStore 显示按年度订阅许可） | **既定仿真核心**；Java API + mph 决定自动化上限 |
| MATLAB/Simulink | ④②弱 ③ ⑤ | 年度订阅（按人 + 工具箱加购；学术/学生价低；家庭版更低）| LiveLink for MATLAB 是 COMSOL 官方集成；PySci 以 Python 替代，不必引入 |
| Mathematica | ④③⑤ | 年度订阅或永久+服务费；学术价 | 与 SymPy（④理论计算域）重叠，不引入 |
| Ansys Workbench | ②⑥③ | 询价制，年度租约/永久+维护，商业价高（数万级）；学术打包价 | 对照标杆；无 Python 一等 API（PyMAPDL 等社区/官方插件存在） |
| SimScale | ②（CFD/FEM/热） | SaaS 订阅（免费社区版受限 + 按席位年订，询价） | 云 CAE 对照；本地不可控，不引入 |
| Origin / OriginPro | ③⑤ | 永久许可 + 维护费（学术个人价低，商业价高） | 科研绘图商业标杆；matplotlib 生态替代 |
| Maple | ④⑤ | 永久 + 维护 或 订阅；学术价 | 同 Mathematica，不引入 |
| Fusion 360 | ⑥②（基本仿真/CAM） | 年/月订阅；个人非商用免费层 | ⑥商业对照；无开放 API 生态优势（有 Python API 但绑定云端许可） |

（价格未逐一核实到 2026 具体数字，均为模式描述 + 量级参考，已在表中标注"约价/询价制"。）

## 4. PySci 引入判定汇总（口径：star≥100 / 2026-05—11 逐月提交 / 减少自研）

| 候选 | star≥100 | 逐月提交 | 减自研 | 结论 |
|---|---|---|---|---|
| FreeCAD | ✅ 35,392 | ✅（weekly tag 贯穿） | ✅ | **引入**：⑥ GUI/格式转换主力 |
| CadQuery | ✅ 6,105 | ⚠️ 持续推送但逐月明细未能核实 | ✅ | **引入**（或 build123d）：⑥ 代码级建模 |
| build123d | ✅ 3,686 | ⚠️ 同上 | ✅ | 与 CadQuery 二选一（建议先 CadQuery，API 更成熟；追求 Pythonic 可选 build123d） |
| MPh (mph) | ✅ 609 | ⚠️ 2026 内 5–8 月活跃、9–11 月无提交 | ✅✅ | **引入**：COMSOL 唯一现实 Python 通道；"发版驱动"维护，风险低（封装面小） |
| COMSOL-MCP (wjc9011) | ✅ 944 | ✅（2026-02 创建后持续，push 2026-10-31） | ✅ | **引入（可选层）**：LLM agent 驱动 COMSOL；须绑定 mph 使用 |
| ParaView | ✅ 1,729 | ✅ | ✅ | **按需引入**：大型结果后处理 |
| gmsh | ⚠️ 未能核实 | ⚠️ | ✅ | **按能力引入**：网格刚需时用其 Python API（GPL，注意与自研代码的许可证边界） |
| 其余（Salome/Code_Aster/Elmer/MOOSE/OpenFOAM/FEniCSx/MFEM/GetDP/pyiron/AiiDA/VisIt/OpenSCAD/Blender） | 各有不达标项 | — | — | 不引入 / 备选记录（见 §2 表） |

## 5. 推荐的最小组合（工程链）

```
⑥ 参数化 CAD:  CadQuery（首选）/ build123d   → STEP/STL
                 └─ FreeCAD（GUI 后备、格式转换）
⑥ 网格:        gmsh（Python API，按需）        → mesh（COMSOL 亦可自划）
② 仿真:        COMSOL 6.x  ←  Python 胶水 = mph（+ 可选 COMSOL-MCP 给 agent）
③ 绘图:        COMSOL mph 导出图（云图/曲线） + matplotlib（论文图） [+ ParaView 按需]
⑤ 文档:        文档能力域（Word/Typst 技能）；图像/表格由上述工具产出后流入
```

自研面收敛为：**各工具之间的调用脚本（薄胶水）**——参数传递（几何参数 → CadQuery → STEP → COMSOL import）、结果导出与绘图约定、以及少量 MCP 包装（若不用 wjc9011）。

## 6. 来源 URL

COMSOL 集成：
- https://github.com/MPh-py/MPh ；https://mph.readthedocs.io/en/stable/releases.html（1.4.0 @2026-08-30）
- https://github.com/wjc9011/COMSOL_Multiphysics_MCP ；https://raw.githubusercontent.com/wjc9011/COMSOL_Multiphysics_MCP/master/README.md
- https://github.com/juijunnarkar/comsol_multiphysics_mcp ；https://github.com/XiaoSe532/comsol-codex-mcp ；https://github.com/Zhangyoupeng1996/Codex_MCP_Comsol
- https://www.comsol.com/forum/thread/241931/comsol-python （无官方 Python API）
- https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/model_manager_ref_api.60.02.html ；https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/COMSOL_ProgrammingReferenceManual.pdf

开源工具元数据（star/许可/pushed_at，repos.ecosyste.ms）：
- https://repos.ecosyste.ms/hosts/GitHub/repositories/FreeCAD%2FFreeCAD
- https://repos.ecosyste.ms/api/v1/hosts/GitHub/repositories/CadQuery%2Fcadquery 、.../gumyr%2Fbuild123d 、.../openscad%2Fopenscad 、.../ElmerCSC%2Felmerfem 、.../idaholab%2Fmoose 、.../Dhondtguido%2FCalculiX 、.../aiidateam%2Faiida-core 、.../pyiron%2Fpyiron 、.../pyiron%2Fpyiron_atomistics 、.../OpenFOAM%2FOpenFOAM-13
- https://repos.ecosyste.ms/hosts/GitHub/owners/FEniCS （dolfinx 1,250★）、.../owners/visit-dav （visit 539★）、.../owners/mfem （mfem 2,285★、PyMFEM 301★）、.../owners/Kitware （ParaView 1,729★、VTK 3,234★、trame 712★、SMTK 60★）、.../owners/blender （blender 21,906★）、.../owners/SalomePlatform （smesh 46★ 等）
- gmsh：https://gmsh.info （官方自托管 GitLab，star 未能核实）；GetDP：https://getdp.info
- Code_Aster：https://code-aster.org ；SALOME：https://www.salome-platform.org 、https://git.salome-platform.org

商业对照：
- https://www.comsol.com （询价/学术许可）；https://ww2.mathworks.cn/pricing-licensing.html
- https://www.simscale.com/product/pricing/ ；https://de.maplesoft.com/products/maple/pricing/
- https://www.originlab.com/index.aspx?go=Purchase&PricingAndOrdering&pid=4408
- https://www.autodesk.com/products/fusion-360/blog/fusion-subscription-types/
- 高校 COMSOL 年度订阅示例：https://webstore.illinois.edu/shop/product.aspx?zpid=7304

## 7. 风险与待核实项

1. MPh"逐月提交"不满足严格口径（2026-09—11 无提交），但属发版驱动型维护且无可替代，建议以"pin 版本 + 薄封装"方式引入并监控。
2. CadQuery/build123d/gmsh 的"逐月提交"仅由 pushed_at 推断，逐月明细未能核实（api.github.com 限流）。
3. OpenFOAM/Gmsh/GetDP/Code_Aster 因仓库分散或自托管，star 口径失真/缺失，已在文中标注。
4. 若未来 salome-meca 提供 Windows 原生支持、或 COMSOL 官方推出 Python LiveLink，§5 组合需重估。
