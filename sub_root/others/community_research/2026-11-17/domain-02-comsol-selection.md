# 能力域②COMSOL 仿真 · 选型细化调研报告（2026-11-17）

> 状态：待用户审核。工作 Agent 按"新需求三段式·选型"产出。
> 判定口径（用户已裁决通用口径）：① star ≥ 100；② 2026-05 ~ 2026-11 **逐月**有提交；③ 引入可减少自研代码量。
> 口径②豁免原则（2026-11-17 裁决）：**按无可替代性分级豁免**——无可替代者豁免自动生效（pin 版本 + 原地 ADR 记录维护风险）；有替代余地者不豁免、仍拒绝。
> 硬件约束：仅 CPU、无独显、16GB 内存（可用 ~5GB）。仿真本体 COMSOL 是外部商业软件不受此约束；Python 侧避免重型本地计算。
> 输入：`00-synthesis.md`（总调研+裁决）、`science-engineering-suites.md`（C 线）、`domain-03-plotting-selection.md`（③域移交项 #9/#10）、旧 pySciWS 仿真域参考实现（仅对照，**严禁复制代码**）。

## 0. 结论速览

1. **首选组合：mph（Python 通道，豁免 pin 1.4.0）+ COMSOL-MCP（agent 层，豁免 pin commit，可选）+ ②域技能文档 + 薄胶水**。
   mph 是 COMSOL 唯一现实 Python 通道（官方无 Python API），COMSOL-MCP 是唯一成熟的 agent 驱动层（100+ 工具，Neurocomputing 2026 论文背书）。二者均**差口径②**，但均属"无可替代"或"临界活跃"，按豁免原则处理。
2. **本机环境已探测且兼容**：COMSOL Multiphysics **6.4.0.293**（2026-09-26 装于 `D:\XiGPrograms\comsol\6.4\base`，注册表 `HKLM\SOFTWARE\COMSOL\COMSOL64\COMSOLROOT`）；mph 走**注册表发现**，可找到该非默认路径安装。旧项目 venv 实证 mph 1.4.0 + JPype1 1.7.1 + Python 3.13 + COMSOL 6.4 跑通（有 solve/export 产物存档）。
3. **旧能力清单 16 项**：社区工具直接覆盖约 5 项；内容资产（原则/配方/坑点/声明模板）迁移约 4 项；**真正需保留的差异化薄胶水约 5 项（≈600–1000 行）**——核心是"出版级出图 + sidecar"（mph/COMSOL-MCP 均无此控制面）；不保留/移交约 4 项。对比旧实现 10 个 Python 模块（约数千行）+ 配套脚本，自研面大幅收窄。
4. **③域移交项裁定建议**：COMSOL 栅格 sidecar 叠加——**生产端（sidecar.json）随出版级出图保留**，消费端 overlay **薄化 ≈200 行入②域**（或降级为 imshow 配方文档）；imgcheck（logo 擦除配套）**不保留**（文档化 COMSOL 原生 `logo*d=off`）。
5. **待裁决 6 点**（§6）：COMSOL-MCP 豁免 vs 直驱 mph、手册知识管线三选一、sidecar 消费端形态、出图薄胶水范围、domain-check 去留、pyvista 渲染移交③确认。

## 1. 任务一：旧能力清单（仅对照，严禁复制代码）

来源：`pySciWS/orchestration/skills/comsol-simulation/`（SKILL.md + references 6 册）、
`src/pysci/skills/comsol_simulation/tools/`（10 模块）、`tests/skills/comsol_simulation/`（9 测试）、
`data/skills/comsol_simulation/`（手册 MD/FTS5/坑点库/runs 产物）；③域移交项见 `domain-03-plotting-selection.md`。

| # | 能力项 | 旧实现方式（仅对照） | 新形态覆盖 | 是否保留（建议） |
|---|---|---|---|---|
| 1 | 会话/许可证管理：standalone 与 client–server 启动、持久 `comsolmphserver` + `--connect-port` 复用、单许可证占用探测（**jvm.dll 判据**，`_jpype.pyd` 误报）、重复实例检测 | `tools/session.py` + `tools/mcp_server.py`（WMI 脱离派生、HTTP probe、license_holders） | mph.start()/Server（Windows 1.3.0 起默认 client–server）；DSH MCP 宿主托管进程 | **保留（薄化 ≈100 行）**：许可证探测判据与单证编排是本机（单许可证）刚需；spawn 逻辑尽量依托 DSH |
| 2 | 通用模型驱动：load/create/inspect、参数、几何/物理/网格、求解、evaluate/export | 社区 `comsol` MCP（旧项目已 pin upstream + UPSTREAM.lock） | **COMSOL-MCP**（同一社区仓） | **保留（社区工具）** |
| 3 | 参数化建模 helper + recipes：block/rect/circle/polygon/array、材料、物理、网格、参数扫描、外部 geom/mesh 导入（STEP）、recipe 注册表 | `tools/build.py`（自研） | mph Node/create + COMSOL-MCP geometry(16)/physics(28)/mesh(4) 工具（含 `geometry_import`） | **保形态不保代码**：recipe 内容（如 waveguide2d_acpr）作配方迁技能文档 references；几何 helper 被 MCP 工具替代 |
| 4 | 求解/批处理：`run solve` + **plot group 自动重绑新 dataset**、comsolbatch 批跑、参数扫描、problems 扫描、Java stack 提取 | `tools/run.py` | mph.solve()/inner()/outer() + COMSOL-MCP study_solve(_async)/study_wait | **保留（薄化）**：自动重绑 ≈50 行（solve-then-export 跨调用安全的关键）；Java stack 提取为排错配方 |
| 5 | **出版级出图**：`export image` 色标范围/polar-rmax/几何 bbox/extent 控制、`<png>.sidecar.json`（crop_box_px/extent_recovered/blank 自检）、`--clean` 隐藏 colorbar/legend/title、品牌 logo 剥离 | `tools/export.py` + `postprocess.py`（detect_frame_box/strip_brand 等） | **无社区覆盖**——mph `Model.export()` 只跑导出节点，无像素级控制；COMSOL-MCP `results_export_image` 同样裸 | **保留（差异化薄胶水核心，≈300–500 行）** |
| 6 | 数据/表/网格导出：CSV/VTK 场数据（--expr/--selection/--filter-indicator）、**export polar**（参数化弧线扫角）、table、mesh、geometry | `tools/export.py` | mph evaluate()/export() + COMSOL-MCP results_export_data/results_evaluate | **保留（薄化）**：export polar 为远场扫角特有→薄胶水或配方文档 |
| 7 | 结果渲染与物理验证：pyvista render/slice/probe/integrate、field_stats、mesh_quality、convergence_order、compare_to_reference、check_conservation、validate_field、transfer_function/db/刚性波导解析 | `tools/postprocess.py` | 渲染→③域 pyvista（已定）；验证数值核心 SciPy/NumPy 轻量（CPU ✓） | **部分保留**：验证函数族（参考比对/守恒/收敛阶）作②域薄胶水/配方（研究特有）；渲染系列移交③域 |
| 8 | 模型内省：tree/params/inventory/node dump、**feature-types 试建探测**（COMSOL 无枚举 API）、`node set` 对称改属性、Java 源摘要 | `tools/inspect.py` | mph.tree()/inspect()/Node.property() 部分覆盖 | **保留（薄化 ≈150 行）**：feature-types 探测 + node set 是 agent 高频刚需，社区无 |
| 9 | 手册知识管线：MinerU 云转换 5 本手册→MD、FTS5 索引、`docs search/read` | `tools/docs.py` + `cache/doc_index.db` | COMSOL-MCP `pdf_search`（chromadb + sentence-transformers RAG）/ 或 agent 直接检索 MD | **待裁决**（三选一，§6-2） |
| 10 | 域几何声明四问（domain-check：域尺寸/波长/工作点-异常点偏移，EP 研究联动） | `tools/domain.py` | 无社区（研究流程约束） | **保内容不保代码**：并入技能文档"开工/交付检查清单"；机械检查脚本 ≈100 行可选 |
| 11 | MCP server 托管：`mcp ensure/status/stop`、脱离派生、Origin 403 排障、版本串归一（`6.4.0`→`6.4`） | `tools/mcp_server.py` | DSH MCP 配置机制 + 社区 server 自身 | **薄化**：排障知识（五约束/许可证坑）→技能文档；版本串归一 ≈20 行 |
| 12 | 仿真三原则（勤整理/最大化复用/人类终检）+ Java API 坑点库 | SKILL.md references + `knowledge/java_api_pitfalls.md` | 无社区（经验资产） | **保留（纯内容迁移，零代码）** |
| 13 | ③域移交：**COMSOL 栅格 sidecar 数据坐标叠加**（消费端：crop_box_px→`imshow(extent=…)` 精确像素↔数据映射、overlay 原语、Rectangle 锚点旋转修正） | ③域 `raster.py` 19KB（生产端在本域 #5） | 无社区（裸 imshow 无轴框映射） | **待裁决**：薄化 ≈200 行入②域 vs 文档化 imshow 配方（§6-3）；生产端随 #5 保留 |
| 14 | ③域移交：imgcheck 图像体检（COMSOL logo 擦除流程配套） | ③域 `imgcheck.py` 12KB | 无社区 | **不保留**：品牌剥离用 COMSOL 原生 `options*d=on`+`logo*d=off`（零代码），像素擦除自检并入 #5 薄胶水 |
| 15 | ④域衔接：S 矩阵 COMSOL 验证（cmt_comsol_validation，同入同出比对） | research 平行树 | 无社区 | **保形态**：④域调用②域薄 Python API；实现归实施期/研究树 |
| 16 | 测试面：test_cli/config/export_scale/run_rebind/build_registry/node_set/mcp_server/docs_index/postprocess_vista | `tests/skills/comsol_simulation/` 9 件 | — | QA 重建参照（三段式第三段），不复制 |

**小结**：#2/#3 大部被社区工具吃掉；#5/#8/#13 是无可替代的差异化胶水；#1/#4/#6/#7/#11 薄化保留；#9/#10/#13 有裁决空间；#14 不保留。自研预算 ≈**600–1000 行**（若 sidecar 消费端文档化、domain-check 砍则 ≈400–700 行）。

## 2. 任务二：社区候选细化核实

### 2.1 核实方法与数据源

- 本机探测：注册表（`HKLM\SOFTWARE\COMSOL`、Uninstall 键）+ 常见路径，只读；旧项目 `.venv` dist-info 对账。
- 三口径数据：`img.shields.io/github/stars|last-commit`（2026-11-17 实测）+ ecosyste.ms commits API 逐月探针（复用 `_scripts/d03_month.mjs`，presence 判定 2026-05..11）。
- API 面：mph 官方文档（readthedocs stable = 1.4.0：Model/discovery/releases 页）+ COMSOL-MCP raw README（工具表逐项）。
- 兼容性：mph discovery 机制文档 + 旧项目运行产物实证（`data/skills/comsol_simulation/runs/verify_beam/` 等有 6.4 下 solve/export 产物）。

### 2.2 本机 COMSOL 环境探测（2026-11-17，只读）

| 项 | 值 |
|---|---|
| 安装 | COMSOL Multiphysics **6.4.0.293**（Uninstall 键 DisplayName=COMSOL Multiphysics 6.4，InstallDate=20260926） |
| 安装根 | `D:\XiGPrograms\comsol\6.4\base`（注册表 `HKLM\SOFTWARE\COMSOL\COMSOL64\COMSOLROOT`；**非 C 盘默认路径**） |
| 目录完整性 | bin/java/lib/license/doc/… 齐全；自带 JRE `java\win64\jre` 存在（mph/JPype 所需 JVM 由 COMSOL 自带，无需另装 JDK） |
| mph 发现机制 | **Windows 走注册表**（`search_registry`）+ `where comsol` 兜底——注册表键在，故可发现该 D 盘安装（mph.discovery 文档确认） |
| 旧项目实证 | pySciWS `.venv`：**mph 1.4.0 + JPype1 1.7.1**（Python 3.13）；runs/ 有 COMSOL 6.4 下 solve、export PNG/CSV、sidecar 产物 → **mph 1.4.0 ↔ COMSOL 6.4.0 兼容已实证** |
| 已知坑 | 版本串归一：完整串 `6.4.0` 传给 mph discovery 会 exact-match 失败，需短名 `6.4`（旧项目 cli-02 实测记录）；JPype JVM 启动后不可卸载，disconnect **不释放**许可证，须停进程（旧 mcp-01 实测） |
| 许可证 | 单机单许可证（旧项目"single-license box"）；COMSOL-MCP/mph 显式 start 才占证 |

### 2.3 候选指标总表

| 候选 | ★（11-17） | 许可证 | 版本/最近提交 | 逐月提交 2026-05..11 | API 面/工具面 | agent 可编程性 | 与本机 6.4 兼容 |
|---|---|---|---|---|---|---|---|
| **mph**（MPh-py/MPh，PyPI `MPh`） | 611 | MIT | v1.4.0 @2026-08-30（1.3.2 @08-16）；last commit **august** | 5✓ 6✗ 7✓ 8✓ 9✗ 10✗ 11✗（3/7） | Model 全套：inspect 17 查询/create/property/remove/import/export/solve/mesh/build/evaluate(inner·outer)/save(·java·m·vba)；Node/tree/inspect；`.java` 逃生通道全量 Java API | 纯 Python 库，agent 直接写脚本调用；无 MCP | ✅ 实证（1.4.0+6.4.0.293+Py3.13）；注册表发现 ✓ |
| **COMSOL-MCP**（wjc9011/COMSOL_Multiphysics_MCP） | 981 | MIT | 创建 2026-02；last commit **october（10-31）**；含 Neurocomputing 2026 论文（DOI 10.1016/j.neucom.2026.134481） | 5✓ 6✓ 7✓ 8✓ 9✓ 10✓ **11✗**（月中临界） | **101 工具**：session 4 / model 9 / param 5 / geometry 16 / physics 28 / electrochemistry 10（对 6.3 活会话验证 26 接口 600+ 类型）/ mesh 4 / study 8（含 async+progress+cancel）/ results 9 / knowledge 8 + 4 MCP resources | MCP server（opencode/Claude/DSH 接入）；基于 mph | ✅ 5.x/6.x；README 要求 Python 3.10+、Java（COMSOL 自带 JRE 即可） |
| juijunnarkar/comsol_multiphysics_mcp | 18 | "other"（非标准 SPDX） | pushed 2026-11-04 | 活跃 | Java API 直包的 MCP | MCP | 未核 | 
| XiaoSe532/comsol-codex-mcp | 0 | MIT | 2026-06-25 一次性 | ✗ | — | — | — |
| FEniCSx（dolfinx） | — | LGPL-3.0 | 活跃 | — | 开源 FEM 全套（Python 原生） | 高 | **对照项**：非 COMSOL；换建模范式+重学成本，仿真本体用户既定 COMSOL → 不引入 |
| gmsh | 135（贴线） | GPL-2.0 | 活跃 | — | 几何/网格 | 高 | **⑥域管**（C 线裁决）；COMSOL 也可自划网格 |
| ParaView | 1729 | BSD-3 | 活跃 | — | 大型后处理/VTK | 中 | **③域按需**（pyvista 已覆盖轻量段）；不在②引入 |
| LiveLink for MATLAB | — | 商业 | — | — | 官方脚本级集成 | — | **拒绝**：绑定 MATLAB 商业许可，违背自包含 |
| 自写 JPype 封装 | — | — | — | — | = 重造 mph | — | **拒绝**：违背减自研（口径③） |

### 2.4 三口径/豁免判定

| 候选 | ①★≥100 | ②逐月提交 | ③减自研 | 判定 |
|---|---|---|---|---|
| **mph** | ✅ 611 | ❌（仅 5/7/8 月；发版驱动：1.3.0→1.3.1→1.3.2/1.4.0 节奏） | ✅✅（唯一 Python 通道的薄封装本体） | **豁免引入（自动生效）**——无可替代：COMSOL 官方无 Python API（官方论坛 2019 明示、至今无 LiveLink for Python）；COMSOL-MCP 也以 mph 为底座；直接 Java/JPype 自包 = 自研 mph。**pin 1.4.0 + 原地 ADR 记"发版驱动维护"风险**（封装面小、`.java` 逃生通道兜底） |
| **COMSOL-MCP** | ✅ 981 | ⚠️ 临界：仅缺 2026-11（17 日，距上次提交 17 天；977→981★仍在涨） | ✅（100+ 工具 + 会话/许可证管理，替代大量自研） | **待用户裁决（推荐豁免 pin 引入）**——严格口径"有替代"（agent 可直驱 mph 脚本），但替代意味着自写 MCP/工具层（自研面 +200–400 行，口径③反噬）；且"11 月无提交"是**月中观察**不是停更证据。建议：豁免引入（可选 agent 层）pin commit + ADR；或观察至 11 月底复测 |
| juijunnarkar | ❌ 18 | — | — | **拒绝**（差①） |
| comsol-codex-mcp | ❌ 0 | ❌ | — | **拒绝**（差①②） |
| FEniCSx/LiveLink/自包 | — | — | ❌ | **拒绝**（既定 COMSOL / 商业绑定 / 违背减自研） |

### 2.5 COMSOL-MCP 安全面与许可证需求（细化核实）

- **怎么起 COMSOL**：显式 `comsol_start` 才启动（mph.start() 单例 client；Windows 走 client–server 后端）；`comsol_connect` 连远程 COMSOL Server；**不显式 start 不占证**（旧项目 2026-10-01 实测"Nothing starts implicitly"）。mph 单例限制：每 Python 进程一个 Client，`comsol_disconnect` 清会话但 **JPype JVM 不卸载**，释放许可证须杀进程。
- **许可证需求**：本机 COMSOL 6.4 单许可证；MCP server 空闲态（未 start）不占证，故可常驻；一旦 start 即独占，CLI/GUI 须避让（旧项目单证编排经验直接可迁）。
- **安全面**：100+ 工具**全权无分层**（可建模/求解/写 `comsol_models/` 文件/跑长解算）；对 agent 是"COMSOL 本体全权"，风险=误操作模型与占满 CPU/内存（16GB 机器建议限核求解，旧 `_resolve_cores` 思路迁技能文档）；RAG 知识库（`pdf_search`）需额外装 pymupdf+chromadb+sentence-transformers，嵌入模型 CPU 可跑（小模型，~0.5GB）但**可选**。
- **成熟度**：Phase 1–5 Done、Phase 6 集成测试 In Progress（README 自述）；工具面属性名/边界条件表有实测记录（README "Key Technical Discoveries"），但覆盖以热/流/静电/声学/电化学为主。

## 3. 衔接设计（③绘图 / ⑤文档 / ④计算 / ⑥3D / ①文献）

| 衔接 | 路线 | 归属 |
|---|---|---|
| ②出图→③重绘（出版链主路线） | COMSOL `export data`/`evaluate`→CSV→**③域 matplotlib/ultraplot 重绘**（期刊样式契约、EPS/PDF 导出、audit） | ③域技能已定；②域只管导出格式约定 |
| ②出图→③栅格叠加（③移交 #9） | `export image`+**sidecar.json**（crop_box_px/extent_recovered）→薄 overlay `imshow(extent=…)` 精确像素↔数据映射 | **生产端归②域（随 #5 保留）**；消费端待裁决（薄化 ≈200 行入② 或文档化） |
| ②场图→③ 3D/truthful colorbar | COMSOL export VTK→**③域 pyvista + truthful colorbar** | ③域（#8 已裁决保留）；②域 postprocess 渲染系列移交 |
| ②图表→⑤文档 | 出版图/CSV 表→⑤域引用注入（zotero-mcp→refs.bib、LaTeX 单轨排版）；②域交付物规范（mph 双通道、人类终检清单）写入技能文档 | ⑤域技能 |
| ④理论→②验证 | SymPy/SciPy 推导（EP 位置/色散）→②域 COMSOL 数值验证（同入同出 S 参数比对，旧 cmt_comsol_validation 形态） | ②域暴露薄 Python API（函数签名级），④域按需 import |
| ⑥几何→②导入 | CadQuery/build123d 导 STEP→COMSOL `geometry_import`（COMSOL-MCP 工具已有） | ⑥域产出、②域接 |
| ①手册→②知识 | COMSOL 手册 PDF→MinerU **云 API**（①域已裁决）→MD；检索端三选一（§6-2） | ①域转换管线复用；检索端归②域 |

**硬件约束核对**：mph/COMSOL-MCP 均为 Java 桥轻量 Python（计算在 COMSOL 进程）；后处理验证用 NumPy/SciPy CPU 轻量；pyvista 离屏渲染 CPU 可行（③域）；RAG 嵌入为 CPU 小模型但可选。**Python 侧无重型本地计算，约束满足**。

## 4. 建议组合与薄胶水估计

**首选（推荐）**：
1. **mph 1.4.0**（豁免 pin + ADR）——Python 通道与薄胶水底座；
2. **COMSOL-MCP**（豁免 pin commit + ADR，**可选层**；或观察至 11 月底）——agent 通用驱动（101 工具）；
3. **②域技能文档**：双驱动分工表、单证编排、仿真三原则、Java API 坑点、域声明检查清单、排障（空白 PNG/版本串/JVM 不卸载）、手册检索工作流；
4. **薄胶水（自研预算 ≈600–1000 行）**：出版级 export+sidecar（≈300–500）｜feature-types 探测 + node set（≈150）｜plot-group 自动重绑（≈50）｜验证函数族（≈100–200）｜会话/许可证探测（≈100）｜sidecar overlay 消费端（≈200，可选）。

**次选（依赖面最小）**：仅 mph + agent 直驱 Python 脚本（不引 COMSOL-MCP）：自研面 +200–400 行（会话管理/工具化），失去 async solve/progress、电化学注册表等现成面。

**拒绝**：FEniCSx（换范式）、LiveLink for MATLAB（商业绑定）、自写 JPype 封装（=自研 mph）、juijunnarkar 18★、comsol-codex-mcp 0★、ParaView/gmsh（归③/⑥域）。

## 5. 待用户确认的取舍点

1. **COMSOL-MCP 豁免与否**（差②临界：11 月仅 17 天未提交，10-31 仍活跃）：A 豁免 pin 引入为可选层（推荐）｜B 观察至 11 月底复测逐月数据再定｜C 严格口径拒绝、走 mph 直驱（自研面 +200–400 行）。
2. **手册知识管线三选一**：A 复用 COMSOL-MCP `pdf_search`（RAG，chromadb+sentence-transformers 可选依赖）｜B 薄 FTS5 自建（旧 docs.py 形态，≈150 行，依赖面最小）｜C agent 直接检索 MinerU 转换 MD（零代码）。
3. **③域移交 #9 sidecar 消费端**：A 薄化 ≈200 行 overlay 入②域（像素精确叠加保能力）｜B 仅文档化 `imshow(extent=…)` 配方（自研面最小）。（生产端 sidecar.json 无论 A/B 都随出版级出图保留。）
4. **出版级出图薄胶水范围确认**：保留 color-range/polar-rmax/geom-bbox/extent/sidecar/blank 自检；品牌剥离只用 COMSOL 原生 `logo*d=off`+文档化（imgcheck 像素擦除砍）——是否认可？
5. **domain-check 域几何声明**：内容并入技能文档检查清单（零代码）｜保留机械检查脚本 ≈100 行（EP 声明四问硬门）。
6. **pyvista 渲染系列移交③域确认**（postprocess render/slice/probe 不入②域，验证函数族留②域）。

## 6. 来源 URL 清单

- mph：https://github.com/MPh-py/MPh ；https://mph.readthedocs.io/en/stable/releases.html （1.4.0 @2026-08-30）；https://mph.readthedocs.io/en/stable/api/mph.Model.html ；https://mph.readthedocs.io/en/stable/api/mph.discovery.html （注册表发现）
- COMSOL-MCP：https://github.com/wjc9011/COMSOL_Multiphysics_MCP ；https://raw.githubusercontent.com/wjc9011/COMSOL_Multiphysics_MCP/master/README.md （工具表/要求/安全面）；论文 DOI https://doi.org/10.1016/j.neucom.2026.134481
- COMSOL 集成现状：https://www.comsol.com/forum/thread/241931/comsol-python （官方无 Python API）；https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/model_manager_ref_api.60.02.html
- 其他候选：https://github.com/juijunnarkar/comsol_multiphysics_mcp ；https://github.com/XiaoSe532/comsol-codex-mcp
- 三口径数据：https://img.shields.io/github/stars/MPh-py/MPh （611）；https://img.shields.io/github/stars/wjc9011/COMSOL_Multiphysics_MCP （981）；https://img.shields.io/github/last-commit/MPh-py/MPh （august）；https://img.shields.io/github/last-commit/wjc9011/COMSOL_Multiphysics_MCP （october）；https://commits.ecosyste.ms/api/v1/hosts/GitHub/repositories/<enc>/commits/ 逐月探针
- 本机探测：`HKLM\SOFTWARE\COMSOL\COMSOL64`（COMSOLROOT=D:\XiGPrograms\comsol\6.4\base）；Uninstall 键 COMSOL Multiphysics 6.4（6.4.0.293，20260926）

## 附：逐月提交证据（ecosyste.ms presence 探针，2026-11-17 跑，`_scripts/d03_month.mjs`）

```json
{"repo":"MPh-py/MPh","monthly":{"2026-05":1,"2026-06":0,"2026-07":1,"2026-08":1,"2026-09":0,"2026-10":0,"2026-11":0},"present":["2026-05","2026-07","2026-08"],"missing":["2026-06","2026-09","2026-10","2026-11"],"all_months":false}
{"repo":"wjc9011/COMSOL_Multiphysics_MCP","monthly":{"2026-05":1,"2026-06":1,"2026-07":1,"2026-08":1,"2026-09":1,"2026-10":1,"2026-11":0},"present":["2026-05","2026-06","2026-07","2026-08","2026-09","2026-10"],"missing":["2026-11"],"all_months":false}
```

（注：presence 探针按月判"有无提交"而非计数；2026-11 为**进行中月份**，截至 11-17。mph 版本节奏佐证"发版驱动"：1.3.0@2025-10-30 → 1.3.1@2025-11-07 → 1.3.2@2026-08-16 → 1.4.0@2026-08-30。）

## 用户裁决记录（2026-11-17）
- **COMSOL-MCP 豁免引入**（pin commit + ADR）；手册知识管线=**COMSOL-MCP pdf_search RAG**（零额外自研，手册 PDF 喂入）。
- sidecar 消费端**薄化 ~200 行入②域**（衔接③ truthful colorbar）；domain-check **留 ~100 行机械检查**。
- 默认确认：imgcheck 砍、品牌剥离走原生 logo*d=off、pyvista 渲染归③、④域 cmt 验证保形态。
