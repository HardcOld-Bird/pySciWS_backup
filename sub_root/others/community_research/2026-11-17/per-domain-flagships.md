# 分能力域旗舰社区项目速览（社区优先调研 · B 计划素材）

- 调研日期：2026-11-17（Asia/Shanghai）
- 覆盖：①文献调研 ②COMSOL/FEM ③科研绘图 ④理论计算 ⑤办公文档写作 ⑥3D 建模 ⑦AI 生图，每域 4 个旗舰候选 + 未入选候选简注
- 性质：速览，不求定论，求数据可追溯

## 0. 判定口径与数据源

### 0.1 PySci 引入判定口径（三口径）
1. **口径1**：GitHub star ≥ 100（衡量社区规模/项目成熟度）
2. **口径2**：近 6 个月（2026-05 ~ 2026-11）**每月均有提交**（衡量持续维护）
3. **口径3**：能减少自研代码量（社区优先与薄胶水原则）

三条全满足 = **可直接引入**；不全满足 = 标注"差口径N"并说明。

### 0.2 数据源与方法（均为 2026-11-17 采集）
| 指标 | 数据源 | 说明 |
|---|---|---|
| star / 许可 / 最后推送 | GitHub REST `api.github.com/repos/{o}/{r}`、GitHub Search API、ecosyste.ms `repos.ecosyste.ms/api/v1/hosts/GitHub/repositories/{o%2Fr}` | ecosyste.ms 带最近同步日期；paper-qa 等以 GitHub API 实时值为准 |
| 近 6 个月逐月提交 | ① `api.github.com/repos/{o}/{r}/stats/commit_activity`（52 周逐日，**默认分支**口径）② GitHub HTML `commits/{branch}?since=2026-MM-01&until=2026-MM-DD` 逐月探测（页面最多列约 35 条，故只作"有/无"） | 两法在 paper-qa、python-docx 上交叉验证一致 |
| 最近发版 | ecosyste.ms releases、GitHub `/releases` 页、PyPI JSON | 三者取可得者 |
| Py3.13 + Windows wheel | PyPI JSON 逐版本扫描：`cp313`/`abi3`/`win_amd64`/`py3-none-any` wheel 标签 + `requires_python` | "uv add 可装"= 存在兼容 wheel 或纯 Python wheel/sdist 依赖链可解析 |
| Agent 可驱动性 | 各仓库 README / 官方文档原文（raw.githubusercontent.com、readthedocs） | |

### 0.3 已知限制（未能核实事项）
- 本会话 `web_fetch` 工具不可用、PowerShell schannel TLS 被沙箱限制，全部网络抓取改由 Node.js `fetch` 完成。
- `api.github.com` 匿名限流 60 次/时，配额一度耗尽；缺口改用 ecosyste.ms、GitHub HTML 逐月探测补齐。
- **openpyxl 主仓不在 GitHub**（在 Heptapod `foss.heptapod.net/openpyxl/openpyxl`），站点被 Anubis 反爬拦截 → **其提交活跃度未能核实**；GitHub 仅有旧镜像（ericgazoni/openpyxl ★203 但 2014-03 后无提交）。
- 月度提交 = 默认分支口径；`pushed_at` 含所有分支，二者偶有出入（paper-qa、python-docx 已标注）。
- GMsh、GROBID 的 star 取自 GitHub Search API（2026-11-17）。
- "近6个月"含进行中的 2026-11（截至 11-17）；月末前无提交者按口径2如实记"缺"，标注"临界"。

---

## ① 文献调研（PDF 解析 / 检索问答 / 文献库管理）

| 候选（仓库） | ★ | 最近发版 | 最后提交 | 月度提交 05→11 | 许可证 | Py3.13+Windows `uv add` | LLM agent 驱动 | 能力边界（一句话） | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| **paper-qa**（Future-House/paper-qa） | 9,610 | v2026.08.12（2026-08-12） | 2026-11-09（非默认分支；默认分支 10-25） | ✗ 缺 05/07/09/11，窗口内仅 6 提交 | Apache-2.0 | ✓ 纯 wheel（`paper-qa` 2026.8.12，requires≥3.11）；依赖树重、需配 LLM key | CLI `pqa` + Python API（README 未见内置 MCP） | 对本地文献库做"带引用的 RAG 问答"，不负责解析 PDF 原文、也不做文献库管理 | **差口径2**（口径1、3 满足）：默认分支 5、7、9、11 月零提交 |
| **GROBID**（grobidOrg/grobid，原 kermitt2/grobid） | 5,232 | 0.9.1（2026-08-04） | 2026-11-16 | ✓ 全（143 提交） | Apache-2.0 | ✗ 无 wheel：Java 17+ 服务，需 Docker/gradle/二进制 | REST API + `grobid-client-python`（pip，★417） | 把学术 PDF 抽成 TEI 结构化引用/正文，不做全文 Markdown、不做检索问答 | **3/3 可直接引入**（注明外部 Java/Docker 运行时） |
| **marker**（datalab-to/marker） | 40,903 | v2.1.0（2026-10-29） | 2026-11-13 | ✓ 全（71） | Apache-2.0 | ✓ 纯 wheel `marker-pdf` 2.2.0（2026-11-12，requires≥3.10）；torch 等重依赖，CPU 模式可无 GPU 跑 | CLI（`--mode fast/balanced` 等）+ Python API | 高精度 PDF→Markdown/JSON 转换，不检索、不问答 | **3/3 可直接引入** |
| **zotero-mcp**（54yyyu/zotero-mcp，PyPI 名 `zotero-mcp-server`） | 5,916 | v0.15.2（2026-11-01）；PyPI 0.17.2（2026-11-15） | 2026-11-13 | ✓ 全（逐月≥1） | MIT | ✓ 纯 wheel（requires≥3.10）；需本机 Zotero 桌面端 + API key | MCP server（stdio/SSE，进官方 MCP registry） | 让 agent 读写 Zotero 文献库，不做 PDF 解析与问答 | **3/3 可直接引入**（注明外部 Zotero） |

**未入选候选简注**（未列全表指标）：
- **MinerU**（opendatalab/MinerU）★83,482，最后推送 2026-11-11，MinerU 4.1.x 于 2026-11-03 密集发版；许可为 **Apache-2.0+附加商业条款**（非纯净 OSS）；`mineru` pip 纯 wheel 4.2.3（2026-11-16，requires≥3.11，Py3.13 可装）；与 marker 同赛道（PDF→Markdown），可作 marker 备选。
- **docling**（docling-project）PyPI 2.150.1（2026-11-14）活跃，IBM 系 PDF→Markdown；`scholarly` PyPI 1.7.11 自 2023-01 未发版（陈旧）；`semanticscholar` PyPI 0.12.0（2026-03，MIT，社区封装）；OpenAlex 为 REST API（PyPI `openalex` 0.1.0 停在 2024-12）；connectedpapers 为免费 Web 服务、源码未完整开放，均未达"旗舰可引入"核实门槛。

**来源 URL**：
- https://github.com/Future-House/paper-qa ｜ https://api.github.com/repos/Future-House/paper-qa ｜ https://api.github.com/repos/Future-House/paper-qa/stats/commit_activity ｜ https://pypi.org/project/paper-qa/ ｜ https://github.com/Future-House/paper-qa/releases/tag/v2026.08.12
- https://github.com/grobidOrg/grobid ｜ https://github.com/grobidOrg/grobid/releases ｜ https://raw.githubusercontent.com/grobidOrg/grobid/master/LICENSE ｜ https://github.com/grobidOrg/grobid-client-python
- https://github.com/datalab-to/marker ｜ https://github.com/datalab-to/marker/releases/tag/v2.1.0 ｜ https://pypi.org/project/marker-pdf/ ｜ https://raw.githubusercontent.com/datalab-to/marker/master/README.md
- https://github.com/54yyyu/zotero-mcp ｜ https://pypi.org/project/zotero-mcp-server/ ｜ https://raw.githubusercontent.com/54yyyu/zotero-mcp/main/README.md
- 月度探测示例：https://github.com/Future-House/paper-qa/commits?since=2026-05-01&until=2026-05-31
- 其他：https://github.com/opendatalab/MinerU ｜ https://pypi.org/project/mineru/ ｜ https://pypi.org/project/docling/ ｜ https://pypi.org/project/scholarly/ ｜ https://api.openalex.org/

---

## ② COMSOL / 有限元仿真

| 候选（仓库） | ★ | 最近发版 | 最后提交 | 月度提交 05→11 | 许可证 | Py3.13+Windows `uv add` | LLM agent 驱动 | 能力边界（一句话） | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| **mph**（MPh-py/MPh，PyPI 名 `mph`） | 609 | v1.4.0（2026-08-30） | 2026-08-30 | ✗ 缺 06/07/09/10/11（仅 05、08 有提交） | MIT | ✓ 纯 wheel 1.4.0（requires≥3.10）；**但必须有 COMSOL+LiveLink 才能真正运行** | Python API（JPype Java 桥调 COMSOL API），可脚本化建模/求解/取结果 | 用 Python 驱动已安装的 COMSOL，自带求解器仍是 COMSOL；不替代 COMSOL、无 GUI | **差口径2**（口径1、3 满足）：发版式开发，近 6 个月 5 个月零提交 |
| **COMSOL_Multiphysics_MCP**（wjc9011/COMSOL_Multiphysics_MCP） | 944 | 无 release | 2026-10-31 | ✗ 缺 11（临界：10-31 后无提交） | MIT | 无 PyPI 包（`comsol-mcp`/`comsol-multiphysics-mcp` 均 404），需从 GitHub 源码装；**且需 COMSOL** | MCP server（README 明示对接 Claude、opencode 等 agent；有 Neurocomputing 2026 论文） | 为 agent 提供 COMSOL 多物理仿真的 MCP 工具面，能力≈mph 之上的协议封装 | **差口径2**（临界：2026-11 暂无提交；1/3 满足） |
| **FEniCSx / DOLFINx**（FEniCS/dolfinx） | 1,250 | v0.11.0.post0（2026-06-10） | 2026-11-13 | ✓ 全（626） | LGPL-3.0 | ✗ 无 PyPI 包（`fenics-dolfinx` 404）：conda / Docker / Windows-vcpkg 编译 | Python API（脚本驱动求解，无官方 CLI/MCP 封装） | 开源通用 FEM 求解器（C++ 核 + Python 绑定），不驱动 COMSOL、不含 CAD 前处理器 | **3/3 可直接引入**（注明安装非 pip、栈较重） |
| **FEATool Multiphysics**（precise-simulation/featool-multiphysics） | 462 | 1.18.0（2026-02-03） | 2026-09-03 | ✗ 仅 07(1)、09(7)，缺 05/06/08/10/11 | 专有商业（ecosyste.ms 标 other） | ✗ 主栈为 MATLAB/Octave（语言构成 MATLAB），非 pip 引入 | GUI + MATLAB 脚本，agent 接口弱 | 一体化商业 FEA 工具（可导出 FEniCS/OpenFOAM），Python 亲和性差 | **差口径2、口径3存疑**：月度稀疏 + 商业许可 + MATLAB 栈 |

**未入选候选简注**：scikit-fem、Nutils 等"纯 pip 可装"的轻量 FEM 本轮未列指标（下轮可补，用于与 FEniCSx 的"安装重量"对比）；COMSOL 自带 Methods/Java API 亦可编程，属自研范畴不计社区引入。

**来源 URL**：
- https://github.com/MPh-py/MPh ｜ https://pypi.org/project/mph/ ｜ https://mph.readthedocs.io/ ｜ https://api.github.com/repos/MPh-py/MPh/stats/commit_activity（配额期改用 HTML：https://github.com/MPh-py/MPh/commits/main?since=2026-05-01&until=2026-05-31）
- https://github.com/wjc9011/COMSOL_Multiphysics_MCP ｜ https://raw.githubusercontent.com/wjc9011/COMSOL_Multiphysics_MCP/main/README.md ｜ https://doi.org/10.1016/j.neucom.2026.134481
- https://github.com/FEniCS/dolfinx ｜ https://github.com/FEniCS/dolfinx/releases/tag/v0.11.0.post0 ｜ https://pypi.org/project/fenics-dolfinx/（404，佐证无官方 wheel）
- https://github.com/precise-simulation/featool-multiphysics ｜ https://repos.ecosyste.ms/hosts/GitHub/repositories/precise-simulation%2Ffeatool-multiphysics

---

## ③ 科研绘图

| 候选（仓库） | ★ | 最近发版 | 最后提交 | 月度提交 05→11 | 许可证 | Py3.13+Windows `uv add` | LLM agent 驱动 | 能力边界（一句话） | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| **matplotlib**（matplotlib/matplotlib） | 23,456 | v3.11.2（2026-09-11） | 2026-11-14 | ✓ 全（1311） | matplotlib 许可（PSF 衍生；PyPI 分类 PSF） | ✓ `cp313-cp313-win_amd64` wheel（3.11.2） | Python API（savefig 出图；无官方 MCP） | 2D 静态出版级绘图的事实标准；不做交互仪表盘、不做 3D 重型渲染 | **3/3 可直接引入** |
| **plotly**（plotly/plotly.py） | 18,899 | v7.1.0（2026-09-15） | 2026-11-13 | ✓ 全（355） | MIT | ✓ 纯 wheel 7.1.0（静态导出另需 kaleido） | Python API + HTML/JS 交互图，浏览器即界面 | 交互式 Web 图（html/png 导出），统计图语法弱于 seaborn/plotnine | **3/3 可直接引入** |
| **seaborn**（mwaskom/seaborn） | 14,111 | v0.13.2（2024-01-25） | 2026-10-08 | ✗ 缺 05/06/08/09/11（窗口仅 16） | BSD-3-Clause | ✓ 纯 wheel（requires≥3.8，py313 classifier ✓） | Python API（matplotlib 之上的统计图封装） | 统计关系可视化（箱线/热图/回归），不承担通用绘图 | **差口径2**：口径1、3 满足，但近 6 个月 5 个月零提交、2024-01 起未发版 |
| **plotnine**（has2k1/plotnine） | 4,783 | v0.15.8（2026-08-14）；PyPI 0.15.9（2026-11-03） | 2026-10-29（默认分支） | ✗ 缺 11（临界，10-29 后无提交） | MIT | ✓ 纯 wheel 0.15.9（requires≥3.10） | Python API（ggplot2 式声明语法） | R 语法系"图形语法"实现，补齐 matplotlib 声明式短板；不替代交互图 | **差口径2（临界）**：口径1、3 满足，11 月尚无提交 |

**未入选候选简注**：
- **pyqtgraph**（pyqtgraph/pyqtgraph）★4,444，MIT，0.14.0（2025-11-16），最后推送 2026-11-09，**月度全 ✓**（HTML 探测逐月≥1），纯 wheel（需 PyQt/PySide 外部绑定）——实时/交互大数据绘图的首选补充。
- **veusz**（veusz/veusz）★1,061，GPL-2.0，4.2.1（2026-04-04），最后推送 2026-08-22，**月度 ✗ 缺 07/09/10/11**；PyPI 仅 sdist + 需 PyQt5；有 `--export` 无头批处理——GUI 流程式绘图，口径2 不达标。

**来源 URL**：
- https://github.com/matplotlib/matplotlib ｜ https://github.com/matplotlib/matplotlib/releases/tag/v3.11.2 ｜ https://pypi.org/project/matplotlib/
- https://github.com/plotly/plotly.py ｜ https://github.com/plotly/plotly.py/releases/tag/v7.1.0 ｜ https://pypi.org/project/plotly/
- https://github.com/mwaskom/seaborn ｜ https://github.com/mwaskom/seaborn/releases/tag/v0.13.2 ｜ https://pypi.org/project/seaborn/
- https://github.com/has2k1/plotnine ｜ https://pypi.org/project/plotnine/ ｜ https://github.com/has2k1/plotnine/commits/main?since=2026-11-01&until=2026-11-30
- https://github.com/pyqtgraph/pyqtgraph ｜ https://github.com/veusz/veusz

---

## ④ 理论计算（符号 / 数值数学）

| 候选（仓库） | ★ | 最近发版 | 最后提交 | 月度提交 05→11 | 许可证 | Py3.13+Windows `uv add` | LLM agent 驱动 | 能力边界（一句话） | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| **SymPy**（sympy/sympy） | 15,087 | 1.14.0（2025-04-27） | 2026-11-14 | ✓ 全（1577） | BSD（ecosyste.ms 标 other，PyPI 分类 BSD） | ✓ 纯 wheel 1.14.0（requires≥3.9，py313 classifier ✓） | Python API（+ isympy 交互） | 纯 Python 符号计算（化简/微积分/方程），数值大矩阵与 ODE 刚性求解非其强项 | **3/3 可直接引入**（备注：发版节奏慢于提交节奏，2025-04 后未发版但仓库月均 120–310 提交） |
| **SciPy**（scipy/scipy） | 15,202 | v1.18.1（2026-08-21） | 2026-11-10 | ✓ 全（1262） | BSD-3-Clause | ✓ `cp313-cp313-win_amd64`（1.18.1） | Python API（scipy.optimize/integrate/sparse/…） | 数值算法全家桶（优化/积分/稀疏/信号），不做符号推导 | **3/3 可直接引入** |
| **JAX**（jax-ml/jax） | 36,616 | jax-v0.12.0.1（2026-10-22） | 2026-11-12 | ✓ 全（4095） | Apache-2.0 | ✓ `jax` 纯 wheel + `jaxlib` `cp313-cp313-win_amd64`（0.12.0.1）；GPU/CUDA wheel 以 Linux 为主，Windows 常走 CPU | Python API（jit/grad/vmap，可编程性极强） | 自动微分+JIT 的可组合数值计算/AD 框架，偏"可微编程"而非传统数学工具箱 | **3/3 可直接引入**（GPU 场景需注明平台限制） |
| **SageMath**（sagemath/sage） | 2,627 | 11.0.beta3（tag，2026-10-18；稳定线年度发版） | 2026-11-09 | ✓ 全（2343） | GPL-3.0+（ecosyste.ms 标 other） | ✗ 无 PyPI 包：conda / WSL / Docker / 源码编译 | CLI `sage` + Python（可被脚本调用） | 聚合 200+ 包的大型 CAS（数论/代数/组合），覆盖 SymPy+SciPy 交集但安装体量巨大 | **3/3 可直接引入**（注明非 pip 安装、体量大；按极简原则需权衡） |

**未入选候选简注**：**GNU Octave**（gnu-octave/octave，GitHub 镜像）★677、GPL-3.0、release-11.3.0（2026-07-10）、最后推送 2026-11-06（月度未单独统计）；需外部安装器，MATLAB 语法兼容层定位——在"Python 3.13 + uv"栈里意义有限，且 star 贴线。

**来源 URL**：
- https://github.com/sympy/sympy ｜ https://github.com/sympy/sympy/releases/tag/1.14.0 ｜ https://pypi.org/project/sympy/ ｜ https://api.github.com/repos/sympy/sympy/stats/commit_activity
- https://github.com/scipy/scipy ｜ https://github.com/scipy/scipy/releases/tag/v1.18.1 ｜ https://pypi.org/project/scipy/
- https://github.com/jax-ml/jax ｜ https://github.com/jax-ml/jax/releases/tag/jax-v0.12.0.1 ｜ https://pypi.org/project/jax/ ｜ https://pypi.org/project/jaxlib/
- https://github.com/sagemath/sage ｜ https://github.com/sagemath/sage/releases ｜ https://www.sagemath.org/
- https://github.com/gnu-octave/octave ｜ https://www.octave.org/

---

## ⑤ 办公文档写作（Word/PPT/Excel/LaTeX/论文）

| 候选（仓库） | ★ | 最近发版 | 最后提交 | 月度提交 05→11 | 许可证 | Py3.13+Windows `uv add` | LLM agent 驱动 | 能力边界（一句话） | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| **pandoc**（jgm/pandoc） | 46,941 | 3.13（2026-10-26） | 2026-11-15 | ✓ 全（逐月≥1：16/10/8/6/5/6/5） | GPL-2.0 | ⚠ 无官方 PyPI 包（PyPI `pandoc` 2.4 为陈旧第三方）；走 winget/GitHub exe，或 `pypandoc_binary`（PyPI 1.17，2026-03，捆绑二进制 wheel） | CLI（stdin/stdout、`--from/--to`），agent 最友好的文档转换枢纽 | 30+ 格式互转（docx/odt/html/latex/pptx/…），不负责复杂版式编辑与 Excel 计算 | **3/3 可直接引入**（注明二进制获取途径） |
| **python-docx**（python-openxml/python-docx） | 5,760 | PyPI 1.2.0（2025-06-16）；无 GH release | 默认分支最后提交 2025-06-16（pushed_at 2026-08-01 属其他分支） | ✗ 全 0（HTML+stats 双验证） | MIT | ✓ 纯 wheel 1.2.0（requires≥3.9） | Python API（读改写 docx） | 程序化创建/修改 Word 文档，不渲染、不保真导出 PDF | **差口径2**：口径1、3 满足，近 6 个月默认分支零提交、17 个月未发版 |
| **openpyxl**（主仓在 Heptapod） | GitHub 无达标主仓镜像（最高镜像 ericgazoni/openpyxl ★203 但 2014 后停更） | PyPI 3.1.5（2024-06-28） | 未能核实（Heptapod 被 Anubis 反爬拦截） | 未能核实 | MIT（PyPI） | ✓ 纯 wheel 3.1.5（requires≥3.8，Py3.13 实测可装） | Python API（xlsx 读写/样式/公式） | Excel xlsx 读写事实标准，不做图表渲染与公式计算 | **差口径1+口径2（未核实/存疑）**：GitHub star 口径不适用，Heptapod 提交节奏未核实、PyPI 发版停在 2024-06 |
| **python-pptx**（scanny/python-pptx） | 3,588 | PyPI 1.0.2（2024-08-07） | 2024-08-07 | ✗ 全 0（27 个月无提交） | MIT | ✓ 纯 wheel 1.0.2（requires≥3.8，py313 classifier ✓） | Python API（生成/解析 pptx） | 程序化生成与解析 PowerPoint，不渲染、无模板引擎 | **差口径2**：口径1、3 满足，长期无维护（功能稳定但生态停滞） |

**未入选候选简注**：
- **Typst**（typst/typst）★57,215、Apache-2.0、v0.15.1（2026-07-17）、最后推送 2026-11-06；PyPI `typst` 0.15.2（2026-11-05，`cp39-abi3-win_amd64` wheel → Py3.13 ✓）——LaTeX 替代的现代排版系统，CLI+库双形态，三条口径大概率满足（月度未逐月统计，推送活跃），**强烈建议下轮列为主候选**。
- **Quarto**（quarto-dev/quarto-cli）★6,183、MIT（PyPI 分类）、v1.11.7（2026-10-27）；PyPI `quarto-cli` 1.11.9（2026-11-02，sdist）——论文/报告/Jupyter 发布系统（内嵌 Pandoc），CLI 驱动，需外部运行时。
- **docxtpl**（elapouya/python-docx-template）★2,730、LGPL-2.1、最后推送 2026-07-07（08–11 月无提交 → 差口径2）、PyPI `docxtpl` 0.20.2（2025-11-13，纯 wheel）——Jinja2 模板填 Word，是 python-docx 的上层薄封装。
- **LibreOffice headless**：外部安装，`soffice --convert-to`/UNO 桥做格式转换与 PDF 导出（非 pip，未列指标）。

**来源 URL**：
- https://github.com/jgm/pandoc ｜ https://github.com/jgm/pandoc/releases/tag/3.13 ｜ https://pypi.org/project/pypandoc_binary/ ｜ https://pypi.org/project/pandoc/（2.4，2024-08，陈旧第三方）
- https://github.com/python-openxml/python-docx ｜ https://pypi.org/project/python-docx/ ｜ https://github.com/python-openxml/python-docx/commits（HTML 显示最后提交 2025-06-16）
- https://foss.heptapod.net/openpyxl/openpyxl ｜ https://pypi.org/project/openpyxl/ ｜ https://github.com/ericgazoni/openpyxl（旧镜像）
- https://github.com/scanny/python-pptx ｜ https://pypi.org/project/python-pptx/
- https://github.com/typst/typst ｜ https://pypi.org/project/typst/ ｜ https://github.com/quarto-dev/quarto-cli ｜ https://pypi.org/project/quarto-cli/ ｜ https://github.com/elapouya/python-docx-template

---

## ⑥ 3D 建模（CAD / 参数化 / 网格）

| 候选（仓库） | ★ | 最近发版 | 最后提交 | 月度提交 05→11 | 许可证 | Py3.13+Windows `uv add` | LLM agent 驱动 | 能力边界（一句话） | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| **FreeCAD**（FreeCAD/FreeCAD） | 35,392 | weekly-2026.11.11（周构建）；26.3rc2（2026-10-18） | 2026-11-12 | ✓ 全（2802） | LGPL-2.1 | ✗ 无官方 PyPI 包（`freecad` 404）：Windows 安装包/conda，**自带 Python 运行时**，不进 uv 环境 | `freecadcmd` 无头 CLI + 内置 Python API（可被脚本驱动） | 参数化实体 CAD + 草图/装配/工程图的开源旗舰，App 内脚本与无头运行两条路 | **3/3 可直接引入**（注明外部安装、双 Python 环境问题） |
| **build123d**（gumyr/build123d） | 3,686 | v0.13.0（2026-09-21） | 2026-11-10 | ✓ 全（514） | Apache-2.0 | ✓ 纯 wheel 0.13.0（requires≥3.11,<3.15）+ 二进制依赖 `cadquery-ocp` 8.0.1.1.0（**`cp313-cp313-win_amd64` ✓**，2026-10-05） | Python API（代码即模型；可导出 STEP/STL，jupyter 可视化） | 脚本化 B-rep CAD（OCCT 内核），无 GUI——参数化建模主力，不做网格剖分 | **3/3 可直接引入** |
| **gmsh**（live-clones/gmsh，官方 GitHub 镜像） | 135（贴线过） | GitHub 无 release；PyPI 4.15.2（2026-03-24） | 2026-11-16 | ✓ 全（逐月≥1） | GPL-2.0-or-later（PyPI；GitHub 标 NOASSERTION） | ✓ `py2.py3-none-win_amd64` wheel（SDK 内置，Py3.13 可装） | Python API（`import gmsh`）+ CLI（`gmsh xx.geo -2`） | 有限元网格生成器（CAD 内核 + 后端 20+），只剖网格不求解 | **3/3 可直接引入**（star 135 贴线；GPL 传染性需注意） |
| **trimesh**（mikedh/trimesh） | 3,719 | 5.2.0（2026-10-24）；PyPI 5.2.1（2026-11-12） | 2026-11-09 | ✓ 全（79） | MIT | ✓ 纯 wheel 5.2.1（requires≥3.10，py313 ✓；重型格式依赖可选 extras） | Python API + CLI（`python -m trimesh`） | 三角网格 I/O/布尔/凸包/碰撞等几何工具箱，不建 CAD 也不剖网格 | **3/3 可直接引入** |

**未入选候选简注**：
- **CadQuery**（CadQuery/cadquery）★6,105、PyPI 分类 Apache-2.0（ecosyste.ms 标 other）、v2.8.0（2026-06-20）、最后推送 2026-11-09（月度未逐月统计）、同一 OCP 依赖（cp313 win ✓）——build123d 的同源前辈，生态更大、语法偏 CAD 命令式。
- **OpenSCAD**（openscad/OpenSCAD）★10,638、最后 release 2021.01（**差口径2**：正式版多年未发，开发版有周更）、外部程序 + CLI，CSG 布尔建模，无 Python API（需外部调用）。
- **bpy（Blender as a Python module）**：PyPI `bpy` 5.3.0（2026-11-10）**requires_python `==3.13.*`**、`cp313-cp313-win_amd64` wheel ✓——恰好锁定 Py3.13，无头 Blender 全功能（网格/雕刻/渲染），GPL-3.0；适合 AI 生图/网格后处理联动，但体量大。

**来源 URL**：
- https://github.com/FreeCAD/FreeCAD ｜ https://github.com/FreeCAD/FreeCAD/releases ｜ https://pypi.org/project/freecad/（404 佐证）
- https://github.com/gumyr/build123d ｜ https://github.com/gumyr/build123d/releases/tag/v0.13.0 ｜ https://pypi.org/project/build123d/ ｜ https://pypi.org/project/cadquery-ocp/
- https://github.com/live-clones/gmsh ｜ https://gmsh.info/ ｜ https://pypi.org/project/gmsh/
- https://github.com/mikedh/trimesh ｜ https://pypi.org/project/trimesh/
- https://github.com/CadQuery/cadquery ｜ https://github.com/openscad/OpenSCAD ｜ https://pypi.org/project/bpy/

---

## ⑦ AI 生图（扩散模型图像生成）

| 候选（仓库） | ★ | 最近发版 | 最后提交 | 月度提交 05→11 | 许可证 | Py3.13+Windows `uv add` | LLM agent 驱动 | 能力边界（一句话） | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| **diffusers**（huggingface/diffusers） | 34,902 | v0.42.1（GH，2026-10-27）；PyPI 0.43.0（2026-11-12） | 2026-11-11 | ✓ 全（616） | Apache-2.0 | ✓ 纯 wheel 0.43.0（requires≥3.11）；torch/CUDA 需另装（Windows cp313 torch 已有） | Python API（pipeline 可编程组装）为主 | 库形态的扩散模型加载/训练/推理 SDK，无自带 UI 与工作流编排 | **3/3 可直接引入** |
| **ComfyUI**（Comfy-Org/ComfyUI，原 comfyanonymous 迁移） | 147,190 | v0.44.0（2026-11-04） | 2026-11-17（当天） | ✓ 全（逐月≥1） | GPL-3.0 | ⚠ 无官方 PyPI 包（PyPI `comfyui` 0.0.1 为占位第三方）：git clone + `pip install -r requirements.txt`（或 `pip install comfy-cli`）；需 GPU（CPU 可跑但慢） | HTTP/WebSocket API（`/prompt` 端点）+ comfy-cli，工作流 JSON 可由 agent 生成 | 节点图工作流引擎（生成+生态插件），GUI 与 headless API 双形态 | **3/3 可直接引入**（注明源码安装 + GPU 依赖） |
| **InvokeAI**（invoke-ai/InvokeAI） | 29,059 | v6.14.4（2026-11-03） | 2026-11-11 | ✓ 全（逐月≥1） | Apache-2.0 | ✗ **`invokeai` requires_python `>=3.11,<3.13` → 不支持 Python 3.13**（uv add 在 3.13 环境直接失败，须另开 3.11/3.12 venv） | CLI + REST API（headless 友好） | 面向创作者的托管式生图/生视频 UI+API，节点生态弱于 ComfyUI | **三条口径满足，但 Py3.13 兼容差**——按 PySci 环境属"不可直接引入"，需降解释放 |
| **AUTOMATIC1111 stable-diffusion-webui** | 165,311 | v1.10.1（2025-02-09） | 2026-03-02 | ✗ 全 0（2026-03-02 后窗口内无提交） | AGPL-3.0 | ⚠ 源码运行（venv，官方长期推荐 3.10.6）；窗口内无维护 | HTTP API（`--api` 参数） | 老牌 SD 一体化 WebUI，扩展生态大但开发已放缓 | **差口径2**：口径1、3 满足，近 6 个月零提交、2025-02 起未发版 |

**未入选候选简注**：**Fooocus**（lllyasviel/Fooocus）★53,928、GPL-3.0、最后推送 2025-12-01、最后发版 v2.5.5（2024-08-12）——"开箱即用 SDXL"但**实质停更**（差口径2），仅适合当冻结工具用，不适合作为长期依赖。

**来源 URL**：
- https://github.com/huggingface/diffusers ｜ https://github.com/huggingface/diffusers/releases ｜ https://pypi.org/project/diffusers/
- https://github.com/Comfy-Org/ComfyUI ｜ https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.44.0 ｜ https://raw.githubusercontent.com/Comfy-Org/ComfyUI/master/README.md ｜ https://pypi.org/project/comfyui/（占位包佐证）
- https://github.com/invoke-ai/InvokeAI ｜ https://pypi.org/project/invokeai/（requires_python `<3.13` 佐证）｜ https://github.com/invoke-ai/InvokeAI/releases/tag/v6.14.4
- https://github.com/AUTOMATIC1111/stable-diffusion-webui ｜ https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.10.1
- https://github.com/lllyasviel/Fooocus

---

## 8. 三口径判定汇总（仅列"存在缺口"者，全满足者见各节判定列）

| 候选 | 缺口 | 说明 |
|---|---|---|
| paper-qa（①） | 差口径2 | 默认分支 2026-05/07/09/11 零提交，窗口合计仅 6 提交 |
| mph / MPh-py（②） | 差口径2 | 发版驱动型开发：06/07/09/10/11 五个月零提交 |
| COMSOL_Multiphysics_MCP（②） | 差口径2（临界） | 2026-11-01~17 零提交（最后推送 10-31），11 月未过完 |
| FEATool（②） | 差口径2 + 口径3存疑 | 窗口仅 8 提交；专有商业许可、MATLAB 主栈 |
| seaborn（③） | 差口径2 | 近 6 个月 5 个月零提交，2024-01 起未发版 |
| plotnine（③） | 差口径2（临界） | 11 月零提交（最后提交 10-29，11-03 为打 tag） |
| python-docx（⑤） | 差口径2 | 默认分支最后提交 2025-06-16，窗口全 0 |
| openpyxl（⑤） | 差口径1（GitHub 口径不适用）+ 差口径2（未核实） | 主仓在 Heptapod；镜像最高 ★203 但 2014 停更；PyPI 发版 2024-06 |
| python-pptx（⑤） | 差口径2 | 2024-08 后无任何提交/发版 |
| OpenSCAD（⑥注） | 差口径2 | 正式 release 停在 2021.01 |
| InvokeAI（⑦） | 口径 1/2/3 均过，但 **Py3.13 不兼容**（requires<3.13） | 需 3.11/3.12 独立环境，违反"uv add 直装"目标 |
| AUTOMATIC1111（⑦） | 差口径2 | 2026-03-02 后零提交、2025-02 后未发版 |
| Fooocus（⑦注） | 差口径2 | 最后推送 2025-12-01、最后发版 2024-08，实质停更 |

**三口径全满足、可直接引入的头部候选**：GROBID、marker、zotero-mcp（①）｜ FEniCSx（②）｜ matplotlib、plotly（③）｜ SymPy、SciPy、JAX、SageMath（④）｜ pandoc（⑤）｜ FreeCAD、build123d、gmsh、trimesh（⑥）｜ diffusers、ComfyUI（⑦）。
（"可直接引入"仅指过三口径；个别项仍需外部运行时——COMSOL、Java/Docker、GPU、非 pip 安装——已在各节标注。）
