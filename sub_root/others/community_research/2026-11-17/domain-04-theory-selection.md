# 能力域④理论计算 · 选型细化调研报告（2026-11-17）

> 三段式第一段「选型」的调研部分。旧 pySciWS 仅作能力清单对照，**未复制任何代码**（本报告只含文字提炼）。
> 核实口径（用户裁决）：① star≥100；② 2026-05~2026-11 **每月均有提交**；③ 减少自研代码。
> 豁免规则：差口径②若"无可替代（唯一通道）"→豁免引入（pin 版本+原地 ADR 记风险）；有替代余地→拒绝。
> 技能形态为主（SKILL.md + 社区工具 + 薄胶水），不整机引入大平台（用户裁决③）。

## 0. 结论速览

- **核心四件套**：SymPy（符号推导+LaTeX 导出，三口径全过）+ SciPy（数值 ODE/PDE/积分/优化，全过）+ mpmath（高精度，全过）+ pint（量纲分析，全过）。NumPy 为既有基座，不占新维护面。
- **次选（按需，不入首发）**：JAX（三口径全过、jaxlib cp313-win 直装，但当前能力面不减自研——仅当出现梯度逆设计/大规模可微仿真再加装）。
- **拒绝**：SageMath（三口径过但"整机大平台"违反裁决③+非 pip+GPL，写明触发条件）、GNU Octave、Maxima、Mathematica（商业对照）、DifferentialEquations.jl（Julia 栈+②差 11 月，有替代）、EinsteinPy（②差 5 个月，有替代）、symengine.py（②差+当前版无 cp313-win）。均无需豁免（无"唯一通道"情形）。
- **薄胶水估计**：新自研 ≤250 行（LaTeX 导出微调 ~30、发散安全原语 ~120、EP/拓扑 recipe 与会话模板为文档）vs 旧项目 8 模块约 2,340 行。

---

## 1. 任务一：旧能力清单（仅对照，严禁复制代码）

来源：`pySciWS/orchestration/skills/theoretical-computation/`（SKILL.md + api.md + cmt-pipeline-api.md）、`src/pysci/skills/theoretical_computation/tools/`（8 模块 ≈2,340 行：cas 233 / config 150 / eigen 271 / numerical 291 / session 126 / theory(CLI) 312 / topology 419 / visualize 538）、`tests/skills/theoretical_computation/`（发散安全原语 7 测试）、`data/skills/theoretical_computation/`（3 篇物理配方）。

| # | 能力项 | 旧实现方式 | 新方案是否仍需保留 | 备注 |
|---|---|---|---|---|
| 1 | 符号建模（符号矩阵/传递矩阵链） | `cas.symbolic_matrix`、`transfer_matrix_chain` | **需要** | SymPy `sp.Matrix` 直建（零胶水）；传递矩阵链是物理模式→recipe |
| 2 | 化简/替换/级数展开 | `cas.simplify_expr(strategy=auto/expand)`、`series_expand`、`taylor_coefficients`、`substitute_*` | **需要** | SymPy 直调；strategy 路由与"simplify 大式超时"技巧→故障手册文档 |
| 3 | 结果→LaTeX 导出 | `cas.expr_to_latex(fold_short_frac)`、`matrix_to_latex(pmatrix)` | **需要（对接⑤域）** | sympy.printing.latex 内置；格式微调薄胶水 ~30 行 |
| 4 | 方程组/条件求解 | `cas.solve_system`、`solve_condition` | **需要** | `sympy.solve/linsolve` 直调，胶水 0 |
| 5 | 判别式/EP 条件 | `cas.discriminant_2x2`、`eigenvalue_splitting`、`eigen.ep_condition_symbolic` | **需要**（研究线非厄米物理） | 2×2 判别式=charpoly 系数，recipe 模式 ~20 行 |
| 6 | 参数空间+网格求值 | `numerical.ParamAxis/ParamSpace/make_meshgrid/evaluate_on_grid` | **需要** | `numpy.meshgrid`+`lambdify` 直调；ParamSpace 类封装**不保留**（旧项目特有） |
| 7 | lambdify 封装（分支切割/NaN） | `numerical.lambdify_expr` | 需要（文档） | `+0j` 复数技巧写入故障手册；封装不保留 |
| 8 | 奇点感知自适应 1D 采样 | `numerical.adaptive_sample_1d`（奇点邻域加密、不触奇点、升序去重、退化区间报错） | 可选 | 测试期望明确→薄胶水 ~60 行保留，或降为 recipe |
| 9 | 本征系统（符号/数值） | `eigen.eigensystem_symbolic/numeric(_2d)`、复平面轨迹 | **需要** | `numpy.linalg.eig`/SymPy 直调（零胶水） |
| 10 | 分支追踪（防交叉伪影） | `eigen.track_branches`、`sort_eigenvalues_real` | **需要** | 相位连续性排序是物理脚本模式→recipe；通用排序器不保留 |
| 11 | EP 检测（本征值+本征向量合并） | `eigen.detect_ep`、`eigenvector_coalescence` | **需要**（研究线） | recipe ~30 行判据（判别式过零+本征向量夹角） |
| 12 | 零点集/等值面提取 | `topology.find_zero_set/find_isosurface/find_multiple_isosurfaces`（pyvista 网格求交） | 需要（2D 部分） | 2D：`matplotlib contour`；3D 渲染移交③/⑥域，等值面提取可 skimage/pyvista 直调 |
| 13 | 奇点/临界点检测（Hessian/Morse 指标） | `topology.find_singularities`、`find_critical_points`、`_numerical_hessian` | 需要（可简化） | recipe（数值 Hessian ~30 行+Morse 分类） |
| 14 | 环绕数（拓扑不变量） | `topology.compute_winding_number(_2d)` | **需要**（研究线） | recipe（相位回路积分 ~20 行） |
| 15 | 探索性 2D 绘图 | `visualize.quick_plot_2d/complex/complex_plane`、`export_exploration` PNG | 需要（轻） | matplotlib 直调或 ~50 行胶水；出版图归③域 |
| 16 | 发散安全绘图 | `visualize.mask_divergent`、`quick_plot_divergent`（窗外置 NaN、复数取实部） | 可选 | 测试期望沿用→薄胶水 ~50 行 |
| 17 | 3D 离屏渲染（曲面/零点集/能带） | `visualize.quick_plot_3d_surface/zero_set_3d/band_structure_3d`（pyvista OFF_SCREEN） | **移交③/⑥域，不属④** | 旧项目特有边界，可不保留 |
| 18 | 计算会话/脚手架/持久化 | `session.ensure_session/save_results(.npz)/save_plot/write_log/list_sessions` + `theory new/run/plot/list` | 需要（形态） | 目录约定+脚手架模板承载（data 目录 + notes.md）；run/list 薄胶水或 `uv run` 直跑 |
| 19 | doctor 依赖自检 | `theory.cmd_doctor`（sympy/numpy/scipy/matplotlib/pyvista 版本） | 不保留 | 旧项目特有；pysci-infra `status` 覆盖 |
| 20 | CMT 前向/逆向/COMSOL 验证管线 | `research/example/theory` 三模块（S 矩阵前向核、DE 逆设计+checkpoint、mph 验证） | **旧研究线特有，不保留** | 蓝图：研究线代码只在本地保留；"前向核复用/断点续算"形态可写入 recipe |
| 21 | 物理配方库 | `data/…/recipes/`：EP-BIC 哈密顿量、CPA 传递矩阵、TBIC 能带 | 需要（形态） | 配方**文档**保留，内容按新 API（SymPy/SciPy 直调）重写，不含旧代码 |
| 22 | 高精度数值计算 | 无 | **新增** | mpmath（见 §2） |
| 23 | 量纲分析 | 无 | **新增** | pint（见 §2） |
| 24 | ODE/PDE/积分/优化 | 无显式模块（靠数值管线） | **新增显式** | scipy.integrate/optimize（solve_ivp/quad/…） |

测试反映的期望行为（`test_divergent_primitives.py`）：自适应采样不触碰奇点、邻域确实加密、升序去重、退化区间报错；`mask_divergent` 窗外与 NaN/Inf 置 NaN、复数取实部；`quick_plot_divergent` 返回 Figure——若保留薄胶水，这些行为期望原样作为验收标准（代码不迁移）。

**"旧项目特有、可不保留"汇总**：ParamSpace/ParamAxis 类封装、simplify strategy 路由器、sort_eigenvalues_real、3D pyvista 渲染管线（移交③/⑥）、doctor、config.py dotenv/路径锚点（pysci-infra 覆盖）、theory CLI 门面全集、CMT 研究线三模块。

---

## 2. 任务二：社区候选细化核实

### 2.1 核实方法与数据源

- 星级/最后提交/许可证：`img.shields.io/github/stars|last-commit|license/<owner>/<repo>`（SVG，2026-11-17 拉取）。
- **逐月提交（2026-05~2026-11）**：commits.ecosyste.ms 镜像 `commits?per_page=1&until=<月末>` 月末探针（返回≤该日期最后一条提交的 `timestamp`；落月内=当月有提交；与 shields last-commit 交叉一致）。本域全部候选 7 个月逐一探针，脚本 `_scripts/theory_probe.mjs`。
- Py3.13/Windows：`pypi.org/pypi/<pkg>/json`（requires_python、classifiers、**当前版本**是否有 cp313 win_amd64 / py3-none-any wheel）。
- 许可证以仓库 LICENSE 为准（pint、maxima 已拉原文核实）；shields 识别失败处以此为准。
- 弱点声明：月末探针证明"当月有提交"但不计提交数；bot 提交也算；calyau/maxima 是 SourceForge 主仓的 GitHub 镜像，月提交反映上游同步。

### 2.2 候选指标总表

| 候选（仓库） | 能力 | ★ | ①≥100 | ②逐月 5–11 | ③减自研 | Py3.13/Win | 驱动方式 | 许可证 | 能力边界/备注 |
|---|---|---|---|---|---|---|---|---|---|
| **sympy/sympy** | 符号推导/LaTeX | 15k | ✓ | **全✓**（05-30/06-29/07-30/08-29/09-29/10-30/11-15） | 强（化简/微积分/方程/张量/`latex()` 打印机全内置） | ✓ 1.14.0 纯 wheel（py3-none-any，requires≥3.9） | Python API | BSD | 纯 Python；数值大矩阵/刚性 ODE 非强项→交给 SciPy；`simplify` 大式慢（故障手册给 strategy 技巧） |
| **scipy/scipy** | 数值全家桶 | 15k | ✓ | **全✓**（…11-16） | 强（integrate/optimize/sparse/linalg/interpolate） | ✓ 1.18.1 `cp313-cp313-win_amd64`（requires≥3.12） | Python API | BSD-3-Clause | solve_ivp（BDF/Radau 刚性）、quad 系、minimize/DE、fftpack——覆盖 DifferentialEquations 生态的常规面 |
| **numpy/numpy** | 数值基座 | 33k | ✓ | **全✓**（…11-16） | 既有基座（旧项目已用） | ✓ 2.5.5 cp313 win | Python API | BSD-3-Clause | 不占新维护面；meshgrid/linalg.eig/NaN 掩码 |
| **mpmath/mpmath** | 高精度 | 1.2k | ✓ | **全✓**（05-28/06-29/07-27/08-29/09-28/10-30/11-14） | 强（任意精度浮点/特殊函数/求和求积） | ✓ 1.4.1 纯 wheel（requires≥3.9） | Python API | BSD-3-Clause | SymPy 底层高精度即用它；`mpmath.workdps` 切换精度 |
| **hgrecco/pint** | 量纲分析 | 2.8k | ✓ | **全✓**（05-17/06-10/07-30/08-05/09-18/10-29/11-15） | 强（Quantity/单位换算/量纲一致性检查） | ✓ 0.26.2 纯 wheel（requires≥3.12） | Python API | **BSD-3-Clause**（LICENSE 原文核实） | 与 NumPy 数组互操作好；不做符号量纲推导（SymPy 侧需自行标注） |
| jax-ml/jax | 自动微分/JIT | 37k | ✓ | **全✓**（…11-16） | △ 对当前能力面不减胶水（可微编程/梯度逆设计才用得上） | ✓ jax 0.12.0.1 纯 wheel + jaxlib `cp313-cp313-win_amd64` 0.12.0.1；Windows GPU 走 CPU 为主 | Python API（jit/grad/vmap） | Apache-2.0 | 次选：三口径全过可直接引入，但按极简原则**按需加装** |
| sagemath/sage | 大型 CAS 聚合 | 2.6k | ✓ | **全✓**（…11-14） | △ 覆盖 SymPy+SciPy 交集，但整机引入反而膨胀维护面 | ✗ 无 PyPI：conda/WSL/Docker/源码 | `sage` CLI + Python | GPL-3.0+ | **形态拒绝**（裁决③"不整机引入大平台"+非 pip+GPL）；触发条件见 §3 |
| gnu-octave/octave | MATLAB 兼容层 | 677 | ✓（贴线） | **全✓** | ✗ Python 栈内冗余（MATLAB 语法兼容才是其价值） | ✗ 外部安装器（非 pip） | CLI/外部进程 | GPL-3.0 | 拒绝：③不成立+安装面外挂 |
| calyau/maxima（SF 主仓镜像） | Lisp CAS | 151（镜像） | ✓（贴线） | **全✓**（镜像同步上游） | ✗ 能力被 SymPy 覆盖，反而新增 Lisp 外部安装面 | ✗ 外部安装器（非 pip） | CLI/外部进程 | GPL-2.0+ | 拒绝：有替代（SymPy）→按豁免规则逻辑"有替代余地者拒绝" |
| SciML/DifferentialEquations.jl | Julia ODE/PDE 生态 | 3.2k | ✓ | **5✓6✓7✓8✓9✓10✓ 11✗**（最后 10-17） | ✗ 引入=整个 Julia 语言栈 | ✗ Julia 包（非 pip） | Julia API | MIT | 拒绝：语言栈不符+②差 11 月+有替代（scipy.integrate 常规面）；其刚性/DAE/多尺度极限写入技能文档边界注记 |
| einsteinpy/einsteinpy | 相对论张量 | 707 | ✓ | **5–9✗**（2024-05 后静默两年，10/11 月复活） | 中 | ✓ 0.5.0 纯 wheel | Python API | MIT | 拒绝：②差 5 个月（bursty 复活期）+有替代（SymPy tensor/diffgeom）；若做广义相对论专研可复评 |
| symengine/symengine.py | 符号加速（C++ 核） | 189 | ✓ | **5✗7✗** | △ 性能加速非减自研 | ✗ 当前 0.15.0 无 cp313 win wheel（0.13 曾有） | Python API | MIT | 拒绝/暂缓：②差+③存疑+当前版无轮子 |
| Wolfram Mathematica | 商业 CAS 对照 | n/a | n/a | n/a | ✗ | n/a（商业安装器） | wolframclient-python 桥（需 license） | **专有商业** | 对照项：符号+LaTeX 打磨标杆，但商业许可+闭源→不引入；个人科研预算内无必要 |

### 2.3 三口径/豁免判定

**通过（三口径全过）→ 直接引入（核心）**
- **SymPy**：①15k✓ ②全✓ ③强✓ → **引入（符号主力 + LaTeX 导出源）**。
- **SciPy**：①15k✓ ②全✓ ③强✓ → **引入（数值主力：ODE/PDE/积分/优化）**。
- **mpmath**：①1.2k✓ ②全✓ ③强✓ → **引入（高精度）**。
- **pint**：①2.8k✓ ②全✓ ③强✓ → **引入（量纲分析）**。

**通过但按极简原则按需**
- **JAX**：三口径全过（①37k✓ ②全✓）且 jaxlib cp313-win 直装。但③对当前能力面不成立（不减少 SymPy/SciPy 之上的胶水），价值场景=梯度逆设计/灵敏度分析/大规模可微仿真（旧 CMT 逆设计用 DE 不需要梯度）→ **次选：触发即引入，首发不装**。若用户希望一步到位也可直接引入（无口径障碍）。

**形态/边界拒绝（非豁免问题）**
- **SageMath**：①②③字面全过，但"整机大平台"违反用户裁决③（技能形态为主）+ 非 pip（破坏 uv 项目内自包含安装面）+ GPL-3.0 → **拒绝，写明触发条件**（§3）。
- **Octave / Maxima**：①②过但③不成立（Python 栈内冗余/被 SymPy 覆盖）+外部安装面 → 拒绝。
- **Mathematica**：商业对照，拒绝。

**差口径② → 拒绝（有替代余地；无一构成"唯一通道"，故本域无需豁免）**
- **DifferentialEquations.jl**（11 月空）：能力上有"刚性/DAE/多尺度 ODE 更强"的独到之处，但替代余地存在——scipy.integrate 覆盖常规面、超纲需求可临时外跑 Julia 不入主栈；且整栈引入违反减自研 → 拒绝，边界注记入技能文档。
- **EinsteinPy**（5–9 月静默）：SymPy tensor/diffgeom 覆盖 → 拒绝。
- **symengine.py**（5/7 月空+无当前 cp313-win）：SymPy 直用可接受性能 → 拒绝/暂缓。

---

## 3. 建议组合、分工与薄胶水估计

### 3.1 组合分工（回答"SymPy+SciPy+mpmath 是否已够"）

**够，但补一件**：用户场景的四个面里——符号推导（SymPy）、数值计算（SciPy+NumPy）、高精度（mpmath）——三件套已覆盖；唯一缺口是**量纲分析**，pint 三口径全过且纯 wheel 轻量，直接进核心。故：

```
符号推导/LaTeX ── SymPy（sp.Matrix/solve/series/latex() 打印机）
数值计算       ── SciPy（integrate.solve_ivp/quad、optimize、linalg）+ NumPy 基座
高精度         ── mpmath（workdps；SymPy evalf 底层同源）
量纲分析       ── pint（Quantity 与 NumPy 互操作）
结果→LaTeX⑤   ── sympy.printing.latex + 薄胶水微调（fold_short_frac/pmatrix/复数显示）
                  对齐⑤域约定：MD 写 $...$（pandoc→OMML）；pptx 走 `pysci-doc formula` PNG；LaTeX 原生 amsmath
按需加装       ── JAX（梯度逆设计/JIT/灵敏度；触发即引入，无口径障碍）
```

**SageMath 何时才值得引入**（触发条件，满足其一再评估）：
1. 研究需求进入 SymPy 弱区：数论、代数几何、有限域、群表示（GAP 级）、组合设计、椭圆曲线；
2. 需要 Sage 统一对象系统/Notebook 做长代数推导协作；
3. 需要 Sage 独有算法（如某些模形式/守恒数计算）。
届时也**不进项目 venv**：以独立发行版（WSL/conda/Docker）或 sagecell 云按需使用，走"外部工具"形态（如④COMSOL 的定位），技能文档只记调用约定。

### 3.2 薄胶水估计（新自研面，合计 ≤250 行）

1. LaTeX 导出微调（`expr_to_latex`：fold_short_frac、pmatrix 环境、复数/分式显示对齐⑤域）≈ 30 行；
2. 发散安全原语（`mask_divergent` + 奇点感知 `adaptive_sample_1d` + `quick_plot_divergent`，沿用旧测试期望作验收）≈ 120 行 + 测试；
3. EP/拓扑 recipe 文档（EP 判据、branch tracking、环绕数、Morse 分类、传递矩阵链）：**0 行**（代码模式写入 `data/…/recipes/`，按新 API 重写）；
4. 会话约定（脚手架模板 + `results/plots/notes.md` 目录约定）：模板文件，无代码（若保留 `theory new/run` 薄门面再 +50 行，可选）。

对比旧项目：8 模块 ≈2,340 行 → ≤250 行胶水 + 纯文档；3D 渲染移交③/⑥域，CMT 研究线管线不入仓库。

---

## 4. 待用户裁决点

1. **JAX 时机**：首发不装、按需加装（推荐，极简），还是现在就收进依赖？
2. **发散安全原语/奇点感知采样**：保留为薄胶水（~120 行+测试），还是降级为 recipe 文档（零代码）？
3. **SageMath 触发条件**（§3.1 三条）认可与否；确认"外部工具形态、不进 venv"的处理。
4. **3D 探索渲染（pyvista）移交③/⑥域**：确认不属④维护面。
5. **物理配方库**：EP-BIC/CPA/TBIC 三篇以"新 API 配方文档"形态保留（不含旧代码），确认。
6. **theory CLI 门面去留**：纯 `uv run python <script>` + 脚手架模板（推荐），还是保留 `theory new/run/plot/list` 薄门面？
7. **scipy.integrate 边界注记**：Julia DifferentialEquations.jl 拒绝后，在技能文档写明"刚性/DAE/多尺度 ODE 超出 solve_ivp 时临时外跑 Julia 或 sundials 绑定、不入主栈"，认可？

---

## 5. 来源 URL 清单

- 星级/最后提交/许可证徽章：`https://img.shields.io/github/{stars,last-commit,license}/<owner>/<repo>`（12 候选逐一拉取，2026-11-17）
- 逐月提交探针：`https://commits.ecosyste.ms/api/v1/hosts/GitHub/repositories/<owner>%2F<repo>/commits?per_page=1&until=<月末>`（2026-05~2026-11 逐月；脚本 `_scripts/theory_probe.mjs`，与 shields last-commit 交叉一致）
- PyPI 版本/轮子：`https://pypi.org/pypi/{sympy,scipy,numpy,mpmath,Pint,jax,jaxlib,einsteinpy,symengine}/json`（2026-11-17）
- pint LICENSE（BSD-3 原文）：https://raw.githubusercontent.com/hgrecco/pint/master/LICENSE
- Maxima COPYING（GPL 文本）：https://raw.githubusercontent.com/calyau/maxima/master/COPYING（主仓为 SourceForge，GitHub 为镜像）
- 旧项目能力清单来源：`pySciWS/orchestration/skills/theoretical-computation/`、`pySciWS/src/pysci/skills/theoretical_computation/tools/`、`pySciWS/tests/skills/theoretical_computation/`、`pySciWS/data/skills/theoretical_computation/`（仅阅读提炼，未复制代码）
- 对照格式与交叉数据：`community_research/2026-11-17/domain-01-literature-selection.md`、`per-domain-flagships.md`（④段 SymPy/SciPy/JAX/SageMath 首轮数据与本轮复核一致）

## 用户裁决记录（2026-11-17）
- 核心四件套确认（SymPy/SciPy/mpmath/pint）；JAX 按需不首发；拒绝清单认可。
- 自研件取推荐档：发散安全原语留胶水（~120 行，7 项测试为验收）；3 篇配方→recipe 文档；theory CLI 砍为纯 uv run+模板；SageMath 触发时外部用不进 venv；3D 渲染移交③/⑥；scipy.integrate 边界注记认可。
