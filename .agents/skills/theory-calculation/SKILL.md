---
name: theory-calculation
description: ④理论计算能力域操作面：符号建模/化简/求解（SymPy）→数值验证（SciPy/NumPy）→高精度（mpmath）→量纲检查（pint）→LaTeX 导出→⑤域排版的工作流、发散安全原语用法、EP/拓扑物理配方与 SciPy 边界注记。需要推导/化简公式、算本征值或色散、扫参数空间、做高精度数值、查量纲、把推导结果导成 LaTeX 排版时使用。
---

# Theory Calculation（④理论计算）

理论计算能力域的操作面：**社区四件套承载全部计算能力，只在接缝处用薄胶水**。
选型与用户裁决见 `sub_root/others/community_research/2026-11-17/domain-04-theory-selection.md`；
本技能放"怎么用"，配方/须知在 `references/`（按需加载）。薄胶水在
`sub_root/src/pysci/skills/theory_calculation/`（≤250 行，无 CLI）。

## 能力分层（谁负责什么）

| 能力 | 承载 | 用法 |
|---|---|---|
| 符号建模/化简/求解/级数 | **SymPy 1.14** | `sp.Matrix` / `sp.solve` / `sp.linsolve` / `sp.series` 直调（零胶水） |
| 数值计算（ODE/PDE/积分/优化/本征） | **SciPy 1.18 + NumPy 2.5** | `scipy.integrate/optimize/linalg`、`numpy.linalg.eig` 直调 |
| 高精度 | **mpmath 1.3** | `mpmath.workdps(dps)` 切精度；SymPy `evalf` 底层同源 |
| 量纲分析 | **pint 0.26** | `pint.Quantity` + 单位换算/一致性检查（符号量自行标注） |
| 结果→LaTeX | sympy 打印机 + **薄胶水** | `latex_export.expr_to_latex / matrix_to_latex`（输出归一化、确定性） |
| 发散安全 | **薄胶水** | `divergence_safety`：奇点感知采样/发散遮断/安全绘图（见须知） |
| 会话约定 | **薄胶水** | `session.ensure_session/save_results/append_note`（results/plots/notes.md） |
| LaTeX→PDF / 公式图 | ⑤文档写作域 | `pysci-doc latex build` / `pysci-doc formula`（本域不重复实现） |
| 出版图 / 3D 渲染 | ③科研绘图 / ⑥3D 建模 | 探索图止步于 matplotlib Figure；出版与渲染整体移交 |

## 工作流（七步）

1. **符号建模**：`sp.symbols`/`sp.Matrix` 建模（传递矩阵链、Bloch 哈密顿量等模式见配方）。
2. **化简/方程**：`sp.simplify`（大式超时→`expand`/跳过，见 troubleshooting）、`sp.solve`/`sp.linsolve`。
3. **数值验证**：`sp.lambdify`→NumPy 求值；本征走 `numpy.linalg.eig`；网格 `np.meshgrid`。
   发散量先 `divergence_safety.adaptive_sample_1d` 采样、`quick_plot_divergent` 目检。
4. **高精度**：关键数值 `mpmath.workdps(50)` 复核（SymPy 侧 `expr.evalf(50)`）。
5. **量纲检查**：pint `Quantity` 换算与量纲一致性核验（符号表达式自行标注单位）。
6. **LaTeX 导出**：`latex_export.expr_to_latex(expr)`（矩阵用 `matrix_to_latex`，pmatrix）。
7. **⑤域排版**：MD 写 `$...$`（pandoc→OMML）；LaTeX 贴 amsmath；pptx 走 `pysci-doc formula`；
   成稿编译 `pysci-doc latex build`。

## 调用形态（无 CLI，用户裁决 2026-11-17）

- **纯 `uv run python <脚本>` + 模板**：脚手架模板
  `sub_root/data/skills/theory_calculation/templates/theory_script.py`（复制即用，仓库内直跑=最小冒烟链）。
- **会话目录约定**：`data/research/<n>_<name>/theory/<slug>/{results,plots,notes.md}`，
  由 `session.ensure_session(name, slug)` 唯一创建（data/ 外写入自动报错）。
- 代码分工：研究线专有模型放 `src/pysci/research/<name>/theory/<slug>.py`；本技能只留通用胶水。

## 薄胶水 API（唯一自研面）

| 模块 | 函数 | 职责 |
|---|---|---|
| `latex_export` | `expr_to_latex(expr, *, fold_short_frac=True)` | 打印机输出归一化（确定性；短分式折叠 `x / y`） |
| `latex_export` | `matrix_to_latex(matrix, *, env="pmatrix")` | 矩阵→amsmath 矩阵环境 |
| `divergence_safety` | `adaptive_sample_1d(interval, *, singularities, base_resolution, approach_points)` | 奇点感知 1D 采样（不触奇点、邻域加密、升序去重） |
| `divergence_safety` | `mask_divergent(y, window)` | 发散遮断（窗外/NaN/Inf→NaN，复数取实部） |
| `divergence_safety` | `quick_plot_divergent(x, y, *, ylim, title, ax)` | 发散安全探索绘图（返回 Figure） |
| `session` | `ensure_session / save_results / append_note` | 会话骨架 / npz 落盘 / 时间线日志 |
| `inverse_design` | `run_inverse_design / CheckpointCallback / load_checkpoint` | 逆设计 scipy DE + 每代 checkpoint/续跑（见 recipe-inverse-design.md） |

行为合同与验收以旧项目 7 项发散安全测试期望为准绳（对照期望值重写，代码未迁移）；
使用须知见 [references/divergence-safety.md](references/divergence-safety.md)（含积分/级数发散
检测、收敛半径/围道检查模式与 `+0j` 分支切割技巧）。

## 配方与参考（按需加载）

- [references/recipe-ep-bic.md](references/recipe-ep-bic.md) — EP-BIC 哈密顿量：判别式 EP 条件、本征值扫描、复平面轨迹。
- [references/recipe-cpa-transfer-matrix.md](references/recipe-cpa-transfer-matrix.md) — CPA 传递矩阵链与零极点函数。
- [references/recipe-tbic-band.md](references/recipe-tbic-band.md) — TBIC 能带：Bloch 哈密顿量、网格求值、分支排序。
- [references/ep-topology-recipes.md](references/ep-topology-recipes.md) — EP 判据/分支追踪/环绕数/Morse 分类代码模式（文档配方，不做代码）。
- [references/recipe-cmt-ep-ndim.md](references/recipe-cmt-ep-ndim.md) — EP n×n 谱判别式：F̃=0 ∧ defective（排除 diabolic），2×2 判别式的 n×n 推广，含浮点鲁棒性注记。
- [references/recipe-inverse-design.md](references/recipe-inverse-design.md) — 逆设计协议：目标函数协议 + scipy DE + 每代 checkpoint（scipy 替自研 DE 裁决与行为差异注记）。
- [references/divergence-safety.md](references/divergence-safety.md) — 发散安全使用须知。
- [references/scipy-boundaries.md](references/scipy-boundaries.md) — scipy.integrate 边界注记（刚性/DAE/多尺度）。
- [references/troubleshooting.md](references/troubleshooting.md) — 故障手册（simplify 超时、lambdify NaN 等）。

## 边界与拒绝清单提醒

- **SageMath 不引入**（用户裁决）。触发条件（满足其一再评估）：研究进入 SymPy 弱区
  （数论/代数几何/有限域/群表示/组合设计/椭圆曲线）、需要 Sage 统一对象系统做长代数协作、
  需要 Sage 独有算法。届时走**外部工具形态**（WSL/conda/Docker/sagecell），不进项目 venv，
  如④COMSOL 的定位，仅记调用约定。
- **JAX 不首发**：梯度逆设计/灵敏度分析/大规模可微仿真出现时按需加装（无口径障碍）。
- **scipy.integrate 超纲**（DAE/多尺度/极端刚性）：临时外跑 Julia 或 sundials 绑定，不入主栈
  ——见 [references/scipy-boundaries.md](references/scipy-boundaries.md)。
- 3D 离屏渲染（pyvista）移交③/⑥域；ParamSpace 封装、simplify strategy 路由器、doctor、
  theory CLI 全集、CMT 研究线管线整体**不保留**（⭐ 研究线留本地不入仓；☆ 方法论拆为
  recipe/薄胶水吸收——EP n×n 判据、逆设计协议，用户裁决 2026-11-17）。

## 验收

1. 机械层：`sub_root/tests/tests.cmd all`（LaTeX 导出确定性 + 发散安全 7 项判定 + 会话骨架）。
2. 链路冒烟：模板脚本直跑（符号求导→数值→LaTeX→落盘）→ 结果贴⑤域 `pysci-doc latex build` 出 PDF。
