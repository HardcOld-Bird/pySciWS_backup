# ③绘图 + ④理论计算 实施计划（状态：草案，待用户确认）

> 三段式第二段产物。选型输入：`sub_root/others/community_research/2026-11-17/domain-03-plotting-selection.md`、
> `domain-04-theory-selection.md`（各附用户裁决记录）。完成后归档至 `archived/`。

## 1. 范围

**做**：③④两域"技能文档 + 社区依赖 + 薄胶水 + 机械断言"落地。
**不做**（按裁决）：标注语义四判据、零遮挡像素判据（规则文字入 SKILL.md）、COMSOL 栅格叠加（归②域）、
ParamSpace/strategy 路由器/theory CLI 全集/doctor、CMT 研究线管线（研究线不入库）；plotly/JAX/plotnine/
ParaView 首发不装（按需加装）。

## 2. 选型定稿

### ③绘图（用户修正案：出版链 ultraplot 路线）
| 角色 | 引入 | 版本策略 | 口径/豁免 |
|---|---|---|---|
| 出版写作面 | **ultraplot**（347★，MIT，月度发版，自带 MCP server 供 agent 查 API） | pin（注意 matplotlib<3.12 约束） | 三口径全过 |
| 底座 | matplotlib（3.11.x，随 ultraplot 约束对齐） | pin | 三口径全过 |
| 期刊样式 | **SciencePlots**（9.4k★，纯 mplstyle 资产） | **豁免 pin**（差②，零代码依赖维护面≈0，ADR 记录） | 豁免：无可替代样式积累 |
| 3D/场 | pyvista（3.9k★，vtk cp313-win） | pin | 三口径全过 |
| 交互（可选） | plotly+Kaleido（按需加装，不首发） | — | — |

### ④理论计算
| 角色 | 引入 | 口径 |
|---|---|---|
| 符号/数值/高精度/量纲 | SymPy + SciPy + mpmath + pint（全 pin；NumPy 既有基座） | 全过，无需豁免 |
| 结果→LaTeX | sympy 打印机 + ~30 行胶水（衔接⑤域） | 自研薄胶水 |
| 可选 | JAX（按需加装） | — |

## 3. 实施步骤

1. **依赖引入**：pyproject + uv sync（ultraplot、matplotlib、SciencePlots、pyvista、SymPy、SciPy、
   mpmath、pint，全 pin）；E 类同流程（update-deps status 校验）。
2. **技能文档**（`.agents/skills/`）：
   - `scientific-plotting/SKILL.md`：出版链工作流（ultraplot+SciencePlots 期刊样式/尺寸契约、
     EPS/PDF/SVG/PNG 导出、audit 轻量自检、预览闭环）、3D/场（pyvista）、按需件（plotly/ParaView）；
     期刊样式以 `styles/*.mplstyle` 资产固化家风。
   - `theory-calculation/SKILL.md`：符号建模→化简→求解→数值（SciPy 边界注记）→高精度→量纲→
     结果 LaTeX 导出；EP 判据/分支追踪/环绕数/Morse 配方文档（recipe）。
3. **薄胶水**（`src/pysci/skills/`）：
   - ③ ≈400-700 行：figures CLI 路由、导出封装（字体嵌入/fonttype=42/尺寸契约）、样式应用、
     面板标号、轻量 audit；样式资产 `styles/*.mplstyle`；
   - ④ ≤250 行：LaTeX 导出封装 ~30、发散安全原语 ~120（7 项测试为验收）、杂项 ~100。
4. **机械断言**：`tests/skills/scientific_plotting/`、`tests/skills/theory_calculation/`（固定样例；
   渲染/编译类标 hardware）；`tests.cmd all` 全绿。
5. **冒烟**：出版图真实产物（双栏尺寸/字体嵌入/公式标注/EPS+PDF）目检；符号计算→LaTeX→⑤域排版链
   衔接冒烟。

## 4. 风险与 ADR

- ultraplot API 演进快（2.x 重写史）→ pin 版本，升级前跑其自带 MCP 查 changelog；
- ultraplot 约束 matplotlib<3.12 → 依赖矩阵由 uv 解析，冲突时记 ADR；
- SciencePlots 差②豁免（内容资产定性）→ ADR 记录；
- pyvista/vtk 体积大 → 按需 import，不入最小路径。

## 5. 验收标准

③：APS/Nature 尺寸契约图真实产出（矢量+字体嵌入+公式标注+色盲安全循环），audit 轻量自检通过；
④：符号推导→数值→LaTeX 导出→⑤域 PDF 排版链全通，发散安全 7 测试过；
`tests.cmd all` 全绿；归档。

## 6. 实施结果（2026-11-17 收尾记录）

- 验收标准**全部达成**：③真实冒烟（APS 170mm 五件产物+FontFile2 嵌入+audit 全 PASS+VLM 目检通过）；
  ④模板直跑冒烟 + **④→⑤跨域链实测**（sympy→LaTeX→`pysci-doc latex build`→PDF ok）。
- 依赖全 pin 落位（ultraplot 2.7.5/matplotlib 3.11.2/SciencePlots 2.2.2 豁免/pyvista 0.49.2/
  sympy 1.14.0/scipy 1.18.1/mpmath 1.3.0（sympy 兼容降 pin）/pint 0.26.2），4 条 ADR 入 pyproject。
- ultraplot MCP 暂不挂（mcp 2.x 与项目 mcp<2 硬冲突，后备 uv tool 隔离路线记 ADR）。
- 薄胶水：③ 788 行+样式资产 61 行；④ 249 行。console script `pysci-figures` 已补注册。
- 沙盒 seam 薄解：`ensure_plotting_env()` 重定向 HOME/MPLCONFIGDIR（须 import 前调用）。
- 最终 `tests.cmd all`：ruff 全过 + 196 passed / 6 hardware（另单独验证全过）。
