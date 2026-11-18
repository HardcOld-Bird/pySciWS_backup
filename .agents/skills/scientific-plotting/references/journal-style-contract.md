# 期刊样式契约（③绘图 reference 1/3）

> 何时读：选样式、定尺寸/字体、改色板、排查"图进期刊被排版拒"。数据真源：
> `sub_root/src/pysci/skills/scientific_plotting/styles/*.mplstyle` 与 `styles.JOURNAL_WIDTHS`；
> 本册是规则文字（Why + 自查口径）。

## 尺寸契约（audit 容差 ±2mm）

| 期刊族 | 单栏 | 双栏 | 字体主链（回退） | 正文字号 |
|---|---|---|---|---|
| APS（PRL/PR） | **85 mm** | **170 mm** | Arial → Helvetica → DejaVu Sans | 8pt |
| Nature | **89 mm** | **183 mm** | Helvetica → Arial → DejaVu Sans | 7pt |

- 宽度由 `export.fit_figure_size(fig, journal=..., span=...)` 施加到 figsize；
  高度自定（内容驱动），但保持合理长宽比（单栏图高一般 ≤ 100mm）。
- **audit 以成品实测宽度为准**（±2mm）。`savefig.bbox="tight"` 会裁切到内容包围盒，
  成品宽度 < 名义 figsize——不得拿名义值冒充实测（假绿防护）。

## 可编辑矢量（fonttype 契约，硬性）

| 键 | 值 | 效果 |
|---|---|---|
| `pdf.fonttype` | `42` | TrueType 软字体嵌入（PDF 可在 Illustrator/Inkscape 改字） |
| `ps.fonttype` | `42` | EPS 同上（Type 42 `/sfnts`；Type3 位图字体 = 硬伤） |
| `svg.fonttype` | `none` | 保留真实 `<text>` 节点（不转路径），末修可编辑 |

导出层（`export.export_figure`）保存期间**强制**这三键（rc_context），即使图脚本
忘了铺契约层样式也不会交出不可编辑的图。验收判定：PDF 有 `/FontFile2`、EPS 有
`/sfnts`、SVG 有 `<text>`（`audit.audit_bytes` 机械判定）。

## 正斜体 mathtext 约定

- **变量/物理量斜体**：`$T$`、`$\alpha$`（`mathtext.default: it`）；
- **单位、函数名、算子正体**：`\mathrm{}`——`$c = 3.0\times10^{8}\,\mathrm{m/s}$`、
  `$\mathrm{e}^{i\pi}$`（`\mathrm{e}`）、`$\sin x$`（mathtext 内建函数自动正体）；
- **微分 d、矩阵/矢量**：按期刊惯例（多数期刊微分 d 正体 `$\mathrm{d}x$`；矢量粗体
  `$\boldsymbol{x}$` 或箭头，选定后全图一致）；
- 数字与单位之间细空格 `$3.0\,\mathrm{mm}$`；轴标签量纲用 `/` 或括号，全图一种。
- mathtext 字体集 `dejavusans`（随 matplotlib 自带，无字体缺失风险）；
  需要 AMS 全量排版时用 `usetex`（本机 TeX Live；注意渲染机器必须装 TeX）。

## 色盲安全循环（家风）

- **基表**：Okabe-Ito 8 色——`#E69F00 #56B4E9 #009E73 #F0E442 #0072B2 #D55E00 #CC79A7 #000000`。
- **家风循环**（线图默认，`styles.HOUSE_CYCLE`）：上表**去 `#F0E442`**（黄在白底上线条
  对比度不足），**蓝 `#0072B2` 打头**，7 色轮转。
- 与 `house.mplstyle` 的 `axes.prop_cycle` 严格一致（audit 校验两者同源）；
  换色板 = 改两处 + 过色盲核验（红绿不可单靠色相区分；线型/标记做冗余编码）。
- 数据 colormap 不计入设计用色克制计数（1~2 色系铁律见 SKILL.md）；
  连续场优先感知均匀色图（viridis/cividis/magma），避免 rainbow/jet。

## 样式资产与叠加优先级

| 样式名 | 层 | 内容 |
|---|---|---|
| `house` | 审美层（家风） | 色循环、spines/ticks/legend/errorbar 约定、mathtext 默认 |
| `journal-aps` | 契约层 | APS 字体链/字号 + fonttype=42/svg none |
| `journal-nature` | 契约层 | Nature 字体链/字号 + fonttype=42/svg none |

**叠加顺序（谁覆盖谁）**：审美层在前、**契约层最后应用覆盖**——
`style_context("house", "journal-aps")`。SciencePlots（`science`/`no-latex` 系）是
可选审美基底：需要时插在 house 之前（`matplotlib.style.use(["science", "no-latex"])`
或叠加同名序列），只在**非冲突键**上生效；契约键（宽度/字体链/fonttype）永远归本技能。
未装 scienceplots 时该层自动缺席，不阻断。

ADR（2026-11-17）：SciencePlots 三口径差②（6/7 月空）**豁免 pin**——定性为"样式内容
资产源"而非代码依赖：家风已把用到的自定义项固化进 `styles/*.mplstyle`，运行时不 import
scienceplots，维护面≈0。用户修正案同时引入 ultraplot（写作面，347★ 三口径全过）。
