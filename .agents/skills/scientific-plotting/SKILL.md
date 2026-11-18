---
name: scientific-plotting
description: ③科研绘图能力域操作面：出版链（ultraplot 写作面 + matplotlib 底座 + SciencePlots 样式资产）工作流——选期刊样式契约→ultraplot 绘图→公式标注→多子图面板标号→EPS/PDF/SVG/PNG 导出（fonttype=42 字体嵌入）→轻量 audit 自检→预览视觉闭环；3D/场图走 pyvista + truthful colorbar。需要画/改论文插图、出期刊投稿图、导出可编辑矢量、按期刊尺寸契约排版或自检出版合规时使用。
---

# Scientific Plotting（③科研绘图）

科研绘图能力域的操作面：**出版链只有一条**——ultraplot（写作面）+ matplotlib（底座）
+ SciencePlots 样式资产（家风固化）；**只在接缝处用薄胶水**。
选型与用户裁决见 `sub_root/others/community_research/2026-11-17/domain-03-plotting-selection.md`；
本技能只放"怎么用"。薄胶水在 `sub_root/src/pysci/skills/scientific_plotting/`
（CLI：`pysci-figures`，未注册入口时 `python -m pysci.skills.scientific_plotting.cli`）。

## 能力分层（谁负责什么）

1. **ultraplot**（写作面）：子图网格、colorbar、期刊风格排版的日常 API；
   与 matplotlib API 有分叉（双层心智），配方与坑点见 [references/ultraplot-recipes.md](references/ultraplot-recipes.md)。
2. **matplotlib**（底座）：一切最终渲染/保存的宿主；mathtext 公式、tripcolor 场图、
   GridSpec 细排、EPS/PDF/SVG/PNG 全格式导出。
3. **SciencePlots 样式资产**（豁免 pin，内容资产）：审美基底（`science`/`no-latex` 系）。
   家风已固化为 `styles/*.mplstyle`（含 SciencePlots 基础上自定义项），
   **运行时不必装 scienceplots**；契约层 `journal-*` 最后应用覆盖审美层。
4. **pyvista**（3D/场，按需 import）：VTK/VTU 读取、3D 网格/体渲染；2D 出版场图归
   matplotlib tripcolor（truthful colorbar，见下）。
5. **薄胶水** `pysci.skills.scientific_plotting`：尺寸契约、导出封装、样式应用、
   面板标号、轻量 audit、端标取整、`figures` 薄路由 CLI。画图本体在**图脚本**里
   （`build_figure` 形态约定见 [references/export-acceptance.md](references/export-acceptance.md)）。

**可选件（按需加装，不装不写死）**：plotly+Kaleido（交互 HTML 探索；Chrome 依赖）、
ParaView（大网格 3D 后处理，外部二进制）、plotnine（ggplot 语法统计图）。
**公式标注图一律 ultraplot/matplotlib**——plotly 静态导出的 LaTeX 不可靠（已实测，
Kaleido issue #360），勿用 plotly 出含公式的交付图。

## 出版链工作流（七步）

| 步 | 做什么 | 入口 |
|---|---|---|
| 1 | 选样式契约（期刊 + 栏宽） | `pysci-figures styles`；契约速查 → [references/journal-style-contract.md](references/journal-style-contract.md) |
| 2 | ultraplot 绘图（图脚本内） | `styles.style_context("house", "journal-aps")` 铺样式 → ultraplot/matplotlib 画 |
| 3 | 公式标注 | mathtext `$...$`（正斜体约定见契约册）；需 AMS 全量时 `usetex`（本机 TeX Live） |
| 4 | 多子图面板标号 | `labels.label_panel(ax, i)`（`(a)(b)` 角标，锚定面板轴左上角外侧） |
| 5 | 导出 | `export.export_figure(fig, base, journal="aps", span="double")` → EPS+PDF+SVG+PNG + `_preview.png` |
| 6 | 轻量 audit 自检 | `pysci-figures audit out/fig1.pdf out/fig1.eps --journal aps --span double --actual-width-mm W` |
| 7 | 预览闭环 | **read_image 看 `_preview.png` 再交付**（视觉策略 2.4：视觉产物必须 VLM 目检）；迭代回第 2 步 |

**Always Read `_preview.png` before declaring done**——图能干净导出 ≠ 看着是对的。

## 判据规则文字（砍件的规则留档，人工/视觉层兜底）

旧自研机械判据砍掉后，以下规则以**文字纪律**保留（出图前逐条自查；细则见
[references/export-acceptance.md](references/export-acceptance.md)）：

- **零遮挡铁律（硬伤级）**：①文字/刻度不压图线/彩色数据区（移位而非缩字号硬挤）
  ②图体与标题留间距 ③colorbar 标题↔色标、色标↔子图留边距 ④任何元素重叠 = FAIL 无例外
  ⑤**近而不交**：不该相交的线/框贴边或临界间距（<10px @600dpi）= 与重叠同级的缺陷。
- **标注语义四判据**（任务书可计数语义 → 人工核对口径）：**counts**（每类标注原语计数
  与任务书一致）、**contain**（包含关系与边距）、**sizes**（相对大小关系）、
  **connectors**（连接线端点/样式）。画带语义标注的图前，先在图注/notes 里写明
  逐要素对照（参考图对应物 → 物理含义 → 产物元素 → 处置）。
- **布局节奏/编号锚定**：标题与上方外缘间距 > 与下方子图间距；图例置于数据图下方不压
  主体；编号锚定整体 bbox（含图例/colorbar/标注）左上角。
- **colorbar 端标取整 v2**：`ceil_label` 阶梯（`d1.d2×10^k`，`d2∈{0,5}`，`d1=1` 另含
  2/4/6/8，向上取绝不裁峰；径向刻度 rmax 不适用）——`field_panel.ceil_label` 已接线。
- **设计用色克制**：整图设计用色 1~2 个色系；数据 colormap 不计入；线图色循环一律
  家风循环（Okabe-Ito 去黄，色盲安全），改色板须过色盲核验。

## 场面板 truthful colorbar（数据驱动）

`field_panel.plot_field(ax, xs, ys, vals)` + `add_truthful_colorbar(fig, m)`：
面板着色与共享 colorbar **同一个 Normalize**（cmap+norm 严格同源，truthful by
construction），vmax 默认端标取整。数据：CSV 三列或 VTK/VTU（`read_vtk_xyv`，pyvista）。
**COMSOL 栅格 sidecar 叠加不建**（归②仿真域）；③只用 matplotlib `imshow(extent=...)`
文档化用法。3D 场/网格渲染走 pyvista `off_screen` 出 PNG。

## CLI 速查

```
pysci-figures styles                       # 样式资产 + 尺寸契约 + 色循环
pysci-figures audit <files...> [--journal aps --span double --actual-width-mm W]
pysci-figures field-tick 65.48             # 端标取整 v2 换算 → 70
pysci-figures doctor                       # 绘图栈探活（ultraplot/mpl/SciencePlots/pyvista）
```

## 验收

1. 机械层：`sub_root/tests/tests.cmd all`（胶水单测 68 项 + ruff）；真实渲染用例标
   `-m hardware`（`tests.cmd hardware`）。
2. 视觉层：交付前 **read_image 目检 `_preview.png`**（多面板逐子图放大判读）；
   audit 宽度/字号/嵌入/面板号/色循环五项 PASS。
3. 出版合规：宽度契约 ±2mm、fonttype=42（PDF FontFile2 / EPS /sfnts）、
   SVG 保留 `<text>`、EPS 无 Type3 位图字体。

## 不保留 / 拒绝清单提醒

- **不保留**（旧 pySciWS）：figures CLI 全集旧壳（new/build/preview/inspect/overlap/
  raster-panel/compose-grid doctor 细则）、imgcheck、`orchestration/pods/figure`、
  标注语义四判据机械版、零遮挡像素判据（occlusion+ink_gap）。
- **不引入**：seaborn、veusz、pygal、mplfinance、meshio、Altair、bokeh、pyqtgraph
  （拒绝理由见选型文档 §四）；ultraplot 已按用户修正案引入（写作面）。
- **按需加装（不装不写死）**：plotly+Kaleido、ParaView、plotnine。

## References（渐进加载，只读需要的那一册）

- [references/journal-style-contract.md](references/journal-style-contract.md) —
  期刊样式契约：尺寸/字体链/fonttype=42/正斜体表/色盲安全循环/样式叠加优先级。
- [references/ultraplot-recipes.md](references/ultraplot-recipes.md) —
  ultraplot 配方与坑点（API 分叉、pin 策略、panel 标号、colorbar、公式、场面板衔接）。
- [references/export-acceptance.md](references/export-acceptance.md) —
  导出与验收清单：格式矩阵、确定性元数据、audit 判据、假绿防护、预览闭环、图脚本契约。
