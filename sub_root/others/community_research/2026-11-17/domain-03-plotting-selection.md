# ③科研绘图能力域 — 选型细化调研报告

- 日期：2026-11-17
- 阶段：新需求三段式第 1 段「选型」的调研部分
- 用户三口径：① star≥100；② 2026-05~2026-11 每月均有提交；③ 减少自研代码量。差②但**无可替代**→豁免（pin 版本+原地 ADR 记风险）；**有替代余地→拒绝**。
- 技能形态为主（SKILL.md + 社区工具 + 薄胶水），不整机引入大平台。
- 本轮候选：matplotlib、plotly(+Kaleido)、seaborn、plotnine、pyqtgraph、veusz、Altair、pygal、mplfinance、ParaView、COMSOL 自带出图；补充更优者：pyvista、SciencePlots、ultraplot、bokeh、meshio。

## 一、旧能力清单（pySciWS，仅提炼能力、未复制代码）

来源：`orchestration/skills/scientific-plotting/SKILL.md` + `references/plotting-01~04.md`、`src/pysci/skills/scientific_plotting/tools/*`（18 个模块 ≈300KB 自研）、`tests/skills/scientific_plotting/*`（23 个测试）、`data/skills/scientific_plotting/{recipes,templates}`。旧 venv 实装：matplotlib 3.10.9、plotly 7.1.0、scienceplots 2.2.2、ultraplot 2.6.0、pyvista（src 中未 import plotly/ultraplot，属试验性装入；pyvista 被 field.py 用于 VTK 读取）。

| # | 能力项 | 旧实现方式 | 新方案是否保留 | 新方案覆盖 / 薄胶水预算 |
|---|---|---|---|---|
| 1 | 期刊风格预设与排版契约（APS 85/170mm、Nature 89/183mm；Arial 8pt/Helvetica 7pt+回退链；`fonttype=42`/`svg.fonttype=none` 文字可编辑；正斜体 mathtext 约定） | 自研 `style.py` 15KB（rcParams 预设） | **保留** | matplotlib rcParams + `styles/*.mplstyle`（内容资产）固化期刊样式 ≈30 行胶水 |
| 2 | 色盲安全调色板（Okabe-Ito/Tol bright/vibrant + 自定义注册 + prop_cycle） | 自研 `palette.py` 7KB | **保留** | 色板 hex 列表写进样式数据（非代码）；色盲模拟见 #5 |
| 3 | 多子图布局与面板标号（grid/nested 嵌套/`(a)(b)(c)` 角标/共享轴） | 自研 `layout.py` 5KB（GridSpec 薄封装） | **保留** | matplotlib GridSpec/constrained_layout 原生 + 角标薄胶水 ≈50 行 |
| 4 | 多格式导出 + PNG 预览视觉闭环（EPS/PDF/SVG + 600dpi PNG + `_preview.png` 供 agent Read；确定性元数据） | 自研 `export.py` 5KB | **保留** | `savefig` 薄封装（matplotlib 原生支持 EPS/PDF/SVG/TIFF）≈60–100 行 |
| 5 | 出版规范机械自检 audit（宽度±2mm/最小字号/字体嵌入/面板号计数/色盲 deuteranopia 模拟/交付件宽度假绿防护） | 自研 `audit.py` 22KB | **部分保留（待裁决 A）** | 无社区对应；建议裁剪为轻量 audit（宽度/字号/嵌入/面板号）≈200–400 行，色盲模拟可查社区小库或砍 |
| 6 | 标注语义机械声明（`ANNOTATIONS.yaml` + `gid` 约定 + counts/contain/sizes/connectors 四类几何判据） | 自研 `geometry.py` 24KB | **默认不保留（待裁决 B）** | 旧项目特有：为旧 reviewer「任务书语义机械核对」量表配套 |
| 7 | 零遮挡像素判据（occlusion：文字墨迹∩彩色数据区/图线真身；ink_gap：cKDTree pixel/region/pairwise 测距） | 自研 `occlusion.py` 11KB + `ink_gap.py` 10KB | **默认不保留（待裁决 C）** | 旧项目特有：H4 reviewer 量表配套；「零遮挡铁律」**规则文字**保留进 SKILL.md |
| 8 | 数据驱动场面板 truthful colorbar（VTK/CSV→tripcolor，面板与共享 colorbar 的 cmap+norm 严格一致；端标取整 `ceil_label`） | 自研 `field.py` 23KB（pyvista 读 VTK） | **保留（薄化，待裁决 D）** | pyvista 读 VTK/CSV + matplotlib tripcolor ≈100–200 行；端标取整规则作文档规则 |
| 9 | 栅格面板数据坐标叠加（COMSOL sidecar `crop_box_px`→`imshow(extent=…)` 精确映射；overlay 原语；Rectangle 锚点旋转修正） | 自研 `raster.py` 19KB | **部分保留（待裁决 D）** | 与②仿真域出图衔接仍有用；薄化 ≈200 行，否则文档化 `imshow` 用法 |
| 10 | 图像体检 imgcheck（尺寸/色域/饱和像素/边缘水印·logo 检测） | 自研 `imgcheck.py` 12KB | **不保留** | 旧项目特有：COMSOL logo 擦除流程配套，随②域走 |
| 11 | 图生产管线约定（`build_figure` 契约、src 代码/data 产物分离、STYLE.yaml 逐键覆盖、notes.md 迭代日志、fake-green 防护） | 自研 `scaffold.py` 8KB + `runner.py` 20KB | **保留（薄化）** | 形态约定写进 SKILL.md；薄 runner（发现+运行+preview）≈100–200 行或全文档化 |
| 12 | 统一 CLI 门面 `figures`（doctor/styles/new/build/preview/audit/inspect/overlap/raster-panel/compose-grid/field-panel/list） | 自研 `figures.py` 45KB | **保留（大幅薄化）** | 薄路由 CLI ≈200–300 行（调 matplotlib/pyvista/plotly，不再内置重判据） |
| 13 | scienceplots 可选审美层（`scienceplots: true` 开关 + 排障） | style.py 集成 + 测试 | **待裁决 E** | SciencePlots 差②（见 §三），按口径默认拒绝 |
| 14 | 外部媒体面板（概念图：Blender 渲染/位图→`imshow` 面板；SVG 后备） | `concept_plus_data` 模板 + 文档 | **保留（文档化）** | matplotlib 原生能力，模板作为内容资产 |
| 15 | 可视化坑点配方（发散曲线 nan 断线、采样密度↔轴限匹配、栅格数据坐标、Rectangle 旋转锚点、极坐标半空间…） | `data/skills/.../recipes/viz_pitfalls.md` | **保留** | 内容资产迁入新技能 references（不复制代码） |
| 16 | 手动 SVG/EPS 微调（SVG 保留真实 `<text>` 节点→Illustrator/Inkscape 末修） | export 层约定 + 文档 | **保留（文档化）** | matplotlib `svg.fonttype='none'` 即可 |
| 17 | 交互式图 | 无（旧域纯静态；plotly 在 venv 但 src 未用） | **新增** | plotly(+Kaleido) 交互层（见 §二分工） |

**旧项目特有、可不保留**（除表中 #6/#7/#10 默认不保留外）：`orchestration/pods/figure`（旧编排 pod：bench 产物、任务收发收件箱、.qoder 配置——旧架构配套）；audit 假绿防护/CLI 参数形式细则等旧编排纪律（`plotting-02`）可精简进 SKILL.md 一页。

结论：旧清单 17 项中，**9 项被 matplotlib 原生 + 薄胶水直接覆盖**（#1–4、#11、#12、#14–16）；#5/#8/#9 为中小胶水（待裁决）；#6/#7/#10 为旧 reviewer/COMSOL 流程配套，建议不保留。旧域 ≈300KB 自研可收敛到 **400–700 行纯胶水**（若砍 A–D 则 <300 行）。

## 二、出版图 vs 交互图：分工与关键机制

### 科研出版图场景的硬需求 → matplotlib 主场

| 需求 | matplotlib | plotly | plotnine | Altair |
|---|---|---|---|---|
| 论文级矢量 PDF/SVG/**EPS**/**TIFF** | ✅ 全格式原生（EPS/PS/PDF/SVG/TIF/TIFF/WebP/PNG） | ⚠️ PNG/JPEG/WebP/SVG/PDF，**无 EPS/TIFF**；WebGL trace 矢量导出内嵌栅格 | ✅（matplotlib 后端） | ✅ SVG/PDF/PNG（vl-convert），无 EPS |
| 期刊尺寸/字体规范（双栏 mm、字号、字体嵌入 fonttype=42） | ✅ rcParams 全控制 | ⚠️ 模板只能设 px 级（mm 需 96dpi 换算胶水）；无字体嵌入控制 | ✅ | ❌ 字体控制弱 |
| 公式标注 LaTeX in figure | ✅ mathtext 全量（`$...$` 正斜体约定）；usetex 可选（需 TeX） | ⚠️ 仅浏览器 MathJax 渲染；**静态导出不稳定**（Kaleido [issue #360](https://github.com/plotly/Kaleido/issues/360) LaTeX 不渲染/差异，SO 多例） | ✅（继承 mathtext） | ❌ 无 LaTeX |
| 多子图/误差棒/热图/等高线 | ✅ 全覆盖（GridSpec/contour/tripcolor） | ✅ 子图/误差棒/热图；等高线需 heatmap 近似 | ✅（语法层） | ⚠️ 热图✅、**等高线无原语** |
| 交互探索（缩放/悬停/图例开关） | ❌ | ✅ 主场（HTML 交付，agent 可直接产出） | ⚠️ matplotlib 后端无交互 | ✅ vega-lite 交互 |
| 大数据量交互 | ❌ | ✅ WebGL（scattergl 等） | ❌ | ⚠️ |
| agent 可编程/无头导出 | ✅ Agg 无头 | ✅ write_image（Kaleido）/write_html | ✅ | ✅ chart.save |

### 分工组合（建议）

1. **静态出版图 → matplotlib（唯一主力）**：一切进论文/投稿的图；LaTeX 公式标注、EPS/TIFF、字体嵌入等硬需求只有它全过。
2. **交互图 → plotly**：给用户在浏览器里探索的图（HTML 单文件交付）；静态兜底走 Kaleido。**公式标注需求图一律走 matplotlib**（plotly 静态导出的 LaTeX 不可靠）。
3. **3D → pyvista（Python 内轻中量）+ ParaView（重量级后处理，按需外部）**：pyvista 覆盖 VTK 读取/切片/等值面/体渲染并可出静态 PNG；大网格流线、复杂后处理用 ParaView `pvpython`/`pvbatch` 无头脚本。
4. **统计图（可选）→ plotnine**：ggplot 语法减样板；不引则 matplotlib 直接画。
5. 两层都产 `_preview.png` 供 agent 视觉验收（沿用旧域视觉闭环，与 2.4 视觉策略一致）。

### 样式/「家风」机制（技能文档固化期刊样式）

- **matplotlib（出版主力）**：`plt.style.use` / rcContext / **`.mplstyle` 文件（纯数据，可版本化）**。方案：技能 references 携带 `styles/journal-aps.mplstyle`（85/170mm、Arial 8pt、fonttype=42）、`journal-nature.mplstyle`（89/183mm、Helvetica 回退链 7pt）+ `pysci-house.mplstyle`（家风：Okabe-Ito 色循环、spines/ticks 约定、正斜体 mathtext 默认）——**样式是内容资产不是代码**，零维护胶水。
- **plotly（交互层）**：`plotly.io.templates`（内置 `plotly_white` 等）+ 自定义 Layout template（JSON 数据）可固化家风；期刊 mm 尺寸需薄换算。plotly 模板**控制不了**字体嵌入/EPS——不承担出版合规。
- 结论：期刊样式固化在 **mplstyle 数据 + SKILL.md 规则**（正斜体表、零遮挡铁律、colorbar 端标取整等文档化）；这正是「技能文档可固化期刊样式」的落点。

## 三、候选指标表（三口径 + Py3.13/Win + agent 驱动 + 许可 + 边界）

核验方法：① stars 用 `repos.ecosyste.ms` 元数据（辅以 shields）；② 2026-05…11 逐月用 `commits.ecosyste.ms` `?since=&until=&per_page=1`（`_scripts/d03_month.mjs`）；GitHub HTML 探针 429 限流，两个边界案例以发版/pushed_at 交叉证据裁定（见注）；③ wheel 用 `pypi.org/pypi/<pkg>/json` 全量 release 扫描（`_scripts/pypi.mjs` + 轮子名核对）。原始数据：`_scripts/out_d03_{meta,month,pypi,xcheck}.jsonl`。

| 候选 | ① star | ② 2026-05~11 逐月 | ③ 减自研 | Py3.13/Win | agent 驱动 | 许可 | 能力边界/备注 | 判定 |
|---|---|---|---|---|---|---|---|---|
| **matplotlib** 3.11.2（2026-09-11） | 23.5k ✅ | ✅ 7/7 | ✅ | ✅ cp313 win_amd64 wheel | Agg 无头 `savefig` | PSF | 出版矢量全格式；mathtext/LaTeX；style sheet 机制；3D 仅 mplot3d 弱项 | **引入（核心）** |
| **plotly.py** 7.1.0（2026-09-15） | 18.9k ✅ | ✅ 7/7 | ✅ | ✅ 纯 py wheel | write_html / write_image | MIT | 交互 HTML 主场；无 EPS/TIFF；LaTeX 仅浏览器 MathJax | **引入（交互层）** |
| **Kaleido** 1.5.1（2026-11-12） | 504 ✅ | ✅ 7/7*（8 月缺=索引滞后：v1.4.0@2026-08-31 发版证据） | ✅ | ✅ py3-none-any | Python API | MIT | plotly 静态导出唯一引擎（orca 已弃）；**v1.x 需本机 Chrome/Chromium**（`plotly_get_chrome`）；LaTeX 渲染不稳定 | **引入（plotly 导出配套）** |
| **pyvista** 0.49.2（2026-10-28） | 3.9k ✅ | ✅ 7/7 | ✅（VTK 读取+3D 渲染替代旧 field.py 依赖） | ✅ 纯 wheel + vtk 9.7.2 cp313 win_amd64 | `off_screen` 渲染/导 PNG | MIT | 3D 网格/体渲染/切片；2D 出版图仍归 matplotlib | **引入（3D/场数据）** |
| **plotnine** 0.15.9（2026-11-03） | 4.8k ✅ | ✅ 7/7*（11 月缺=索引滞后：0.15.9@11-03 发版+pushed_at 证据） | ✅（ggplot 语法统计图） | ✅ 纯 wheel（依赖 statsmodels 等均有 cp313 wheel） | matplotlib 后端无头 | MIT | 统计图语法层；mathtext 公式可用；与 matplotlib 双 API 心智 | **可选引入（待裁决 E）** |
| seaborn 0.13.2 | 14.1k ✅ | ❌ 5/7 月空（0.13.2 停 2024-01，仅维护性零星提交） | ✅ | ✅ 纯 wheel | matplotlib 后端 | BSD-3 | 统计图便捷层；与 matplotlib/plotnine 重叠 | **拒绝（差②、有替代）**；用户常用可豁免→待裁决 E |
| Altair 6.3.0 | 10.5k ✅ | ✅ 7/7 | ✅ | ✅ + vl-convert-python（cp37-abi3 win ✅） | `chart.save` SVG/PDF/PNG 无头 | BSD-3 | 声明式 vega-lite；**无 LaTeX**、无等高线、出版排版控制弱 | **拒绝（与 plotly 同位、出版面弱）** |
| pyqtgraph 0.14.0 | 4.4k ✅ | ✅ 7/7 | ✅ | ✅ 纯 wheel（需 Qt 绑定：PySide6 LGPL/PyQt6 GPL） | Qt 事件循环/离屏 | MIT | 实时/大批量 GUI 曲线；agent 无头场景边际价值低 | **暂不引入（过三口径但无需求位）** |
| veusz 4.2.1 | 1.1k ✅ | ❌ 4/7 月空 | ✅ | ❌ **PyPI 仅 sdist 无 wheel**（Windows 走官方安装器） | 脚本/embed API | GPL-2.0 | 独立出版绘图 GUI+脚本；与 matplotlib 重叠 | **拒绝（差②+无 wheel+有替代）** |
| pygal（Kozea/pygal）3.1.3 | 2.8k ✅ | ❌ 5/7 月空（仅 6/7 月） | ✅ | ✅ 纯 wheel | `render_to_file` SVG | LGPL-3 | SVG 图表；无 LaTeX、出版排版弱、生态萎缩 | **拒绝（差②+有替代）** |
| mplfinance 0.12.10b0 | 4.5k ✅ | ❌ 0/7（push 停 2024-08） | ✅ | ✅ 纯 wheel | matplotlib 后端 | BSD | 金融 K 线专项；科研弱需求 | **拒绝（差②）** |
| **ParaView**（5.x 滚动） | 1.7k ✅ | ✅ 7/7 | ✅ | ⚠️ **无 PyPI**（404）→官方二进制发行版 | pvpython/pvbatch 无头脚本 | BSD-3 | 大网格 3D 后处理（流线/切片/体渲染）；重量级外部软件 | **引入（按需外部工具，装机待裁决 F）** |
| COMSOL 自带出图 | N/A（商业软件） | N/A | N/A | 外部 license | Model/Java API | 商业 | 场图导出 PNG+sidecar；**归②仿真域** | **不在③引入**（作为 #9 栅格面板输入源） |
| SciencePlots 2.2.2（2026-02） | 9.4k ✅ | ❌ 6/7 月空（last=6 月） | ✅ | ✅ 纯 wheel | matplotlib style sheet | MIT | 纯样式数据（ieee/nature/science 预设）；内容可由自写 mplstyle 替代 | **拒绝（差②、有替代）** 或按豁免→待裁决 E |
| ultraplot 2.7.5（2026-11-16） | 347 ✅ | ✅ 7/7 | ✅ | ✅ 纯 wheel | matplotlib 包装 | MIT | 子图/colorbar 便捷层；API 与 matplotlib 分叉、双层心智；旧 src 未用 | **暂不引入（过三口径但与 matplotlib 重叠）** |
| bokeh 3.10.2 | 20.5k ✅ | ✅ 7/7 | ✅ | ✅ 纯 wheel | HTML/导出 | BSD-3 | 交互可视化，与 plotly 同位 | **暂不引入（无需求位）** |
| meshio 5.3.5 | 2.3k ✅ | ❌ 0/7（push 停 2024-07） | ✅ | ✅ 纯 wheel | Python API | MIT | 网格格式互转；VTK 读取由 pyvista 覆盖 | **拒绝（差②、有替代）** |

\* 两个边界案例的裁定依据：plotnine `v0.15.9` 于 2026-11-03 发 PyPI（时间与 ecosyste.ms `pushed_at=2026-11-03` 吻合）→ 11 月有提交，commits 索引滞后为假阴性；Kaleido `v1.4.0` 于 2026-08-31 发 GitHub release（含 release 提交）→ 8 月有提交。二者按 7/7 记，报告保留星号供复核。

## 四、豁免与拒绝清单（三口径判定汇总）

**全三口径通过 → 直接引入**
1. **matplotlib 3.11.x**（核心，PSF）——出版图唯一主力 + 样式机制宿主。
2. **plotly.py 7.x**（MIT）——交互 HTML 层；配 **Kaleido 1.x**（MIT，*带注通过）做静态兜底，注意 Chrome 依赖。
3. **pyvista 0.49.x**（MIT）——VTK/场数据读取 + 3D 渲染；vtk 9.7.2 有 cp313 win wheel。
4. **ParaView**（BSD-3，按需外部安装）——重量级 3D 后处理；无 PyPI，安装与否待裁决 F。

**可选（过三口径，引入与否待裁决）**
- plotnine 0.15.x（*带注通过）——统计图 ggplot 语法层。

**豁免引入（差②但无可替代）**：本轮无。Kaleido 经交叉证据实为 7/7，无需豁免。

**拒绝（差②且有替代余地）**
- seaborn（5/7 月空）→ matplotlib/plotnine
- veusz（4/7 月空 + 无 wheel + GPL）→ matplotlib
- pygal（5/7 月空）→ matplotlib/plotly
- mplfinance（0/7）→ matplotlib 薄画（若真需 K 线）
- meshio（0/7）→ pyvista
- SciencePlots（6/7 月空）→ 自写 mplstyle 样式数据（内容资产）；若认定期刊样式数据「无可替代」可改豁免，待裁决 E

**拒绝（三口径过但与已选重叠、无需求位/出版面弱）**
- Altair（无 LaTeX、无等高线，与 plotly 同位）
- bokeh（与 plotly 同位）
- pyqtgraph（实时 GUI 场景，agent 无头边际）
- ultraplot（与 matplotlib 重叠，API 分叉）

## 五、建议组合（定稿候选）

**首选组合：matplotlib（静态出版）+ plotly+Kaleido（交互）+ pyvista（3D/场数据）**
- 静态出版：matplotlib 承担全部硬需求（矢量全格式/LaTeX/字体嵌入）；期刊样式以 `styles/*.mplstyle` 内容资产固化（APS/Nature/家风三件套）。
- 交互：plotly 产 HTML 给用户探索；需静态时 Kaleido（装 Chrome）；含 LaTeX 的图强制走 matplotlib。
- 3D/场：pyvista 读 VTK/渲染；真值 colorbar 场面板（旧 #8）薄化保留在 matplotlib tripcolor 路线。
- 统计图：默认 matplotlib 直接；plotnine 待裁决 E 后定。

**次选/按需**：ParaView（仅当出现大网格 3D 后处理需求时安装）；plotnine（统计图高频时引入）。

**拒绝**：seaborn、veusz、pygal、mplfinance、meshio、SciencePlots（或豁免）、Altair、bokeh、pyqtgraph、ultraplot——理由见 §四。

## 六、薄胶水预算（残余自研）

| # | 项 | 估算 | 条件 |
|---|---|---|---|
| 1 | 薄 CLI/流程门面（build/preview/styles/doctor 路由，替代旧 figures 45KB） | ≈200–300 行 | 必做 |
| 2 | 导出封装（EPS/PDF/SVG/TIFF + `_preview.png` + fonttype=42/确定性元数据） | ≈60–100 行 | 必做 |
| 3 | 期刊/家风 mplstyle 应用（样式数据 + `rc_context` 薄封装） | ≈30 行 + 数据资产 | 必做 |
| 4 | 面板标号/布局（`(a)(b)` 角标、grid） | ≈50 行 | 必做 |
| 5 | plotly 导出胶水（write_html/write_image + Chrome 探测提示） | ≈50 行 | 必做 |
| 6 | truthful colorbar 场面板（pyvista 读 VTK/CSV→tripcolor，旧 #8 薄化） | ≈100–200 行 | 待裁决 D |
| 7 | 栅格面板数据坐标叠加（旧 #9 薄化） | ≈200 行 | 待裁决 D |
| 8 | 出版 audit 裁剪版（宽度/字号/嵌入/面板号；色盲模拟可选） | ≈200–400 行 | 待裁决 A |
| 9 | ANNOTATIONS 四判据（旧 #6，若保留） | 重写 ≈数百行 | 待裁决 B |
| 10 | 零遮挡像素判据（旧 #7，若保留） | 重写 ≈数百行 | 待裁决 C |

**量级判断**：必做 ≈400–700 行（旧域 ≈300KB 自研）；若 A–D 全部砍掉，自研面收敛到 **<300 行纯胶水**；若全保留则回到旧域量级，不建议。

## 七、待用户裁决点

- **A. audit 机械自检保留多少**：旧 22KB（宽度/字号/嵌入/面板号/色盲模拟/假绿防护）→ 建议轻量版（宽度/字号/嵌入/面板号 ≈200–400 行）+ 视觉验收；或全砍只留 checklist。
- **B. ANNOTATIONS.yaml 标注语义四判据是否随域重建**：旧 reviewer「任务书语义机械核对」配套（24KB），非绘图本体能力 → 默认砍。
- **C. 零遮挡像素判据（occlusion+ink_gap ≈21KB）**：同为旧量表配套 → 默认砍，「零遮挡铁律」规则文字保留 SKILL.md。
- **D. 场图两条数据驱动路线（truthful colorbar #8、栅格 sidecar 叠加 #9）是否保留**：#8 与 pyvista 组合是③的正经能力（建议保留）；#9 与②域 COMSOL 出图耦合（裁掉则③只文档化 imshow）。
- **E. 统计图/样式层取舍**：plotnine（可选引入？）、seaborn（拒绝 or 豁免）、SciencePlots（拒绝 or 按样式数据豁免）。
- **F. 交互/3D 的装机面**：plotly+Kaleido 需本机 Chrome（无头机用 `plotly_get_chrome`）；ParaView 是否安装（按需）。无 Chrome 环境则 plotly 静态导出降级为「只出 HTML」。
- **G. 拒绝清单是否认可**：seaborn/veusz/pygal/mplfinance/meshio/Altair/bokeh/pyqtgraph/ultraplot；如需其一（尤其 pyqtgraph 实时 GUI、Altair JSON 规格）须按「无可替代」给豁免。

## 八、来源 URL

- 星标/元数据：`https://repos.ecosyste.ms/api/v1/hosts/GitHub/repositories/<owner>%2F<repo>`（matplotlib、plotly.py、seaborn、plotnine、pyqtgraph、veusz、vega/altair、mplfinance、Kitware/ParaView、SciencePlots、ultraplot、plotly/Kaleido、pyvista、bokeh、nschloe/meshio）；辅以 `https://img.shields.io/github/stars|last-commit/<o>/<r>.json`（含 Kozea/pygal：2.8k★、last=7 月）
- 逐月提交核验：`https://commits.ecosyste.ms/api/v1/hosts/GitHub/repositories/<owner>%2F<repo>/commits/?since=YYYY-MM-DD&until=YYYY-MM-DD&per_page=1`（2026-05…11 每月一次；脚本 `_scripts/d03_month.mjs`，原始数据 `out_d03_month.jsonl`）；GitHub HTML commits 页 429 限流，边界案例改用发版证据交叉裁定
- PyPI 版本/wheel：`https://pypi.org/pypi/<pkg>/json`（matplotlib 3.11.2 cp313 win_amd64、plotly 7.1.0、kaleido 1.5.1（py3-none-any）、pyvista 0.49.2 + vtk 9.7.2（cp313 win_amd64）、plotnine 0.15.9（2026-11-03）、altair 6.3.0 + vl-convert-python 1.9.0（cp37-abi3 win_amd64）、veusz 4.2.1 **仅 sdist**、scienceplots 2.2.2、ultraplot 2.7.5、seaborn 0.13.2、pygal 3.1.3、mplfinance 0.12.10b0、meshio 5.3.5、bokeh 3.10.2；`paraview` 404→无 PyPI）
- plotly 静态导出与 Chrome 依赖：https://plotly.com/python/static-image-export/ （Kaleido ≥1.0.0，`plotly_get_chrome`）
- plotly LaTeX 静态渲染缺陷：https://github.com/plotly/Kaleido/issues/360 ；https://plotly.com/python/LaTeX/
- Veusz 命令行/embed API：https://veusz.github.io/docs/manual/api.html
- 旧能力清单来源：`D:\XXXIIIGGG\projects\pySci\pySciWS\orchestration\skills\scientific-plotting\{SKILL.md,references/*}`、`src\pysci\skills\scientific_plotting\tools\*.py`（18 模块）、`tests\skills\scientific_plotting\*.py`（23 测试）、`data\skills\scientific_plotting\{recipes,templates}`、`orchestration\pods\figure`（旧编排 pod，判定不保留）

## 用户裁决记录（2026-11-17）
- **出版链修正案**：ultraplot（写作面，三口径全过 347★/月度发版/自带 MCP server）+ matplotlib 底座 + SciencePlots 样式资产（差②豁免 pin，零代码依赖维护面≈0）。原报告'ultraplot 重叠无需求位'判定作废。
- plotly 降为**可选件**（仅交互 HTML 需求时装；Chrome/Kaleido 依赖随之消解）；pyvista 3D 不变；ParaView/plotnine 按需。
- 自研件取推荐档：audit 轻量版留；标注四判据、零遮挡像素判据砍（规则文字入 SKILL.md）；truthful colorbar 留、COMSOL 栅格叠加归②域。拒绝清单其余项认可。
