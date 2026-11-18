# ultraplot 配方与坑点（③绘图 reference 2/3）

> 何时读：动手画图前（写作面 API 怎么用、与 matplotlib 何处分叉）、升级 ultraplot 前。
> 版本策略：**pin**（`pyproject.toml` 锁定；2.x 重写史，API 演进快）——升级前用其自带
> **MCP server** 查 API/changelog，升级后跑 `tests.cmd all` + 真实渲染冒烟。

## 定位（为什么是写作面）

ultraplot 是 matplotlib 的**排版/写作层**：`uplot.subplots()` 一站式建图网格、
自动 colorbar/图例收纳、期刊风格参数、面板标号。底座仍是 matplotlib——一切
artist/savefig/生态最终走它。**双层心智**：ultraplot 的 `Axes` 是 matplotlib 子类，
但部分方法签名/行为分叉（下面坑点）；拿不准时查 ultraplot 文档（自带 MCP），
不要按 matplotlib 直觉硬试。

## 常用配方

```python
import ultraplot as uplt
from pysci.skills.scientific_plotting import export, labels, styles

with styles.style_context("house", "journal-aps"):      # ①铺样式（契约层最后）
    fig, axs = uplt.subplots(ncols=2, refwidth="170mm")  # ②尺寸走契约宽度
    axs[0].plot(x, y1, label="run A")                     # color/label 继承家风
    axs[1].plot(x, y2, label="run B")
    for i, ax in enumerate(axs):                          # ④面板标号
        labels.label_panel(ax, i)                         #   （或 uplot 自带 abc=True）
    axs.format(xlabel=r"$t\,\mathrm{(s)}$", ylabel=r"$E\,\mathrm{(eV)}$",  # ③公式
               suptitle=r"$\Delta G = -RT\ln K$")
    export.export_figure(fig, "out/fig1", journal="aps", span="double")   # ⑤导出
```

- **面板标号**：ultraplot 的 `abc=True`/`axs.format(abc=...)` 可直接出 (a)(b)；
  统一走 `labels.label_panel` 是为了角标位置/字重口径一致（audit 计数同源）。
- **colorbar**：ultraplot 自动收纳（`fig.colorbar(m, loc="r")` 语法）；场面板用
  `field_panel.plot_field`（truthful norm）后交给 uplt/mpl colorbar 皆可。
- **公号标注**：mathtext `$...$`（正斜体约定见契约册）；公式标注图**禁止**转 plotly。
- **嵌套/复杂版式**：`uplot.subplots` 的 `ncols/nrows/width_ratios` 够用则不碰
  GridSpec；真需要缝合既有 mpl Axes 时，直接用底座 GridSpec。

## 坑点清单（实测口径）

0. **首次 import 写 `~/.ultraplot`（Windows 固定，XDG 仅 Linux）**——受限沙盒下直接
   `PermissionError: 拒绝访问 ~/.ultraplot`；matplotlib 的 `%LOCALAPPDATA%\matplotlib`
   缓存同样受限（降级临时目录、反复重建字体缓存）。**解法**：`import ultraplot`（含
   matplotlib）**之前**调 `styles.ensure_plotting_env()`（HOME/USERPROFILE/MPLCONFIGDIR
   一并重定向进项目 `data/skills/scientific_plotting/cache/home`，自包含 + 沙盒安全）。
   图脚本顶部固定三行：`from pysci.skills.scientific_plotting import styles` →
   `styles.ensure_plotting_env()` → `import ultraplot as uplt`。
8. **图例/caption 自动轴**：ultraplot 给外置图例（如 `legend(loc="b")`）建独立
   "subplot_N" 轴——audit 按"可见 + 携带数据（`has_data()`）"计面板轴，自动轴不算；
   图脚本也不要给这种轴打面板标号。
9. **默认共享轴范围**：`uplt.subplots(ncols=2)` 同行**共享 y**（刻度标签只留最左；
   两面板尺度差大时小信号被压扁，实测确认）——尺度不同用 `sharey=False`（或
   `share=False`）解锁；刻意对比同量纲时共享反而是优势，按图意选。

1. **API 分叉**：ultraplot 2.x 重写过一轮——`uplot.subplots` 的 kwargs、
   `axs.format()` 与 matplotlib 的 `set_*` 不是逐一等价；报错先查 ultraplot 文档
   （MCP），不要改用 matplotlib 绕过去而不留痕（版式一致性归 ultraplot 管）。
2. **rcParams 仍然有效**：ultraplot 不吞 rcParams——`styles.style_context` 铺的
   契约键（字体链/fonttype/色循环）照常生效，这就是"样式资产 + 契约层"能管住它的原因。
3. **refwidth 单位**：用 `"170mm"`/`"85mm"` 直接对应尺寸契约；`refwidth` 指**单个
   子图**宽度，多列时总宽 ≈ 各列之和（含间距）——audit 以成品实测 ±2mm 复核。
4. **tight bbox 假绿**：ultraplot 默认倾向紧凑布局；导出宽度以实测为准（见契约册
   "假绿"条）。
5. **matplotlib 版本约束**：ultraplot pin 带 `matplotlib<3.12` 一类约束，由 uv 解析；
   冲突时记 ADR，不手改 pin 硬凑。
6. **图例/colorbar 与零遮挡**：自动收纳 ≠ 零遮挡——出预览后仍按铁律目检
   （标题压 colorbar、图例压线是高频硬伤）。
7. **3D 不走 ultraplot**：mplot3d 弱项；3D/体渲染走 pyvista（`off_screen` 出 PNG），
   2D 场图走 matplotlib tripcolor + `field_panel`。

## 与薄胶水的分工

ultraplot/matplotlib 负责**画**；`pysci.skills.scientific_plotting` 只管
**契约**（尺寸/字体/色循环）、**落盘**（导出封装）、**验收**（audit/preview）。
图脚本不 import 胶水以外的自研件，不维护第二套样式入口。
