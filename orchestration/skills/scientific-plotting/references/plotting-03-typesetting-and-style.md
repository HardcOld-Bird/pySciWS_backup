# plotting 分册 3/3 — 排版（零遮挡铁律）+ 样式（scienceplots 审美层）

> **组员侧预防层**，与 reviewer 量表 v0.4 的 H4「零遮挡铁律」成对：这里防患于未然，那里兜底
> 判 FAIL。分册索引见同目录 `plotting.md`。

## 零遮挡铁律（硬伤级四条；出图前逐条自查）

排版破损里最该零容忍的四类，**任一命中即硬伤**（reviewer 一律判 FAIL，无例外）：

1. **文字 / 刻度值不得压主图图线**。处理 = **移位且不破坏语义**（如雷达幅值刻度沿射线外移、
   或移至直径边），不是缩字号硬挤。
2. **图体与标题之间保持间距**（`ax.set_title(pad=...)` 或布局留白），标题不贴子图边。
3. **colorbar 标题 ↔ 色标、色标 ↔ 子图之间保持边距**（label 的 `pad` / `layout` 留白），
   避免三者相互顶死。
4. **任何元素重叠 = 硬伤，无例外**（含标题压 colorbar）。重叠不是「可接受的轻微瑕疵」。
5. **近而不交**：不该相交的线 / 框（如内框与场图边界）须保持距离——**贴边 / 临界间距
   （<10px @600dpi）＝缺陷**（与重叠**同级**，不降为告警）。

自查手段：`build`/`preview` 后 **Read `out/<stem>_preview.png`**，再逐子图放大判读（`panels/`）；
`figures audit <figdir>` 查宽度/字号/嵌入/色盲；`figures inspect <png>` 查水印/logo、**等效物理尺寸
（mm，由 PNG dpi 元数据换算）**，样式声明 `savefig.bbox=tight` 时并给一行**裁切提示**（成品尺寸
＝裁切后包围盒，与自述 `figsize` 名义值不同——核验「自述 mm vs 成品实测」无须再反解版面常量）。

## scienceplots 审美层（可选；契约层优先）

批语 5 的用户取向：**积极使用 scienceplots 预设**（Nature/Science/APS 系审美优先）。集成方式：

- `STYLE.yaml` 写 `scienceplots: true`（或风格序列 `["science", "nature", "no-latex"]`），
  `figures build` 时由 runner 铺为**审美基底**；
- 代码里也可 `style_context("aps", scienceplots=True)` / `build_rcparams(..., scienceplots=True)`。

**优先级铁律（谁覆盖谁）**：本技能**契约层**——设计宽度（85/170、89/183 mm）、字体链、
色盲调色板、可编辑矢量导出（`pdf.fonttype=42`、`svg.fonttype="none"`）、mathtext——
**后应用、覆盖 scienceplots**；scienceplots 只在**非冲突键**上生效（如 `grid.linewidth`、
`figure.facecolor`）——即「审美用 scienceplots，契约用本技能」。默认序列
`science + no-latex`（`no-latex` 收尾确保未装 LaTeX 也能渲染）。可用风格：`figures styles` 列出。
scienceplots 未安装时该层自动关闭（fail-soft，不阻断绘图）；请求了未注册风格名则报 `KeyError`
（静默失效升级为可见失败）。

## 排障（When something breaks）

1. `figures doctor` — 报后端、各预设解析到的字体、Agg 后端。
2. `FALLBACK` 字体行 = 预设首选字体缺失（如 Helvetica）；图仍以回退字体渲染。装字体 + 清
   matplotlib 字体缓存即恢复首选。
3. `audit` 失败会点明违背的规范（width / fontsize / fonttype / labels / colorblind）并给修法。
   工具链不可用或依据不足 → 记 n/a，**不得据此判 FAIL**。
4. **inset 子轴 + 父轴标题错位**：`Axes.inset_axes` 会触发 matplotlib 对**父轴标题**的
   auto 定位——标题可能跳到别的行高度（实测第三列 R 面板标题跳到首行 L 标题位置）。
   修法：父轴显式 `ax.set_title(..., y=1.0)` 关闭 auto 定位。另：`audit --panels` 已按
   「面板轴」计数（**排除 colorbar 轴与不可见轴**），外置 colorbar **无需**再藏进 inset
   宿主（2026-10-18，backlog 20261018-110827-figure）。
