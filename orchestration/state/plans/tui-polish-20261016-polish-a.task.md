# 任务书 tui-polish-20261016 · polish-a：视觉四件（选中态配色/树状制表符对齐/标题配色/决策行间距）

## 背景 / 目标

TUI 大改版（tui-overhaul-20261016，9 环节已全交付）经用户**实机验收**，整体认可，但提出 9 条视觉/交互抛光反馈。本环节修其中 4 条**纯视觉**问题（用户原话语义为准）：

1. **选中态可读性**：列表项选中后背景变橙色（漂亮，保留），但**标题文本仍是白色难辨、状态徽标（也橙色）直接不可见**。要求：**选中态下标题与状态徽标改黑色**。适用范围：任务书列表（用户点名）＋决策审计列表（用户点名 `[!]` 选中后不可见）——**统一推广到全部列表**（主菜单、计划、已裁决列表同理）与 ChoiceCard 选中态（若有同病）。
2. **次级行树状制表符**：子项前的 `└` 很漂亮，但**非末项应使用 `├`**，使相邻子项的制表符上下连线（树状）。另外两个连带问题：
   a. 主项行**标题紧贴状态徽标**（连一个空格都没有）——徽标与标题之间留出可见间距；
   b. **状态徽标字符串长度不一**（`delivered`/`rejected`…宽度不同），而次级行结构是「徽标列＋└ 标签…」，徽标列 auto 宽把 `└` 顶得**上下行不在一条竖线**——徽标列**定宽**（或把连线列固定），保证 `├`/`└` 上下严格对齐连线。
3. **标题配色**：详情页所有标题（头卡「任务书 tb-…」、Markdown 内「全流程报告：…」「任务概述」等各级标题）当前是**蓝色**，深灰蓝背景下几乎不可辨。改为**高对比色**（如亮黄/亮青，与全局主题协调），覆盖：头卡标题（`_T_TITLE` 族）与 **Markdown widget 的 h1–h6**（Textual Markdown DEFAULT_CSS 需覆写）。
4. **决策审计行**：`[!]` 前缀与后续标题之间**无分隔**（加空格/间距）；选中后 `[!]` 不可见（并入第 1 条修）；详情页标题蓝（并入第 3 条修）。

## 输入（只读）

- `src/pysci/skills/orchestration/tools/tui/widgets.py`：`make_row(badge, title, meta, indent, markup)`（徽标/标题/meta 三格 Horizontal）、`ChoiceCard`、`_T_HL` 高亮（background accent+bold）、theme 常量。
- `src/pysci/skills/orchestration/tools/tui/app.py`：`_tb_list_item`（次级行现结构 `make_row(badge, f"└ {label}　{title}", indent=+1)`——`└` 在标题格内、被前面 auto 宽徽标格顶歪）、列表 CSS（`#tb-list > ListItem.-highlight` 等）、决策行渲染（`[!]`/`[i]` 旗标＋标题）。
- `src/pysci/skills/orchestration/tools/taskbooks.py`：`status_badge()`/`STATUS_BADGES`（七态字形，长度不一）。
- 全局规范：图标后空格（保持）、常量单一出处（theme.py，判据测试在场——新增 CSS 字面量须进 theme 或复用既有常量）。

## 交付

1. **选中态**：列表 `.-highlight` 下 `.row-title-cell/.row-badge-cell/.row-meta-cell`（及行内一切文本）强制深色（黑），橙底上可辨；七模块主菜单行同理（`● NEW` 徽标选中时可见）。既有 `_T_HL` 视觉（橙底粗体）不变，只改前景。
2. **次级行连线**：`├ ` 用于非末子项、`└ ` 用于末子项（family members 顺序本就是展示序）；连线列上下对齐（徽标列定宽或结构调整，自选实现，但**验收以「同一家族各行 ├/└ 在同一显示列」为准**，注意 CJK/ASCII 混排显示宽度）；徽标与标题之间留间距（组行与子项行一致）。
3. **标题色**：头卡标题＋Markdown h1–h6 全部换高对比色（建议亮黄 `$warning` 或亮青，二选一或更优）；深灰蓝背景下清晰可辨。
4. **决策行**：`[!]/[i]` 旗标与标题间加间距（与任务书列表徽标间距风格统一）。

测试（同步/新增 `tests/skills/orchestration/` 用例）：
- 选中态：CSS 判据（`-highlight` 下深色前景在场）＋ Pilot 选中后读面不回归；
- 连线：同家族子项行 `├/└` 选择正确（末子项 `└`、其余 `├`），行结构中连线列偏移一致（可用渲染文本宽度断言）；
- 标题色：`_T_TITLE`/Markdown 覆写 CSS 在场且非原蓝色；
- 全量 `uv run --no-sync pytest tests` 绿。

## 白名单（写权限）

- 可写：src/pysci/skills/orchestration/tools、tests/skills/orchestration

## 验收预期

1. 全量 pytest 绿。
2. 逐条对应用户反馈 1/2/3/8 的现象消除（用户将实机复看）。
3. 图标空格判据、常量单一出处判据、s9 跨面钉（禁语/键位/滚动）仍绿。
