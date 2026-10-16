# 任务书 tui-overhaul-20261016 · s4-app-frame：app.py 通用面（键位收敛＋滚动接线＋概览板美化）

## 背景 / 目标

TUI 大改版 9 环节串行链第 4 环。前置已合并 main：s1 数据层、s2 共享层（`tui/theme.py`、`tui/widgets.py`：`ScrollMarkMixin/MarkedVerticalScroll/MarkedList/MarkedTree`、`ChainListItem/RowListView/RowActivated`、`card/section_title/hint/make_row/ChoiceCard`）、s3 modules（廉价指纹＋TTL memo＋提示清理＋图标空格）。后续 s5–s8 建各新视图（任务书分组/计划/决策/盲审）。

本环节做 **app.py 通用面**：footer 键位收敛、滚动位置指示接线（修复「滚轮/滚动条导航时位置指示不更新」用户实测缺陷）、提示机制退役、概览板美化。**行为面变更仅限本任务书列明的**；视图结构（新面板）留给 s5–s8。

**全局 UI 规范（全程生效）**：图标后加空格再跟文字（s3 已全量落地，新写字符串照此）；所有详情页成段文本框一律 Textual `Markdown` 渲染（任务书/计划/决策/裁决详情 = 头卡＋Markdown 正文；计划详情保留可折叠 Tree＋树下 Markdown 文本面板，冲突时优先保 Tree）——本环节只在接线处预留（如 tb-detail 正文容器结构），不建新视图。

## 输入（只读）

- `src/pysci/skills/orchestration/tools/tui/app.py`：现状 `_ACTIONS_BY_VIEW`（49-61）、`_GATED_ACTIONS`、`_SCROLL_ACTIONS`、`_SCROLL_PANES`、`_SCROLL_MARK_HEAD`、`_DETAIL_HINT_MARKERS`/`_split_detail`、`_refresh_scroll_mark`/`_scroll_widgets`/`_reset_scroll`、`_mark_seen`、`_render_summary`、`action_back`、`compose`/`on_mount`/`_hide_all_views`。
- `src/pysci/skills/orchestration/tools/tui/widgets.py`（s2）：`ScrollMarkMixin.watch_scroll_y` 回调 `app.on_pane_scrolled(self)`（fail-soft）。
- `src/pysci/skills/orchestration/tools/tui/modules.py`（s3）：detail 输出已无提示尾行（`_split_detail` 无行可分）。
- 测试面：`tests/skills/orchestration/test_tui_{user_paths,visual_structure,detail_scroll,menu,spike}.py`。

## 交付

### 1. footer 键位收敛（用户点名：t/j 只在主菜单）

- `_ACTIONS_BY_VIEW`：`menu: {taskbooks, journal}`；`detail: ∅`；`taskbooks: {tb_new, tb_accept, tb_reject}`（`o` 键 s5 随分组列表再加）；`tb-detail: {tb_accept, tb_reject}`；`journal/tb-new/tb-reject: ∅`。
- 同步 `test_tui_user_paths.py`/`test_tui_visual_structure.py` 键位矩阵断言（t/j 在非 menu 视图不得出现在 `screen.active_bindings`）。

### 2. 滚动位置指示接线（用户实测缺陷：滚轮/滚动条导航不更新位置）

- `_SCROLL_PANES` 泛化为 `视图名 → (pane_id, mark_id)`（去掉正文 id）；s4 收编现有三视图：`detail:(#detail-body,#detail-scroll)`、`tb-detail:(#tb-detail-body,#tb-detail-scroll)`、`taskbooks:(#tb-list,#tb-scroll)`。**s5–s8 各自视图落地时追加本表条目**（在各自任务书里注明）。
- 新增 `PysciTUI.on_pane_scrolled(pane)`：由 `ScrollMarkMixin.watch_scroll_y` 回调 → `_refresh_scroll_mark()`（滚轮、滚动条拖动、键盘、程序化赋值全通路实时刷新）。
- `compose`：`#detail-body`/`#tb-detail-body` 换 `MarkedVerticalScroll`、`#tb-list` 换 `MarkedList`；任务书列表底部新增 `#tb-scroll` 位置行（文案同 `_SCROLL_MARK_HEAD`，ListView 也报「位置 整屏（无需滚动）/xx%（第 y/max 行）」）。
- `_reset_scroll` 双拍校准保留；`_scroll_widgets`/`_refresh_scroll_mark` 按新表结构适配（taskbooks 视图的状态落 `self.tb_scroll_mark` 新可断言属性）。

### 3. 提示机制退役（用户点名删除「Esc/← 返回主菜单」等冗余）

- 删除 `_DETAIL_HINT_MARKERS`/`_split_detail`（数据层提示行 s1/s3 已清，机制无存在意义）。
- `detail_hint`/`tb_detail_hint` 改为 **TUI 生成的操作提示**（如 tb-detail 的「a 接受　r 驳回」），**一律不含**「Esc/← 返回主菜单」「Esc/← 返回任务书列表」——Esc 语义由 Footer「返回/退出」广告承担（顺带消除「← 无绑定却在提示里」的假广告）。
- `detail_text`（静态通道/测试读面）同步为新结构（标题/正文/操作提示三段，无「Esc/←」字样）。

### 4. 概览板美化（需求 13）

- `#summary`：CSS 改为主题统一形态（`_T_BORDER_SOFT` 细框＋`_T_TITLE` 标题行＋`_T_BODY` 正文；颜色格式与各面板一致）；`_render_summary` 首行加「〔模块名〕概览」标题（数据仍是 `mod.summary()` 3–5 行，截行逻辑不动）。

### 5. 杂项

- `_mark_seen`：`cursor()` 与水位相同则**不写盘**（每秒 tick 不再每秒写 `mirror-seen.json`）。
- `action_back` 现有回退链保持（tb-detail→taskbooks、其余→menu、menu→退出）；**后续环节各自视图落地时追加回退分支**（plan-detail→plans 等，在 s6–s8 任务书注明）。
- `check_action` 语义不变（受管 action 按视图闸控；滚动 action 只在 `_SCROLL_PANES` 视图可用）。

## 白名单（写权限）

- 可写：src/pysci/skills/orchestration/tools/tui、tests/skills/orchestration

## 验收预期

1. `uv run --no-sync pytest tests` 全绿。
2. footer 钉（tests）：menu 视图 `t`/`j` 在 `screen.active_bindings` 且有效果；detail/taskbooks/tb-detail 视图二者不出现；「广告的键必须真有效果」矩阵全过。
3. 滚动钉（tests）：`pane.scroll_y = N`（模拟滚轮/滚动条）后位置指示**立即**更新；`#tb-scroll` 在任务书列表在场且随列表滚动变化；键盘 PgUp/PgDn/Home/End 既有行为不回归。
4. 禁语钉（tests）：`detail_text`/`tb_detail_hint` 等读面不含「Esc/← 返回」「批复入口」「口述动作」等字样；`_split_detail`/`_DETAIL_HINT_MARKERS` 从源码消失。
5. 概览板：`#summary` 首行为「〔模块名〕概览」标题行，样式常量全部引自 `theme.py`（常量单一出处判据仍绿）。
