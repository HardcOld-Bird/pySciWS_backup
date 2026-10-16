# 交付全文留档：devops-20261016-173739-tui-overhaul-20261016-s2-widgets

- member: devops
- kind: result
- code: 0
- plan: tui-overhaul-20261016
- step: s2-widgets
- task_file: orchestration/state/plans/tui-overhaul-20261016-s2-widgets.task.md
- sid: 73d2478d-8fd1-4bb5-978f-65b55b82aeef
- duration_ms: 1631310
- ts: 2026-10-16T18:05:40+08:00

## 交付正文

任务书 tui-overhaul-20261016-s2-widgets（TUI 共享层）已完成交付：main merge commit **e1c643b**（feat commit 47d46ae），push 成功（5222be8..e1c643b main→main）。

**改动清单**（5 文件，+551/−16）：
1. `src/pysci/skills/orchestration/tools/tui/theme.py`（新建）：9 个 `_T_*` 主题常量整体迁入（值一字不改），成为 `tui/` 全目录唯一出处；模块 docstring 写明插值引用纪律。
2. `src/pysci/skills/orchestration/tools/tui/app.py`：删除本地 9 常量定义与原注释块，改为 `from .theme import (_T_BG, …)`（9 个原位插值照旧，CSS f-string 一字未动，**行为零变化**）。
3. `src/pysci/skills/orchestration/tools/tui/widgets.py`（新建，s3–s8 命名接口）：
   - `ScrollMarkMixin`（覆写 `watch_scroll_y`→`app.on_pane_scrolled(self)`，调 super 保基座行为，缺失/异常 fail-soft）+ `MarkedVerticalScroll/MarkedList/MarkedTree`；
   - `ChainListItem`（记录 `Click.chain`）、`RowActivated(Message)`（index/item_id/source="key"|"click"/chain）、`RowListView`（Enter→action_select_cursor 发 RowActivated 不发 Selected；点击通路读 click_chain 发同型消息）。**实现要点**：Textual 消息派发沿 MRO 逐类调同名 handler（`message_pump._get_dispatch_methods` 实测：基座 `_on_list_item__child_clicked` 会双跑、多发一条 Selected 与重复激活）——已用 `event.prevent_default()` 拦截，探针实测单击=[1]、双击=[1,2]、每 Click 恰一条激活；
   - 视觉件 `card()/section_title()/hint()/make_row()/ChoiceCard`（-selected 高亮开关 + `Chosen(value)` 消息），DEFAULT_CSS 全部 f-string 插值 theme 常量，无重复 CSS 字面量。
4. `tests/skills/orchestration/test_tui_visual_structure.py`：常量单一出处判据改为**扫 `tui/` 目录全 .py 聚合文本**（每个钉住的字面恰 1 次），其余判据不动。
5. `tests/skills/orchestration/test_tui_widgets_s2.py`（新建 7 测）：theme 迁入完整性、ScrollMark 触发 + 宿主缺席 fail-soft、RowListView Enter/单击/双击三路（含 `selected_seen == []` 钉「不发 Selected」）、ChoiceCard 点击与高亮、视觉件构造断言。

**测试/验证证据**：
- `PYTHONPATH=<wt>/src uv run --no-sync pytest <wt>/tests -q` → **3738 passed, 42 skipped, 6 deselected**（3m11s，全绿；含改判据后的 test_tui_visual_structure 与新增 test_tui_widgets_s2，既有 7 个 TUI Pilot 测试语义断言零调整）；
- 验收 #2 实测：`python -c "from pysci.skills.orchestration.tools.tui import theme, widgets"` → import ok；CSS 字面量唯一性由 `test_visual_constants_defined_once_in_source`（目录扫描）钉住并通过（`border: round $accent`/`$warning`/标题条/弱文案在 theme.py 各恰 1 次；app.py 既有 `.row-badge` 的 bold-warning 是不同视觉参数且不在钉住判据内，按「app.py 行为零变化」原则未动）；
- ruff format/check 改动面全过，pre-commit 钩子全过；`pysci-dev worktree merge` 交集为空、会话数不减（76→76）、自动摘面。
