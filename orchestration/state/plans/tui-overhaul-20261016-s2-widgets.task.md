# 任务书 tui-overhaul-20261016 · s2-widgets：TUI 共享层（theme.py 主题常量 + widgets.py 组件）

## 背景 / 目标

TUI 大改版 9 环节串行链的第 2 环（s1 已交付数据层：`plans.overview_rows/plan_digest/plan_tree_nodes`、`taskbooks.families/family_label/md_path/detail_markdown`、`queues.water_lines` 修复、`taste.blind_queue/blind_case/ruling_rows/apply_review(blind=)`、`mirror.append_deviation/pending_deviations/deviation_lines`，均已合并 main）。本环节建 TUI **共享层**：主题常量唯一出处 + 共享组件，供 s3–s8 各视图复用。**不改 TUI 行为**（app.py 只做常量迁移动作等价的改动），行为变更留 s4 起。

用户全局诉求：所有界面统一 UI 设计、现代美观风格统一。现状：主题常量 `_T_*` 9 个定义在 `src/pysci/skills/orchestration/tools/tui/app.py:107-115`，`test_tui_visual_structure.py` 用「运行期拼接 grep 判据」钉住同一视觉参数（CSS 字面量）在源码中只出现一次。迁入 theme.py 后该判据改为扫 `tui/` 目录聚合，纪律保持。

## 输入（只读）

- `src/pysci/skills/orchestration/tools/tui/app.py`：现状常量 `_T_BORDER/_T_BORDER_SOFT/_T_INSET/_T_TITLE/_T_SUBTLE/_T_BODY/_T_HINT/_T_HL/_T_BG`（107-115 行）与 CSS f-string 插值用法；`_rows()` 的列表行结构（Horizontal+Static+classes）。
- `tests/skills/orchestration/test_tui_visual_structure.py`：`test_visual_constants_defined_once_in_source` 的判据实现（迁移后改扫目录）。
- `.venv/Lib/site-packages/textual/`（8.2.8，只读参考）：`events.py`（Click.chain）、`widgets/_list_view.py`、`widget.py`（watch_scroll_y）。

## 交付

### 1. `src/pysci/skills/orchestration/tools/tui/theme.py`（新建）

- 9 个 `_T_*` 常量整体迁入（值一字不改），成为**唯一出处**（每个 CSS 字面量在 `tui/` 全目录源码中恰出现 1 次，就在本文件）。
- app.py 改为 `from .theme import _T_BG, ...` 并原位使用（行为零变化；CSS f-string 插值照旧）。

### 2. `src/pysci/skills/orchestration/tools/tui/widgets.py`（新建）

**已核实机制事实**（照做勿重推）：
- Textual 8.2.8 **没有** `Scroll` 事件类；滚轮/滚动条拖动/键盘/程序化赋值全部经 `scroll_y` reactive → 唯一可靠接线 = 覆写 `Widget.watch_scroll_y(old, new)`。
- `ListView.Selected` 由 Enter（`action_select_cursor`）和单击（`ListItem._on_click` → `_ChildClicked` → `_on_list_item__child_clicked`）都会发，**无法区分键鼠**；`Click.chain`（2=双击，`events.py:641`）；`Pilot.click(sel, times=2)` 会按 chain=1、2 发两次。

组件清单（命名即后续环节接口，s3–s8 以此为准）：
- `ScrollMarkMixin`：覆写 `watch_scroll_y`，调 `getattr(self.app, "on_pane_scrolled", None)`——存在则 `self.app.on_pane_scrolled(self)`（app 侧实现在 s4；**fail-soft**：缺失/异常不塌滚动）。
- `MarkedVerticalScroll(ScrollMarkMixin, VerticalScroll)`、`MarkedList(ScrollMarkMixin, ListView)`、`MarkedTree(ScrollMarkMixin, Tree)`。
- `ChainListItem(ListItem)`：覆写 `_on_click` 记录 `self.click_chain = event.chain` 再放行原逻辑。
- `RowListView(ListView)`：覆写 `action_select_cursor`（Enter → 发自定义消息，**不发 Selected**）与 `_on_list_item__child_clicked`（读 `event.item.click_chain` → 发同型消息）。消息类 `RowActivated(Message)` 携带 `index: int`、`item_id: str | None`、`source: str`（"key"|"click"）、`chain: int`。
- 视觉件（DEFAULT_CSS 一律 f-string 插值引用 theme 常量，**不得重复 CSS 字面量**）：
  - `card(*lines, id=None, classes="")`：圆角边框卡（`_T_BORDER+_T_BG+_T_INSET`），返回 Static（多行文本）；
  - `section_title(text)`：分区标题行（`_T_TITLE`）；`hint(text)`：弱化提示行（`_T_HINT`）；
  - `make_row(badge, title, meta="", indent=0)`：统一列表行 Horizontal（徽标+标题+右对齐 meta；indent 控左侧留白），返回 Horizontal；
  - `ChoiceCard(Static)`：盲审大选项卡——`-selected` class 高亮切换（`toggle_selected()/selected` 属性），`on_click` 发 `ChoiceCard.Chosen` 消息（携带 `value`，构造时传入）。

### 3. 测试同步与新增

- `test_tui_visual_structure.py` 的常量单一出处判据改为**扫 `tui/` 目录全 .py 聚合文本**（每个字面恰 1 次）；其余判据不动。
- 新增 `tests/skills/orchestration/test_tui_widgets_s2.py`：
  - `ScrollMarkMixin`：构造测试 App 挂 `on_pane_scrolled`，`pane.scroll_y = N`（模拟滚轮/滚动条路径）即触发回调；
  - `RowListView`：Enter → `RowActivated(source="key", chain=1)`；`Pilot.click(times=2)` → 收到 `chain>=2` 的 `RowActivated(source="click")`；单击 → `chain=1`；
  - `ChoiceCard`：点击后 `-selected` 高亮与 `Chosen` 消息值正确；
  - `card/section_title/make_row` 渲染不塌（构造即过）。

## 白名单（写权限）

- 可写：src/pysci/skills/orchestration/tools/tui、tests/skills/orchestration

## 验收预期

1. `uv run --no-sync pytest tests` 全绿（含改判据后的 test_tui_visual_structure 与新增 test_tui_widgets_s2）。
2. `uv run --no-sync python -c "from pysci.skills.orchestration.tools.tui import theme, widgets"` 正常；`grep -rn "border: round" src/pysci/skills/orchestration/tools/tui/` 等各 CSS 字面量在 tui/ 目录内恰出现 1 次（判据已由测试钉住）。
3. app.py 运行行为零变化：既有 7 个 TUI Pilot 测试不因本环节改动（除常量判据外）而需调整语义断言。
