# 任务书 tui-overhaul-20261016 · s6-plans-ui：计划与派发（列表→详情：可折叠 Tree＋Markdown 面板）

## 背景 / 目标

TUI 大改版 9 环节串行链第 6 环。前置已合并 main：s1 数据层（`plans.overview_rows()/plan_digest()/plan_tree_nodes()`＋mtime 缓存）、s2 共享层（`RowListView/ChainListItem/RowActivated`、`MarkedList/MarkedVerticalScroll/MarkedTree`、`card/section_title/make_row`）、s3 modules（plans detail 走 overview_rows、已删「推进：/多 plan 并行」尾行、图标空格）、s4 app 通用面（`_SCROLL_PANES: 视图→(pane_id,mark_id)`＋`on_pane_scrolled`；`ScrollMarkMixin` 已覆盖滚动/内容尺寸/面板尺寸三路）、s5 任务书面板（行模型/async 重建范式 `await lv.clear()`/`o` 键/Markdown 详情——**新视图照此范式**）。

本环节实现用户需求 4+5+6：
- 点进「计划与派发」先**由新到旧列所有 plan**（行=状态徽标+id+进度+goal），点击进详情页；
- 详情页**美观渲染**（用户裁定：**保留可折叠 Tree 环节树** ＋ **一切成段文本框用 Textual Markdown 渲染**——树下 Markdown 文本面板联动选中环节；二者冲突时优先保 Tree）；
- 删除「推进：`orch plan advance …`」「多 plan 并行…」提示（s3 已删数据面，UI 面再钉不出现）；
- 进入板块**无卡顿**（s1 缓存已备；本环节 UI 只走 `overview_rows()`/`load_state` 缓存路径，不得全量解析 YAML）。

## 输入（只读）

- `src/pysci/skills/orchestration/tools/plans.py`（s1）：`overview_rows() -> [{id, goal, mark, done, total, created, completed_at, abandoned_at, current}]`（mark=`▶ 进行中`/`✔ 完成`/`⊘ 中止`，已按 created 降序）；`plan_tree_nodes(pid) -> [{id, path, kind:"super"|"leaf", mode, status, member, brief, children}]`；`load_state(pid)`（mtime 缓存）。
- `src/pysci/skills/orchestration/tools/tui/widgets.py`：`MarkedTree`、`RowListView/RowActivated`、`card()/section_title()/make_row()`；`ScrollMarkMixin` 三路重算（滚动/内容尺寸/面板尺寸）。
- `src/pysci/skills/orchestration/tools/tui/app.py`（s4/s5）：`_SCROLL_PANES` 注册方式、`action_back` 回退链、async 列表重建范式（s5 注记：`await lv.clear()` 防 DuplicateIds）、`#tb-title` 一行式操作说明风格。
- 测试面：`test_tui_spike.py`/`test_tui_menu.py`/`test_tui_user_paths.py`（menu Enter 分派语义会变——plans 模块不再进通用 `detail`）。

## 交付

### 1. `plans` 列表视图（新容器 `#plans`）

- menu「计划与派发」Enter → 本视图（不再走通用 `#detail`；`_open_detail` 分派按模块 key 改写）。`action_back`：`plans → menu`。
- 行（`RowListView`＋`make_row`，async 重建）：徽标（图标＋空格）＋ plan id ＋ 进度 `done/total` ＋ goal（超长截断）；序= `overview_rows()` 序（新→旧）；单击/Enter 行 → `plan-detail`。
- `#plans-scroll` 位置行＋`_SCROLL_PANES["plans"] = ("#plans-list", "#plans-scroll")`；`#plans-title` 一行操作说明（↑↓ 选｜Enter 看详情｜Esc 返回）。
- 空态：「（暂无计划）」一行。

### 2. `plan-detail` 视图（新容器 `#plan-detail`）

布局（上→下）：
1. **头卡**（`card()`）：goal、状态徽标、进度 done/total、created（及 completed/abandoned 时刻若有）。
2. **可折叠 Tree**（`MarkedTree`，数据 `plans.plan_tree_nodes(pid)`）：superstep 节点 label `〔serial|parallel〕 <id>`；叶子 label `<图标> <id> [<status>] <member>｜<brief>`（图标带空格；done→✔、failed→✗、其余·）；默认展开全部、可折叠收起。
3. **Markdown 文本面板**（`Markdown(id="plan-detail-md")`）：**树节点选中联动**——渲染该节点详情的成段文本（Markdown 组装：节点 id/路径、status、member、brief 全文、evidence 路径、task_file、session_id、耗时信息等有啥写啥；superstep 节点显示其 children 概览）。默认选中=当前进行中环节（无则根/首叶）。
- 滚动：整页可滚（`MarkedVerticalScroll` 包 1–3 区）或树内滚——**选一并注册** `_SCROLL_PANES["plan-detail"] = (pane_id, "#plan-detail-scroll")`，保证位置行在场且随滚轮/拖动实时变。
- `action_back`：`plan-detail → plans`；标题行 `#plan-detail-title` = plan id。

### 3. 测试（新增 `tests/skills/orchestration/test_tui_plans_panel.py` + 同步既有）

- 列表：行数=plan 数、新→旧、行含徽标/进度/goal；Enter 进 `plan-detail` 且 `view=="plan-detail"`。
- 详情：Tree 节点数与 `plan_tree_nodes` 一致、superstep/叶子 label 形态正确；选中节点 → Markdown 面板文本含该节点 brief；空 plan/缺 state 不崩。
- 性能钉：monkeypatch `plans._load_yaml` 抛异常后，**plans 列表视图仍能打开**（证明 UI 只走 state.json 缓存路径）。
- 禁语钉：plans 列表/详情读面与渲染不含「推进：」「多 plan 并行」。
- 同步 `test_tui_spike.py`/`test_tui_menu.py`/`test_tui_user_paths.py`：menu Enter 分派（plans 模块→`plans` 视图；queues/system/pending 仍→通用 `detail`）。

## 白名单（写权限）

- 可写：src/pysci/skills/orchestration/tools/tui、tests/skills/orchestration

## 验收预期

1. `uv run --no-sync pytest tests` 全绿。
2. §3 新增测试全过（列表序/Tree 结构/Markdown 联动/性能钉/禁语钉）。
3. 实机手感（自证）：46 个 plan 量级下打开「计划与派发」**无可感知卡顿**（列表构建只读 state.json 缓存）。
4. 图标空格判据、常量单一出处判据仍绿。
