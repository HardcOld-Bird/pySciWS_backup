# 任务书 tui-overhaul-20261016 · s5-taskbooks-ui：任务书分组列表＋Markdown 详情＋VSCode

## 背景 / 目标

TUI 大改版 9 环节串行链第 5 环。前置已合并 main：s1 数据层（`taskbooks.families()/family_label()/md_path()/detail_markdown()`、`status_badge` 图标空格）、s2 共享层（`RowListView/ChainListItem/RowActivated(index,item_id,source,chain)`、`MarkedList/MarkedVerticalScroll`、`card/section_title/make_row`）、s3 modules（图标空格全量）、s4 app 通用面（`_ACTIONS_BY_VIEW` 收敛、`_SCROLL_PANES: 视图→(pane_id,mark_id)` 泛化＋`on_pane_scrolled` 实时位置指示、`#tb-scroll` 已就位、提示机制退役、`o` 键留给本环节）。

本环节是**用户核心需求**：任务书列表按返工家族分组。用户原始诉求（原话语义，务必逐条满足）：
- 大任务列表**每组只展示最新一条**（无返工=本体）；返工多轮的条目不再散落全列表。
- 单击组条目 → 在其下方**展开缩进次级列表**，按**从新到旧**依次展示每一次返工和本体；单击次级项 → 进对应任务书详情页。
- **双击组条目（每组最新项）→ 直接进该最新项详情页**。
- 交互定案：**统一展开**——单件组（无返工）单击同样展开（次级列表仅「本体」一项）；初始**全部收起**。

全局规范（全程生效）：图标后加空格再跟文字；详情页成段文本框一律 Textual `Markdown` 渲染（本环节 tb-detail 即「头卡＋Markdown 正文」）。

## 输入（只读）

- `src/pysci/skills/orchestration/tools/taskbooks.py`（s1）：`families() -> [{root, latest, members:[…新→旧…], size}]`（家族按 latest.ts 降序；members 为 state 记录 dict）、`family_label(state) -> "返工第 n 轮"|"本体"`、`detail_markdown(tb_id) -> str`（段序：delivered「★ 批复参考」置顶→正文→批语→流程报告→返工卡片）、`md_path(tb_id) -> Path`、`status_badge(status)`、`accept/create_rework/list_all`。
- `src/pysci/skills/orchestration/tools/tui/widgets.py`（s2）：`RowListView`/`ChainListItem`/`RowActivated`（Enter→source="key",chain=1；单击→source="click",chain=1；双击→chain=2，且**不发 Selected**）、`card()/section_title()/make_row()`。
- `src/pysci/skills/orchestration/tools/tui/app.py`（s4）：`_SCROLL_PANES` 表结构与 `on_pane_scrolled`、`_tb_selected_index`/`_tb_act`/`_open_reject_panel`/`action_tb_*` 现状、`#tb-scroll`。
- Textual 8.2.8 机制事实：`Markdown` widget 高度 auto 不自带滚动（外层 `MarkedVerticalScroll` 滚），`update()` 返回 AwaitComplete **异步完成**（测试断言前 `await pilot.pause()`）；`Pilot.click(sel, times=2)` 按 chain=1、2 发两次。

## 交付

### 1. 分组列表（`#taskbook` 面板）

行模型 `self.tb_rows: list[dict]`（列表按它重建；**测试只准走行模型选行，禁止裸下标**）：
```json
{"kind":"group","tb_id":"<最新项 id>","root":"<root id>","title":"…","badge":"◆ delivered","size":3,"round":2}
{"kind":"sub","tb_id":"<该项 id>","root":"<root id>","label":"返工第 1 轮","indent":1}
```
- 展开态 `self.tb_expanded: dict[root_id, bool]`，**初始全部 False（收起）**——主列表一屏只见每组最新/本体（用户核心要求）。
- 组行渲染（`make_row`）：徽标＋标题＋右 meta「共 N 件」（N=1 不显示 meta 或显示「本体」）；次级行：缩进 `└ `＋`family_label`＋徽标＋标题，从新到旧（members 序），本体殿后。
- 交互（`RowListView.RowActivated` 分派，替换现 `on_list_view_selected` 对 tb-list 的处理；menu 列表不动）：
  - **组行 Enter（source="key"）= 立即展开/收起**；**组行单击（chain=1）= 延迟 0.35s 展开/收起**（`set_timer`，chain=2 到达即取消）；**组行双击（chain>=2）= 取消延迟、直接进最新项详情**。若 Pilot 下定时器不稳，允许退化为「单击立即切换、双击=切换+进详情」（最终视图正确即可，测试断言最终态）。
  - **次级行单击/Enter = 立即进该项详情**。
  - **`o` 键（新增 BINDING `("o","tb_open_latest","打开最新")`）= 进选中行目标详情**（组行→最新项、次级行→该项，同 `_tb_target_id` 语义）；加入 `_ACTIONS_BY_VIEW["taskbooks"]` 与 `_GATED_ACTIONS`。
- 批复目标解析 `_tb_target_id()`：组行→`tb_id`（最新项）、次级行→该项 `tb_id`；`a`/`r`（含详情页内）都走它。
- 展开/收起后列表重建并保持高亮在被操作的组行上；`#tb-scroll` 位置行随重建复位（s4 的 `_reset_scroll` 已接列表重建）。
- `#tb-title` 头行改简洁操作说明（↑↓ 选｜Enter 展开/收起｜双击或 o 进详情｜n 新增｜a 接受｜r 驳回），一行内说完。

### 2. tb-detail：Markdown 渲染＋VSCode 按钮

- `#tb-detail-body` 内置 `Markdown(id="tb-detail-md")`（外层 `MarkedVerticalScroll` 已有），打开详情时喂 `taskbooks.detail_markdown(tid)`；头卡保留（tid＋标题＋状态徽标，`card()` 形态）；可断言读面 `self.tb_detail_body` = markdown 文本。
- 新增「用 VSCode 打开原文」按钮 `#tb-open-vscode`＋消息行 `#tb-detail-msg`：`shutil.which("code")` 缺失或 `subprocess.Popen(["code", str(taskbooks.md_path(tid))])` 异常 → `"[!] 打开失败：…"`；成功 → `"[√] 已在 VSCode 打开 <路径>"`。**Popen 不阻塞 UI**。
- `tb_detail_hint` 保持「a 接受　r 驳回」（s4 已定）；a/r 在 tb-detail 内照常可用（目标=`tb_detail_id`）。

### 3. 测试（新增 `tests/skills/orchestration/test_tui_taskbook_groups.py` + 同步既有）

新增（用真实 `taskbooks.create/create_rework` 在 tmp state 建家族数据，或 monkeypatch `families`）：
- 初始收起：多轮家族只出 1 行（组行=最新）；单件组 1 行。
- 单击组行展开：次级行序 = [返工第 2 轮, 返工第 1 轮, 本体]（从新到旧、缩进、label 正确）；再击收起。
- 双击组行 → `tb-detail` 且 `tb_detail_id` == 最新项；`o` 键同效；次级行 Enter → 该项 `tb_detail_id`。
- `a`/`r` 目标：组行 → 最新项 state 变 accepted；次级行 → 该项（`create_rework` 挂到该项）。
- tb-detail：`tb_detail_body` 含「★ 批复参考」段（delivered 态）与 Markdown 结构符；VSCode 按钮两分支（mock `shutil.which`/`subprocess.Popen`）。
- 同步 `test_tui_taskbook_detail.py`/`test_tui_tb_defects_fix.py`：行选择改走 `tb_rows` 行模型辅助；`lv.index` 裸下标一律替换。

## 白名单（写权限）

- 可写：src/pysci/skills/orchestration/tools/tui、tests/skills/orchestration

## 验收预期

1. `uv run --no-sync pytest tests` 全绿。
2. 交互钉见 §3 新增测试全过；`Pilot.click(times=2)` 后最终 `view == "tb-detail"` 且为最新项。
3. 图标空格判据（s3 固化的 grep/测试）仍绿；tb-detail 读面无「Esc/← 返回」字样。
4. `self.tb_rows`/`self.tb_expanded`/`self.tb_detail_id` 等可断言状态面与界面实际渲染一致（测试读状态面）。
