# 交付全文留档：devops-20261016-201417-tui-overhaul-20261016-s7-boards-ui

- member: devops
- kind: result
- code: 0
- plan: tui-overhaul-20261016
- step: s7-boards-ui
- task_file: orchestration/state/plans/tui-overhaul-20261016-s7-boards-ui.task.md
- sid: 73d2478d-8fd1-4bb5-978f-65b55b82aeef
- duration_ms: 1010522
- ts: 2026-10-16T20:31:11+08:00

## 交付正文

任务书 tui-overhaul-20261016-s7-boards-ui（决策审计列表→详情＋队列/系统/待决三看板卡片化）已完成：main merge commit **55b2748**（feat commit 698576b），push 成功（0799ce2..55b2748）。全量 **3796 passed × 连跑两次稳定**。

**改动清单**（7 文件，+1008/−24）：

1. `tui/app.py` —— **决策审计**（需求 7）：menu「决策审计」Enter → `#decisions` 列表（不再进通用 `#detail`）。行模型 `self.decision_rows`（照搬 `inspectd.decision_rows()` 序＝**attention 置顶**、其余 ts 倒序）；行＝`make_row(旗标, "ts[:19]  class/remedy_shape", meta=title[:40], markup=False)`；空态「（暂无决策记录）」；Enter/单击 → `#decision-detail`＝**头卡**（id/ts/class/shape/attention，attention 行标「〔!〕免判决+change——最需人工审计」）＋ **`Markdown(id="decision-detail-md")` 正文**（发现／判决（依据＋裁示）／实施与核验（commit/核验/backlog/发现状态/通道）／留痕，**无值字段省略**）；状态面 `decision_rows`/`decision_detail_id|_title|_head|_md|_scroll_mark`；注册三件套齐全（compose＋`_hide_all_views`＋`_SCROLL_PANES` 两条＋`action_back`：decision-detail→decisions→menu）。
2. `tui/cards.py`（新建 239 行）—— 三看板**结构化块**（卡=`card()`、分区=`section_title()`；数据全走既有数据层，不动 `modules` 文本面）：`queues_blocks()`（**每成员恰一卡含全零「（空）」**，在途条目 `· [m] 在途 id（plan·step）起于 …` 带成员归属；支撑席卡含 backlog/bencher/inspector/超时明细/活跃会话/带宽延期；事件水位卡）、`system_blocks()`（**① 倒计时卡置顶**→② 开关/路由→③ 配置生效值（含边界避让）→④ 仲裁/回退→⑤ 告警（有才出））、`pending_blocks()`（待决事项／待批复交付／**偏差报告（强制处理队列）**）。
3. `tui/app.py` —— **`#detail-body` 卡片栈**：`MarkedVerticalScroll(Static(#detail-body-text), Vertical(#detail-cards))`，看板模块走 `_render_detail_cards`（async `remove_children`+`mount`），其余模块（如「盲审与审查」）沿用文本面（互斥显隐）；`detail_body` 读面同步为卡片文本（**「支撑席存量」等钉串保留**）；`_tick_detail` 改为**只刷倒计时卡**（`_refresh_countdown_card` → `#detail-countdown`，其余卡控件对象不重建、不闪烁），徽标仍走 `_system_stable`/`_COUNTDOWN_MARK` 排除语义。
4. `tui/widgets.py`（s2 共享层**增量**）：`make_row(..., markup=True)` 与 `card(..., markup=True)` 新增开关——**Rich markup 会把 `[i]`/`[sim]` 当标签吃掉**（两处实测：决策旗标 `[i]` 变空、队列卡 `[sim]` 被吞），故决策列表与看板卡显式 `markup=False`。
5. 测试：新增 `test_tui_decisions_panel.py`（7 测：attention 置顶/行形态/空态/头卡/判决与实施段/无值省略/Markdown 组件/回退链/禁语）＋ `test_tui_boards_s7.py`（7 测：每成员一卡含空队、在途带成员名、系统首卡=倒计时、**tick 只刷倒计时卡**（比对卡控件 `id()` 前后一致）、待决三区含偏差报告、读面禁语、非看板模块仍文本面）；同步 `test_tui_user_paths.py`/`test_tui_visual_structure.py`（模块遍历加 decisions 分支）。

**测试/验证证据**：
- `PYTHONPATH=<wt>/src uv run --no-sync pytest <wt>/tests -q` → **3796 passed, 42 skipped, 6 deselected**，**连跑两次均全绿**（3m47s / 4m05s）；相关子集 **453 passed**；
- 验收②：新增 14 测全过；导航（列表→详情→回退链）与 Markdown 渲染均以状态面断言（`decision_rows`/`decision_detail_*`）；
- 验收③：`queues.QUEUE_MEMBERS` 每成员恰一卡（`len(cards)==len(QUEUE_MEMBERS)+2`）＋空队卡含「（空）」＋在途行带 `· [sim] 在途`（实机复看：首卡 `[sim] 待=0 在做=0 完成=0 失败=0 （空）`）；
- 验收④：系统首卡 `id == "detail-countdown"`；`_tick_detail` 后文本变化而**全部卡控件对象 `id()` 不变**（实证「只刷倒计时卡」）；既有「倒计时不进徽标指纹」判据仍绿；
- 验收⑤：三看板与决策读面禁语全过；图标空格判据（s3）与常量单一出处判据（s2）仍绿；
- ruff format/check 全过、pre-commit 全过；`pysci-dev worktree merge` 交集为空、会话数不减（76→76）、自动摘面。

**给 s8 的接口注记**：① 新视图注册四件套（compose 容器＋`_hide_all_views`＋`_SCROLL_PANES`＋`action_back` 分支）——本环节决策两视图已按此落地；② 看板类如需新卡：在 `tui/cards.py` 加 `*_blocks()` 并登记到 `app._detail_blocks_for` 的映射即可（卡/分区渲染与 `detail_body` 文本面自动派生）；③ **卡与列表行的内容是纯文本，务必 `markup=False`**（`[x]` 形 token 会被 Rich 吃掉——本环节两处实锤）；④ 需每秒刷新的卡：给它稳定 `id` 并在 `_tick_detail` 里只 `update()` 该控件（勿重建卡片栈）。
