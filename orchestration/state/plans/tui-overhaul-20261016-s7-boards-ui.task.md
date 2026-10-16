# 任务书 tui-overhaul-20261016 · s7-boards-ui：决策审计列表＋队列/系统/待决三看板卡片化

## 背景 / 目标

TUI 大改版 9 环节串行链第 7 环。前置已合并 main：s1 数据层、s2 共享层、s3 modules（图标空格、禁语清理）、s4 通用面（滚动三路重算、键位收敛）、s5 任务书分组＋Markdown 详情、s6 计划列表＋Tree＋Markdown 联动（46 plan 实测打开 304ms）。s6 接口注记务必遵守：**新视图注册三件套**（compose 容器＋`_hide_all_views` 名单＋`_SCROLL_PANES` 条目＋`action_back` 分支——漏 `_hide_all_views` 会让隐藏面板抢布局，s6 实锤）；列表重建走 **async `await lv.clear()`**；首帧前定位光标延后 `call_after_refresh`；用户数据进 widget id 先过 `_widget_id()`。

本环节实现用户需求 7+10+11+12：
- **决策审计**改「列表→详情页」结构（详情页正文一律 **Textual Markdown 渲染**——用户裁定；UI 美观）；
- **队列与运行**卡片化（现多行文本框被用户评为「简陋」）：每成员一卡（**含空队成员**、在途条目带成员归属）；
- **系统与路由**卡片化：**首卡=距下一次切换倒计时，每秒实时更新**（用户需求 10）；
- **待决与验收**卡片化：待决事项/待批复交付/**偏差报告**（`mirror.deviation_lines()`）分区卡。

全局规范：图标后加空格再跟文字；成段文本框一律 Markdown（本环节 decision-detail 与各卡内成段文本适用；短状态行卡片可静态文本但风格统一）。

## 输入（只读）

- `src/pysci/skills/orchestration/tools/inspectd.py`：`decision_rows()`（行字段 `id, ts, class(mechanical|judgment), remedy_shape(restore|change), title, verdict_ref, ruling, commit, verify, backlog_status, attention, note`）、`stale_queue_items()`、`depths()`、`enabled()`。
- `src/pysci/skills/orchestration/tools/{queues,workq,mirror,workflow,backpressure,drain,benchd,events,channels}.py`：`queues.counts(m)/all_members()/water_lines()`、`mirror.pending_items()/deviation_lines()/_line_*`、`workflow.backlog_pending()/backlog_by_status()`、`benchd.queue_counts()`、`events.status_lines()`、`channels.read_events()` 等（modules.py `_queues_detail/_system_detail/_pending_detail` 现状即拼装范本）。
- `src/pysci/skills/orchestration/tools/routing.py`：`state()`（next_at/next_mode）、`countdown_s()`、`face()`（leadtime_s/valley_preset/windows_text/boundary_hold_s/warnings）。
- `src/pysci/skills/orchestration/tools/tui/{app,modules,widgets,theme}.py` 现状（s3 的 `_system_detail` 首行=倒计时、`_COUNTDOWN_MARK` 排除语义；s4 的 `_tick_detail` 每秒刷新机制）。
- 测试面：`test_tui_{spike,menu,user_paths,visual_structure,app_frame_s4}.py`（menu Enter 分派、`detail_body` 读面）、`test_live_config_panel.py`（「支撑席存量」行钉）、`test_tui_modules_s3.py`。

## 交付

### 1. 决策审计：`decisions` 列表＋`decision-detail` 详情

- menu「决策审计」Enter → `decisions` 列表视图（不再进通用 `detail`）。行（`_RowMarkedList`＋`make_row`）：attention 旗标（`[!]`/`[i]`，**attention 置顶**）＋ ts ＋ class/remedy_shape ＋ title（截断）；其余按 ts 新→旧。
- Enter/单击行 → `decision-detail`：**头卡**（id/ts/class/shape/attention）＋ **Markdown 正文**（`Markdown(id="decision-detail-md")`：judgment/verdict_ref/ruling、remedy（commit/restore）、verify、backlog_status、note 组装成段；无值字段省略）。
- 注册：compose 容器 `#decisions`/`#decision-detail`＋`_hide_all_views`＋`_SCROLL_PANES` 两条＋`action_back`：`decision-detail→decisions→menu`；`#decisions-title`/`#decision-detail-title`/`#decisions-scroll`/`#decision-detail-scroll`。
- 空态「（暂无决策记录）」。`mirror.decisions_detail_lines`（静态通道）不动。

### 2. 队列与运行卡片化（`#detail` 看板，模块 key=queues）

- `#detail-body` 内由**卡片栈**取代单个 Static（卡=`card()`/分区=`section_title()`，风格统一）：
  - **成员队列区**：`queues.all_members()` **每成员一张卡**（含全零「（空）」卡）：计数行（待/在做/完成/失败）＋在途条目（`id`、plan·step、起于时刻——**条目带成员归属**，s1 `water_lines` 文本面已修，卡片面照此语义）；
  - **支撑席存量区**：backlog（pending/needs_leader/锁）、bencher 队列、inspector 开关/锁/积压/超时项（明细）、活跃会话、带宽延期；
  - **事件水位区**：`events.status_lines()`。
- 「支撑席存量」字样**保留**（`test_live_config_panel.py` 钉住）；`modules._queues_detail` 文本面保留（徽标指纹用），但 UI 渲染以卡片为准。

### 3. 系统与路由卡片化（模块 key=system）

- 卡序：**① 倒计时卡（置顶、高亮）**——「距下次切换 MM 分 SS 秒（时刻 → 入谷/入峰）」**每秒实时刷新**；② 开关/路由卡；③ 路由配置生效值卡（leadtime/peak/valley/边界避让 `_hold_line`）；④ 仲裁/回退卡（含回退事件累计）；⑤ 配置告警卡（有 warnings 才显示）。
- 每秒 tick 只刷倒计时卡文本（`_tick_detail` 适配或等价机制），**其余卡不重建**（防闪烁）；徽标指纹继续走 `_system_stable`/`_COUNTDOWN_MARK` 排除语义（徽标不常亮）。

### 4. 待决与验收卡片化（模块 key=pending）

- 卡：**待决事项**（`mirror.pending_items()`，空则「（无待决）」）、**待批复交付**（taskbooks delivered 列表）、**偏差报告（强制处理队列）**（`mirror.deviation_lines()`，空则「（无）」）。需求 12 删除字样 s3/s4 已清，本环节 UI 面再钉不出现。

### 5. 测试

- 新增 `tests/skills/orchestration/test_tui_decisions_panel.py`：列表（attention 置顶/新→旧/行形态）→ Enter 进 detail；Markdown 正文含 ruling/commit 字段；空态；回退链。
- 新增 `tests/skills/orchestration/test_tui_boards_s7.py`（或并入上文件）：队列卡=每成员一卡（含空队）+在途条目带成员名；系统首卡=倒计时且 `_tick_detail` 后文本变化（模拟秒进）；待决卡含偏差报告区。
- 同步 `test_tui_spike.py`/`test_tui_user_paths.py`/`test_tui_visual_structure.py`：menu Enter 分派（decisions→新视图；queues/system/pending 仍 `detail` 但读面=卡片文本）；`detail_body` 读面断言适配。

## 白名单（写权限）

- 可写：src/pysci/skills/orchestration/tools/tui、tests/skills/orchestration

## 验收预期

1. `uv run --no-sync pytest tests` 全绿。
2. §5 新增测试全过；决策列表→详情导航与 Markdown 渲染可断言（状态面 `self.decision_rows`/`self.decision_detail_*`）。
3. 队列卡：`queues.QUEUE_MEMBERS` 每成员恰一卡（含空队卡）；在途条目显示成员归属。
4. 系统首卡=倒计时且每秒跳动（tick 驱动、只刷倒计时卡）；徽标不常亮（既有判据绿）。
5. 各看板读面/渲染无「Esc/← 返回主菜单」「批复入口」「口述动作」字样；图标空格判据、常量单一出处判据仍绿。
