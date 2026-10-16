# 任务书 tui-polish-20261016 · polish-c：文案排版三件（倒计时人性化/回退列表排版/家族编号）

## 背景 / 目标

TUI 抛光计划**最后一环**（polish-a 已交付 `86e2b54`：选中态黑字/`├└` 定宽连线/标题亮黄/决策行间距；polish-b 已交付 `656b435`：VSCode 启动器 `cmd /c` 稳健化/Esc 回进入前视图）。本环节修用户实测反馈的 3 条**文案排版**问题：

**反馈 #6（倒计时人性化＋实时统一）**：「系统与路由」页三处时长呈现可读性差，用户原话摘录：
- 「倒计时：距下次切换还有**3550 分 27 秒**（10-19 08:50 → 入峰）」——"3550 分"难以理解 → **换算天/小时**（如「2 天 11 小时 50 分 27 秒」）；
- 「（距峰 10-19 09:00 还有 **59.3 小时**）」（`routing.py:564` 一带，`{gap/3600:.1f} 小时`）同理；
- 「边界避让：窗口 600s（**距边界 3550 分**，窗口外不延迟）」（`modules._hold_line`）同理。
要求：**统一格式**（天/时/分/秒，按量级省略前导零段）、**统一实时更新**（三处都随 system 页每秒 tick 刷新——现只刷倒计时卡）、**统一精确到秒**。建议实现单点格式化函数（如 `routing.humanize_duration(seconds) -> str`，数据层一处、三处调用）。

**反馈 #7（渠道回退列表排版）**：「系统与路由」页回退段现状（用户原话摘录）：「最近：10-13 08:26 …（channel_error）；10-13 10:53 …；…10-15 09:24 …」一条长串用「；」连接自动换行，且「最近：」前有孤立缩进、换行行无缩进——层级混乱。要求：**每项一行**（列表式），缩进层级统一（「回退：」标题下：统计行、「最近：」小标、逐条一行缩进对齐）。

**反馈 #9（任务书家族自动编号）**：大任务列表按**创建先后**给每个家族自动编号「#1 」等（**本体与返工共享同一编号**）；编号**只显示在主项前**，子项不重复；编号稳定（以家族 root 创建时间序为准，不随列表排序/新家族加入而漂移既有编号——新家族递增新号）。渲染位置：组行标题前（如「#3 ＋标题」），与徽标/连线列排版协调（注意 polish-a 的列宽纪律）。

## 输入（只读）

- `src/pysci/skills/orchestration/tools/tui/modules.py`：`_countdown_line`（`{left//60} 分 {left%60} 秒`）、`_hold_line`（`距边界 {…//60} 分`）。
- `src/pysci/skills/orchestration/tools/routing.py:559-564`：`距峰 {…} 不足提前量` / `距峰 {…} 还有 {gap/3600:.1f} 小时`。
- `src/pysci/skills/orchestration/tools/tui/app.py`：`_refresh_countdown_card`/`_tick_detail`（现只刷倒计时卡——反馈 #6 要求三处统一实时）；`_build_tb_rows`/`_tb_list_item`（家族编号挂点）。
- `src/pysci/skills/orchestration/tools/taskbooks.py`：`families()`（root/成员；编号按 root ts 升序）。
- `src/pysci/skills/orchestration/tools/tui/cards.py`：`system_blocks()` 回退段渲染（反馈 #7）。
- polish-a 列宽纪律：`row-lead-cell` 定宽 2、徽标列 padding；编号不得破坏 `├/└` 对齐。

## 交付

1. **`humanize_duration`**（数据层单点）：`3550*60+27 → "2 天 11 小时 50 分 27 秒"`；600 → "10 分 0 秒"（或"10 分"？——**统一精确到秒**，故分段也带秒：定案为全段「X 天 X 小时 X 分 X 秒」按量级省略前导整零段，但**秒段恒显**）；三处调用点改造＋`_tick_detail` 刷新面扩到含 `_hold_line`/routing 行的卡（或三行并入每秒 tick 的稳定 id 卡，只 update 文本）。
2. **回退列表**：逐条一行、缩进层级统一（无「；」长串、无孤立缩进）。
3. **家族编号**：`families()` 按 root 创建序编号（稳定递增）；组行标题前「#N 」；子项不带；测试钉「创建序编号/家族共享/重进不漂移」。
4. 测试：新增 `test_tui_polish_c.py`（humanize 边界值（<1min/跨天/整点）、三处调用点文案、tick 每秒刷三处、回退列表逐行、家族编号序/稳定/子项不带）；全量绿。

## 白名单（写权限）

- 可写：src/pysci/skills/orchestration/tools、tests/skills/orchestration

## 验收预期

1. `uv run --no-sync pytest tests` 全绿（含新用例）。
2. 用户实机复看三处（倒计时可读/回退列表逐行/编号稳定）。
3. s9 跨面钉＋polish-a/b 判据仍绿；图标空格、常量单一出处仍绿。
