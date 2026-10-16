# 交付全文留档：devops-20261016-180847-tui-overhaul-20261016-s3-modules

- member: devops
- kind: result
- code: 0
- plan: tui-overhaul-20261016
- step: s3-modules
- task_file: orchestration/state/plans/tui-overhaul-20261016-s3-modules.task.md
- sid: 73d2478d-8fd1-4bb5-978f-65b55b82aeef
- duration_ms: 767586
- ts: 2026-10-16T18:21:37+08:00

## 交付正文

任务书 tui-overhaul-20261016-s3-modules（modules.py 重构）已完成：main merge commit **c657a0f**（feat commit d58f93c），push 成功（e1c643b..c657a0f）。

**改动清单**（8 文件，+285/−111）：
1. `tui/modules.py` —— 廉价指纹与 TTL memo：
   - `_fingerprint_module(..., digest=None)`：`cursor()` 走 `digest()`（缺省回落 `fingerprint(detail())`），`updates_since` 语义不变；
   - 新增 `_ttl_memo(fn, ttl=2.0)`（2 秒内同参复用，越窗重算；memo 随 `modules()` 每次新建、不跨会话残留），套在 digest 族（`plans.plan_digest`、system 的 `_system_stable` 指纹）与 count 族（`_taskbook_count`/`_queues_count`/`mirror.taste_cases`/`mirror.decisions_count`）；
   - **system 接线要点**：详情改挂 `_system_detail`（含每秒倒计时，与 `app._tick_detail` 的逐秒刷新设计对齐）、**指纹**挂 `_system_stable` 的 fingerprint——徽标不因倒计时跳动常亮（§2.2 既有判据 test_countdown_excluded_from_badge_fingerprint 仍绿）。
2. `tui/modules.py` —— detail 清理与重排（全部按用户点名）：
   - `_plans_summary/_plans_detail` 改走 `plans.overview_rows()`（summary 零 state 读取；删除死件 `_plan_mark`/`_plan_state`；detail 环节行保留——每 plan 一次 `plans.load_state`，走 s1 mtime 缓存）；删除「推进：…」「多 plan 并行 …」两行；
   - `_review_detail` 删除「reviewer 附属队列」「审查面（inspector 队列）超时项」「审查入口：…」三段（inspector 超时明细保留在队列模块）；
   - `_pending_detail` 删除「批复入口：…」「口述动作：…」尾行，新增「—— 偏差报告（强制处理队列）——」区块（`mirror.deviation_lines()`，空则「（无）」）；
   - `_taskbook_detail` 删除尾行「Enter / `t` 进面板块：…」；
   - `_system_detail` 重排为 首行=倒计时 → 开关/路由 → 路由配置生效值（含边界避让）→ 仲裁/回退/告警；
   - `_queues_detail` 分区标题统一为 `== 成员队列 ==`/`== 支撑席存量 ==`/`== 事件水位 ==` 风格（括注内容保留，钉「支撑席存量」的既有测试照过）。
3. 图标空格全量（`✔/⊘/▶/◆/●/★` 后必隔一空格）：`plans._plan_mark` 与 `overview_rows` 文档串、`cmd_plan_list` 标记 → `▶ 进行中/✔ 完成/⊘ 中止`；`plans.plan_tree_lines`、`taskbooks._STEP_ICONS`、`modules` 环节行——图标字面量改为自带尾随空格并同步格式串（**渲染逐字节不变**，且源码字面量本身不再有「图标紧贴字符」形态）。
4. 测试：`test_tui_data_layer_s1.py` mark 断言同步新值；`test_tui_visual_structure.py` 待决模块断言改为「旧提示字样不出现 + 偏差报告区块在场」；`test_autonomy_e2e.py`/`test_inspector_autonomy.py` 的「超时项」断言改为钉「review 面已清理（明细归队列模块）」；新增 `tests/skills/orchestration/test_tui_modules_s3.py`（9 测）。

**测试/验证证据**：
- `PYTHONPATH=<wt>/src uv run --no-sync pytest <wt>/tests -q` → **3747 passed, 42 skipped, 6 deselected**（3m03s；= 前值 3738 + 新增 9）；
- 验收②性能钉：新增测试以替身 spy 证明 `digest=…` 的 `cursor()` 不调用 `detail()`（且真接线 plans→`plan_digest`、system→`_system_stable`），`_ttl_memo` 1.9s 命中/越 2.0s 重算各实测；
- 验收③禁语钉：`test_module_details_drop_named_prompts` 对七模块 detail 断言 8 条旧提示字样零命中；
- 验收④图标空格：机械判据实测 `grep -rn "✔[^ ]\|⊘[^ ]\|▶[^ ]\|◆[^ ]\|●[^ ]\|★[^ ]" src/pysci/skills/orchestration/tools/` 命中 **10 处，全部为 `[▶]` 成对括号形**（benchd/drain/inspectd/watchdog 的日志前缀与 `mark = "▶" if spawned …` → 输出均为 `[▶]`），即 §3 明示豁免的「成对符号/句末」，**无一处图标紧贴文字**；同判据已固化为测试（豁免集=`空格/]/）/)"/'/`` `），越界即红；
- 验收⑤：`_system_detail` 首行=倒计时行、`_system_stable` 排除含 `_COUNTDOWN_MARK` 的行走测试钉住；
- ruff format/check 全过、pre-commit 全过；`pysci-dev worktree merge` 交集为空、会话数不减（76→76）、自动摘面；main 侧 `from ...tui import theme, widgets, modules` import ok。

**一处需知会的行为面**：system 模块此前把 `_system_stable` 当**详情**用（倒计时行永远不显示，tick 刷新形同虚设）；本环节按任务书「详情=首行倒计时、指纹走 stable」拆开，倒计时现在真正逐秒显示（`_COUNTDOWN_MARK` 排除语义不变，徽标不常亮）。
