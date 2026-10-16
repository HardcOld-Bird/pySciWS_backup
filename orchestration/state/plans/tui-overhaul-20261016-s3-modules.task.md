# 任务书 tui-overhaul-20261016 · s3-modules：modules.py 重构（廉价指纹＋提示清理＋图标空格）

## 背景 / 目标

TUI 大改版 9 环节串行链第 3 环。已合并 main 的前置：s1 数据层（`plans.overview_rows/plan_digest/plan_tree_nodes`＋mtime 缓存、`taskbooks.families/family_label/md_path/detail_markdown`、`queues.water_lines` 修复、`taste.blind_queue/blind_case/ruling_rows/apply_review(blind=)`、`mirror.append_deviation/pending_deviations/deviation_lines`）；s2 共享层（`tui/theme.py` 主题常量唯一出处、`tui/widgets.py` ScrollMarkMixin/MarkedList/RowListView/ChainListItem/card/section_title/hint/make_row/ChoiceCard）。

**用户已裁定的全局 UI 规范（s3–s8 全程生效，本环节起落实）**：
- **图标后一律加一个空格再跟文字**（防终端歧义宽字符与文字重叠——用户实测 `✔完成`/`⊘中止` 重叠）：`✔ 完成`、`⊘ 中止`、`▶ 进行中`、`◆ …`、`● NEW`、`★ …`、`· …` 等凡「图标紧贴文字」处全部补空格。CLI 文本面（`pysci status` 等）同步变化可接受。
- **所有详情页成段文本框一律 Textual Markdown 渲染**（任务书/计划/决策/裁决详情：头卡＋Markdown 正文；计划详情保留**可折叠 Tree 环节树**＋树下 Markdown 文本面板，二者兼得、冲突时优先保 Tree）——本环节不动 UI 结构，但 modules 读数行清理为 s5–s8 铺路。

本环节目标：**徽标指纹不再构建全详情**（性能根因）＋各模块 detail 文本清理（删用户点名的提示字样）＋图标空格全量落地。

## 输入（只读）

- `src/pysci/skills/orchestration/tools/tui/modules.py`：现状结构（`Module` dataclass、`_fingerprint_module`/`_count_module`、`_safe_line/_safe_lines`、七模块 summary/detail、`modules()`）；`_COUNTDOWN_MARK` 常量与 `_system_stable`（倒计时行排除在徽标指纹外，**§2.2 语义必须保持**：徽标数字与详情对得上、system 倒计时不让徽标常亮）。
- `src/pysci/skills/orchestration/tools/{plans,taskbooks,queues,mirror,taste}.py`：s1 交付的新读数（签名见上）；`plans.overview_rows()` 每行 mark 取值 `{"▶进行中","✔完成","⊘中止"}` 与 `taskbooks.status_badge()` 的图标串是本环节图标空格改造点。
- `src/pysci/skills/orchestration/tools/tui/{theme,widgets,app,watermark}.py`：app.py 的 `_BADGE_NONE`/`"● NEW"` 徽标文案、`watermark.fingerprint`。
- 钉格式测试（改动需同步）：`tests/skills/orchestration/test_tui_data_layer_s1.py`（钉 mark 取值集合）、`test_live_config_panel.py`（钉「支撑席存量」行）、`test_tui_menu.py`（模块详情真实性）。

## 交付

### 1. `modules.py`——廉价指纹与 TTL memo

- `_fingerprint_module(key, title, summary, detail, digest=None)`：`cursor()` 改走 `digest()`（缺省回落 `fingerprint(detail())`）；`updates_since` 语义不变。`plans` 传 `plans.plan_digest`；`system` 传「`_system_stable` 的 `fingerprint`」但套 TTL memo（见下）。
- 新增 `_ttl_memo(fn, ttl=2.0)` 小工具（2 秒内同参复用结果）：套在 digest 族与 count 族（`decisions_count`、`_queues_count`、`_taskbook_count`、`mirror.taste_cases`）——每秒 tick/徽标刷新不再反复全量读。
- **语义保持**：digest 变 ⟺ 概览可见读数变（§2.2「徽标数字与详情对得上」）；system 倒计时行继续经 `_COUNTDOWN_MARK` 排除。

### 2. `modules.py`——detail 文本清理与重排

- `_plans_summary/_plans_detail`：改走 `plans.overview_rows()`（不再逐 plan `_plan_state` 重复读）。**删除**尾部「推进：`orch plan advance …`」与「多 plan 并行 …」两行（用户点名）。
- `_review_detail`：**删除**「reviewer 附属队列 = …」「审查面（inspector 队列）超时项 = …」「审查入口：…」三段（用户点名）；保留 `_line_review`/`_line_charter`/`twin_ruling_lines` 文本面。
- `_pending_detail`：**删除**「批复入口：…」「口述动作：…」尾行（用户点名）；新增「偏差报告（强制处理队列）」区块（`mirror.deviation_lines()`，空则一行「（无）」）。
- `_taskbook_detail`：**删除**尾行「Enter / `t` 进面板块：…」（提示字样同族清理）。
- `_system_detail`：重排为「**首行=倒计时**（`_countdown_line`，高亮语义）→ 开关/路由 → 路由配置生效值 → 仲裁/回退/告警」；`_countdown_line`/`_hold_line` 保留（每秒 tick 复用）。
- `_queues_detail`：分区标题统一（`== 成员队列 ==`/`== 支撑席存量 ==`/`== 事件水位 ==` 风格，与全局一致）；「支撑席存量」行**保留**（测试钉住）；inspector 超时明细保留在队列模块（用户只要求从 review 模块删除）。

### 3. 图标空格全量落地（用户实测缺陷修复）

扫描 `src/pysci/skills/orchestration/tools/` 全部用户可见字符串，凡「状态图标/徽标后紧贴文字」处补一个空格，至少覆盖：
- `plans.overview_rows()` 的 mark 取值 → `{"▶ 进行中","✔ 完成","⊘ 中止"}`（同步 `modules._plan_mark`、`test_tui_data_layer_s1.py` 断言）；
- `taskbooks.status_badge()` 全部七态字形输出；
- `app.py` 徽标文案（`● NEW` 等）与 `_BADGE_NONE`（独立徽标列无后随文字则不必加）；
- `queues.water_lines` 在途行、`modules` 环节行 `✔/· `、`★ 批复参考` 等。
判据：`grep -rn "✔[^ ]\|⊘[^ ]\|▶[^ ]\|◆[^ ]\|●[^ ]\|★[^ ]" src/pysci/skills/orchestration/tools/` 无「图标紧贴文字」命中（成对符号/句末除外）。

### 4. 测试同步

- `test_tui_data_layer_s1.py` 的 mark 集合断言同步新取值；
- 其余受影响断言逐个同步；**既有 TUI Pilot 测试中引用被删提示字样处改为断言不出现**（若已有断言依赖「推进：」等字样则更新）。

## 白名单（写权限）

- 可写：src/pysci/skills/orchestration/tools、tests/skills/orchestration

## 验收预期

1. `uv run --no-sync pytest tests` 全绿。
2. 性能钉（tests 断言）：替身 spy 证明 `_fingerprint_module(..., digest=…).cursor()` **不调用** `detail()`；`_ttl_memo` 2 秒内同参只算一次。
3. 禁语钉（tests 断言或 grep）：`modules` 各 detail 输出不含「推进：」「多 plan 并行」「reviewer 附属队列」「审查面（inspector 队列）超时项」「审查入口」「批复入口」「口述动作」「Enter / `t` 进面板块」。
4. 图标空格判据（本任务书 §3 的 grep）无贴字命中。
5. `_system_detail` 输出首行内容为倒计时行；`_COUNTDOWN_MARK` 排除语义不变（system 徽标不因每秒倒计时跳动而常亮——既有测试面继续有效）。
