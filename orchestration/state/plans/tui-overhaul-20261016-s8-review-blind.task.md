# 任务书 tui-overhaul-20261016 · s8-review-blind：盲审交互流程＋偏差报告＋已裁决列表

## 背景 / 目标

TUI 大改版 9 环节串行链第 8 环。前置已合并 main：s1–s7（数据层/共享层/modules/app 通用面/任务书分组+Markdown 详情/计划 Tree+Markdown/决策列表+三看板卡片化）。s7 接口注记必读：新视图注册四件套（compose＋`_hide_all_views`＋`_SCROLL_PANES`＋`action_back`）；**卡与列表行内容是纯文本须 `markup=False`**（`[x]` 形 token 被 Rich 吃掉，两处实锤）；每秒刷新的卡给稳定 `id` 只 `update()` 勿重建。

本环节实现用户需求 8（盲审与审查重做），语义**必须符合 `data/others/original_designs/2_REFACTOR.md` §10（盲审反转）与 §5.4（偏差报告队列）**。用户原始定案：
- 从主菜单进「盲审与审查」：**存在未审核裁决 → 直接进盲审界面**（顶部列明「尚待盲审 N 项」），逐案呈现问题＋**大选项卡片**（可点击选中）＋「提交」按钮；**提交一项后自动呈现下一项**；界面 UI 美观。
- **盲审=先隐去 twin 答案**（2_REFACTOR §10 反转设计）：渲染面在提交前绝不出现 twin 的 choice/reason。
- 提交后与 twin 裁决比对：**一致** → 直接确认；**不一致** → 展开「**原因批复**」多行文本框 ＋「**是否需恢复更改**」三选一（**需恢复/不需恢复/其他**；选「其他」再出多行文本框填具体要求）。
- 无未审核项或盲审处理完 → **已裁决事项列表**（新→旧），点击进**详情页**回顾裁决历史；详情页 UI 美观。
- 删除字样（用户点名）：「reviewer 附属队列 = …」「审查面（inspector 队列）超时项 = …」「审查入口：`orch review …`」（s3 已删数据面，UI 面再钉）；「Esc/← 返回主菜单」同禁。
- **偏差报告**（§5.4 全流程）：入 `mirror-queue.jsonl`（`kind="deviation"`，强制处理待遇）；选「**需恢复**」→ **TUI 直接 `taskbooks.create`** 建「偏差纠正」任务书（priority=high，正文含选择对照/原因/实施要求；落盘即 TASKBOOK-NEW 唤醒组长拆解派 devops）并回填 `taskbook_id`；「其他」只入队（requirements 随行）；「不需恢复」只入队留痕。

全局规范：图标后加空格；**成段文本框一律 Textual Markdown 渲染**（裁决详情=头卡＋Markdown）。

## 输入（只读）

- `src/pysci/skills/orchestration/tools/taste.py`（s1）：`blind_queue() -> [{case_id, ts, case_file, question, option_a, option_b, evidence, case_head}]`（**天然不含 choice/reason/body**；ts 升序）；`blind_case(case_id)`（含 choice 全记录，供比对）；`ruling_rows(limit=0)`（已复核新→旧：`{case_id, ts, choice, confidence, reason, review, night, authority, case_head}`）；`apply_review(case_id, *, verdict, note="", blind=None)`（review 落 `{ts, verdict, note, blind}`，`blind={user_choice, agree, remedy, requirements}`）。
- `src/pysci/skills/orchestration/tools/mirror.py`（s1）：`append_deviation(rec) -> rec`、`pending_deviations()`、`deviation_lines()`；`MIRROR_QUEUE_PATH` 行 schema：`{ts, kind:"deviation", source:"blind-review", status:"pending", text, case_id, user_choice, twin_choice, reason, remedy:"restore|none|other", requirements, taskbook_id?}`。
- `src/pysci/skills/orchestration/tools/taskbooks.py`：`create(title, body, source, priority)`（返回 state dict，`id` 可回填）。
- `src/pysci/skills/orchestration/tools/tui/{widgets,cards,app,theme}.py` 现状（`ChoiceCard`（`-selected` 高亮＋`Chosen(value)` 消息）、`RowListView`、`card()/section_title()/make_row(markup=False)`、`Markdown` 详情范式、注册四件套）。
- `data/others/original_designs/2_REFACTOR.md` §5.4（:200-213）/§10（:317-321）：盲审语义依据。

## 交付

### 1. 入口分派 `_open_review()`（menu「盲审与审查」Enter）

- `taste.pending_review_count() > 0` → **直接进 `#review-blind`**；否则 → `#review-rulings`。注册四件套＋`action_back`：`ruling-detail→review-rulings→menu`、`review-blind→menu`、`review-rulings→menu`。

### 2. 盲审界面 `#review-blind`（逐案状态机）

- 顶部 Static：「尚待盲审 N 项」（N 随进程递减）。
- 案情区：问题（标题级）＋证据（`Collapsible` 或 Markdown 区）＋ **选项大卡 A/B**（`ChoiceCard`，点击选中高亮，键盘可达）；**提交前画面不得出现 choice/reason/confidence**（数据源 `blind_queue()` 天然无答案——渲染面禁用 `blind_case()` 的答案字段直至提交）。
- 「提交」按钮（选中选项后可用）→ 比对 `blind_case(case_id)["choice"]`：
  - **一致** → `apply_review(verdict="confirmed", note="盲审与 twin 一致", blind={user_choice, agree: True})` → 自动呈现下一项（无下一项 → 已裁决列表）；
  - **不一致** → 展开追问区 `#blind-ask`：「① 原因批复」多行框 `#blind-reason` ＋「② 是否需恢复更改」三选一 `#blind-remedy`（**需恢复/不需恢复/其他**；选「其他」才显示 `#blind-requirements` 多行框）＋「确认」按钮。
- 追问确认 → `apply_review(verdict="rejected", note=<原因批复>, blind={user_choice, agree: False, remedy: "restore"|"none"|"other", requirements})` ＋ `mirror.append_deviation({...})`（text 形如「偏差报告 <case_id>：用户选 X vs twin 选 Y；处置=需恢复」）；**remedy=="restore"** → `taskbooks.create("偏差纠正：<case_id>", <正文：选择对照/原因批复/实施要求>, source="mirror-tui", priority="high")` 回填 `taskbook_id` → 自动下一项。
- 状态面（测试读）：`self.blind_case_id`、`self.blind_count`、`self.blind_selected`、`self.blind_ask_visible`、`self.blind_remedy` 等。

### 3. 已裁决列表 `#review-rulings` ＋ 详情 `#ruling-detail`

- 列表行（`RowListView`＋`make_row(markup=False)`）：复核结论旗标（confirmed/rejected）＋ ts ＋ case_head（截断）；序= `ruling_rows()` 序（新→旧）；空态「（暂无裁决记录）」。
- 详情页：**头卡**（case_id/ts/复核结论/night/authority）＋ **Markdown 正文**（案情 question/选项、twin 裁决（choice/confidence/reason——**详情页可显**，盲审已过）、你的盲审选择、原因批复、是否需恢复/具体要求、复核时刻）。
- UI 美观：卡/分区/Markdown 排版与全局一致（图标空格、`markup=False`）。

### 4. 清理

- 「盲审与审查」板块任何渲染面不得出现：「reviewer 附属队列」「审查面（inspector 队列）超时项」「审查入口」「Esc/← 返回」。
- `modules._review_detail` 文本面保留（徽标指纹/静态通道用），s3 已清禁语，不再回加。

### 5. 测试（新增 `tests/skills/orchestration/test_tui_review_blind.py`）

- 有待盲审 → 进 `review-blind` 且顶部计数正确；**渲染面不含 twin choice/reason 字样**（用带独特标记的假案卷钉住）。
- 一致提交 → `taste` 案卷 `review.verdict=="confirmed"`、自动进下一项；全部处理完 → `review-rulings`。
- 不一致 → `#blind-ask` 展开；选「其他」才出现 `#blind-requirements`；提交后 `mirror-queue.jsonl` 落 deviation 行（字段全）；「需恢复」额外落 `taskbooks` 新件且行内 `taskbook_id` 非空；「不需恢复/其他」不建任务书。
- 无待审进入 → 直接 `review-rulings`；裁决详情含 twin 答案与复核结论；回退链正确。
- 同步 `test_tui_user_paths.py`/`test_tui_visual_structure.py`（review 模块分支）。

## 白名单（写权限）

- 可写：src/pysci/skills/orchestration/tools/tui、tests/skills/orchestration

## 验收预期

1. `uv run --no-sync pytest tests` 全绿。
2. §5 新增测试全过（隐答案/两分支/三选一/偏差行/需恢复建任务书/已裁决列表与详情）。
3. 盲审界面美观自证：问题与选项为卡片级排版、选中态清晰、顶部计数实时。
4. 禁语钉：本板块渲染面无「reviewer 附属队列」「审查面（inspector 队列）超时项」「审查入口」「Esc/← 返回」。
5. 图标空格判据、常量单一出处判据仍绿。
