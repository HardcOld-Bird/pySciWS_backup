# 交付全文留档：devops-20261016-182406-tui-overhaul-20261016-s4-app-frame

- member: devops
- kind: result
- code: 0
- plan: tui-overhaul-20261016
- step: s4-app-frame
- task_file: orchestration/state/plans/tui-overhaul-20261016-s4-app-frame.task.md
- sid: 73d2478d-8fd1-4bb5-978f-65b55b82aeef
- duration_ms: 532636
- ts: 2026-10-16T18:33:01+08:00

## 交付正文

任务书 tui-overhaul-20261016-s4-app-frame（app.py 通用面）已完成：main merge commit **6cb462f**（feat commit d17c5d4）。push 限时 25s 超时（校园网/代理常态，已留痕 `state/push-log.jsonl`，按纪律忽略）。

**改动清单**（5 文件，+447/−128）：
1. `tui/app.py` —— footer 键位收敛（用户点名 t/j 只在主菜单）：
   `_ACTIONS_BY_VIEW` = menu:{taskbooks,journal}；detail:∅；taskbooks:{tb_new,tb_accept,tb_reject}；tb-detail:{tb_accept,tb_reject}；journal/tb-new/tb-reject:∅（`o` 键留给 s5 分组列表）。
2. `tui/app.py` —— 滚动位置指示接线（修复「滚轮/滚动条导航位置不更新」实测缺陷）：
   - `_SCROLL_PANES` 泛化为 `视图名 → (pane_id, mark_id)`（去掉正文 id），收编三视图 detail/tb-detail/**taskbooks**；
   - 新增 `PysciTUI.on_pane_scrolled(pane)`（s2 `ScrollMarkMixin.watch_scroll_y` 回调 → `_refresh_scroll_mark()`；只认当前视图面板，隐藏面板光标跟随滚动不串台）；
   - `compose`：`#detail-body`/`#tb-detail-body` → `MarkedVerticalScroll`、`#tb-list` → `MarkedList`，任务书列表底部新增 `#tb-scroll` 位置行（ListView 同报「整屏/顶/xx%（第 y/max 行）」）；`_scroll_widgets/_refresh_scroll_mark` 按新表适配，taskbooks 滚动态落新属性 `tb_scroll`/`tb_scroll_mark`；`_reset_scroll` 双拍校准保留并接入列表重建。
3. `tui/app.py` —— 提示机制退役：删除 `_DETAIL_HINT_MARKERS`/`_split_detail`；`detail_hint` 改由 `_render_detail` 生成（模块详情无专属操作→留空）、`tb_detail_hint` = 「a 接受　r 驳回」，**均不含 Esc 字样**（Esc 语义归 Footer 广告，消除「← 无绑定却出现在提示里」假广告）；`detail_text` 同步为 标题/正文/操作提示 三段结构。
4. `tui/app.py` —— 概览板美化（需求 13）：`#summary` 改主题形态（`_T_BORDER_SOFT` 细框＋`_T_BG`＋`_T_INSET`），新增 `#summary-title`（`_T_TITLE`）/`#summary-body`（`_T_BODY`）；`_render_summary` 首行「〔模块名〕概览」（数据/截行逻辑不动）。
5. `tui/app.py` —— `_mark_seen`：cursor 与已存水位相同则不写盘（倒计时每秒 tick 不再每秒重写 `mirror-seen.json`）；`action_back` 回退链与 `check_action` 语义保持不变。
6. 测试：`test_tui_visual_structure.py`（广告键矩阵改「菜单 t/j 有效果 + 详情不广告」；三区测试改「提示由 TUI 生成、读面禁语」）、`test_tui_detail_scroll.py`（滚动闸改 detail/tb-detail/taskbooks 三视图）、`test_tui_taskbook_detail.py`（tb-detail footer 改「a/r 广告、t/j 不广告」）；新增 `tests/skills/orchestration/test_tui_app_frame_s4.py`（10 测）。

**测试/验证证据**：
- `PYTHONPATH=<wt>/src uv run --no-sync pytest <wt>/tests -q` → **3757 passed, 42 skipped, 6 deselected**（2m57s；= 前值 3747 + 新增 10）；
- 验收②footer 钉：`test_menu_only_advertises_t_and_j` + `test_t_and_j_not_advertised_outside_menu`（`screen.active_bindings` 与 `check_action` 双面；「广告的键必有可观察效果」矩阵已按新语义重写并绿）；
- 验收③滚动钉：`test_scroll_y_assignment_updates_mark_immediately`（`pane.scroll_y = 7` 后指示立即变「第 7/max 行」）、`test_tb_list_scroll_mark_present_and_tracks_scroll`（`#tb-scroll` 在场，scroll_y 赋值与 PgDn 两路都实时更新）、`test_scroll_mark_ignores_hidden_pane`；键盘 PgUp/PgDn/Home/End/g/G 既有用例（test_tui_detail_scroll 全套）继续绿；
- 验收④禁语钉：`test_read_faces_have_no_prompt_words`（tb_detail_hint == 「a 接受　r 驳回」；读面不含「Esc/← 返回」「批复入口」「口述动作」）+ `test_hint_mechanism_retired_from_source`（两个标识符从 app.py 源码消失）；
- 验收⑤概览板：`test_summary_board_title_line_follows_selection`（`#summary` 首子为 `#summary-title`，随选中模块切换「〔甲模块〕概览」→「〔乙模块〕概览」）+ `test_summary_board_title_uses_real_module_title`（真 registry 首模块「〔任务书〕概览」）；常量单一出处判据（s2 目录扫描）仍绿；
- 验收⑤附带 `test_mark_seen_skips_write_when_cursor_unchanged`（spy `watermark.save_seen`：同 cursor 不重写）；
- ruff format/check 全过、pre-commit 全过；`pysci-dev worktree merge` 交集为空、会话数不减（76→76）、自动摘面。

**给后续环节的接口注记**（任务书要求各自环节注明）：s5–s8 新视图落地时——在 `_SCROLL_PANES` 追加 `视图名 → (pane 容器 id, 位置行 id)` 即自动获得滚动键闸控与实时位置指示；在 `action_back` 追加回退分支（如 plan-detail→plans）；分组列表 `o` 键加进 `_ACTIONS_BY_VIEW` 对应视图。
