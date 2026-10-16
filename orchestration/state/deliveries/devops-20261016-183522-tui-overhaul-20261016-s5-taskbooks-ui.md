# 交付全文留档：devops-20261016-183522-tui-overhaul-20261016-s5-taskbooks-ui

- member: devops
- kind: result
- code: 0
- plan: tui-overhaul-20261016
- step: s5-taskbooks-ui
- task_file: orchestration/state/plans/tui-overhaul-20261016-s5-taskbooks-ui.task.md
- sid: 73d2478d-8fd1-4bb5-978f-65b55b82aeef
- duration_ms: 3164042
- ts: 2026-10-16T19:28:09+08:00

## 交付正文

任务书 tui-overhaul-20261016-s5-taskbooks-ui（任务书返工家族分组列表＋Markdown 详情＋VSCode）已完成：main merge commit **e73429e**（feat commit b710162），push 成功（c657a0f..e73429e）。全量测试 **3772 passed × 连跑两次稳定**。

**改动清单**（10 文件，+768/−143）：

1. `tui/app.py` —— 分组列表（`#taskbook`）：
   - **行模型** `self.tb_rows: list[dict]`（`group`：tb_id=最新项/root/title/badge/size/round；`sub`：tb_id/root/label/indent/badge/title）＋ 展开态 `self.tb_expanded: dict[root,bool]`（**初始全 False=收起**，主列表一屏只见每组最新/本体）；`#tb-list` **只按行模型重建**，测试禁裸下标（新增 `_row_index` 语义：一切目标解析走 `_tb_target_id()`）。
   - 渲染走 s2 `make_row`：组行＝徽标＋标题＋右 meta「共 N 件」（N=1 →「本体」）；次级行＝缩进 `└ `＋`family_label`＋徽标＋标题，**从新到旧、本体殿后**（members 序）。
   - 交互（s2 `RowListView.RowActivated` 分派，替换旧 `on_list_view_selected` 的 tb-list 分支；菜单列表不动）：组行 **Enter=立即展开/收起**、**单击=延迟 0.35s 切换**（`set_timer`，双击到达即取消）、**双击(chain≥2)=取消延迟＋直接进最新项详情**；**次级行单击/Enter=立即进该项详情**；新增 **`o` 键**（BINDING `tb_open_latest`，入 `_ACTIONS_BY_VIEW["taskbooks"]` 与 `_GATED_ACTIONS`）＝进选中行目标详情。
   - 展开/收起与批复后**高亮保持在被操作家族组行**（`keep_root`）；`#tb-scroll` 随重建复位（复用 s4 `_reset_scroll`）；`#tb-title` 改成一行操作说明。
   - 关键实现点：`ListView.clear()` 是 **AwaitComplete**（异步移除）——立即 append 会撞 `DuplicateIds`（实测），故任务书面板改 **async 链路**（`action_taskbooks`/`on_list_view_selected`/`action_back`/`action_tb_accept|reject`/`_tb_act`/`on_button_pressed`/`_submit_new_taskbook`/`_submit_reject`/`on_row_activated`/`_open_taskbooks`/`_rebuild_tb_list`/`_toggle_tb_group` 均 async，`await lv.clear()` 后再挂行、再定 index）。新增私有 `_TbList(ScrollMarkMixin, RowListView)` 组合 s2 两个 mixin（滚动知情＋行激活归一）。
2. `tui/app.py` —— tb-detail Markdown：`#tb-detail-body` 内嵌 `Markdown(id="tb-detail-md")`（外层 `MarkedVerticalScroll` 不变），喂 `taskbooks.detail_markdown(tid)`；头卡保留（徽标＋tid＋标题，CSS 走 `_T_TITLE/_T_BORDER/_T_INSET`）；读面 `self.tb_detail_body` = Markdown 文本；新增 `#tb-open-vscode` 按钮＋`#tb-detail-msg` 消息行（`shutil.which("code")` 缺失 / `subprocess.Popen` 异常 →「[!] 打开失败：…」；成功 →「[√] 已在 VSCode 打开 <路径>」，`Popen` 不阻塞 UI）；`tb_detail_hint` 维持「a 接受　r 驳回」。
3. `tui/widgets.py`（s2 共享层**增量**扩展 `ScrollMarkMixin`）：新增 `watch_virtual_size`（Markdown 异步渲染/条目增减后内容尺寸变化）与 `on_resize`（面板隐藏→显示/窗口缩放时 `max_scroll_y` 变化）→ 同一 fail-soft 通知通道，修复「位置指示在内容/面板尺寸落定后不重算」的实测缺陷（详见下方偶发排查）。
4. 测试：新增 `tests/skills/orchestration/test_tui_taskbook_groups.py`（15 测：收起/展开收起/单击延迟/双击不触发切换/`o`/次级行 Enter/a·r 目标/批次返工挂接/Markdown 批复参考段/VSCode 三分支/状态面与渲染一致）；同步 `test_tui_taskbook_detail.py`、`test_tui_tb_defects_fix.py`、`test_taskbooks.py`、`test_rework_close.py`（行模型＋`o` 键＋await）、`test_tui_visual_structure.py`（`_each_view` 用 tb_rows＋`o`）、`test_tui_detail_scroll.py`（`o` 键＋Markdown 高度余量）、`test_tui_app_frame_s4.py`（`o` 键）。

**测试/验证证据**：
- `PYTHONPATH=<wt>/src uv run --no-sync pytest <wt>/tests -q` → **3772 passed, 42 skipped, 6 deselected**，**连跑两次均全绿**（3m14s / 3m10s）；
- TUI/任务书相关子集 **268 passed × 四连跑**全绿；
- 验收②：`test_double_click_opens_latest_detail_without_toggling`（`Pilot.click(times=2)` → `view=="tb-detail"` 且 `tb_detail_id==最新项`，且不触发单击展开）、`test_single_click_toggles_after_delay`、`test_o_key_opens_highlighted_target`、`test_sub_row_enter_opens_that_item` 全过；
- 验收③：s3 图标空格判据仍绿；tb-detail 读面无「Esc/← 返回」（s5 新测试与 s4 禁语钉均绿）；
- 验收④：`test_state_face_matches_rendered_list` 钉「`tb_rows`/`tb_expanded` 与 `#tb-list` 实际条目数逐态一致」；
- **偶发排查（重要）**：全量跑中抓到两处间歇失败——① `test_tb_detail_short_content_says_whole_screen`（位置行显示「顶（第 0/75 行）」而非「整屏」）：根因是**产品缺口**——Markdown 异步渲染/面板由隐藏转显示后内容与尺寸落定，`max_scroll_y` 变了但没人重算位置指示；已用 `ScrollMarkMixin.watch_virtual_size + on_resize` 从源头修掉（修后该用例 10/10 稳定）；② `test_vscode_button_success_passes_md_path`：根因是**我新测试自身的缺陷**（helper 内部重复建家族，两次建件跨秒时 id 不同）——已改为一律返回实际打开的 tid（修后 10/10 稳定）；
- ruff format/check 全过、pre-commit 全过；`pysci-dev worktree merge` 交集为空、会话数不减（76→76）、自动摘面。

**给 s6–s8 的接口注记**（任务书要求各自注明）：新视图落地时——① `_SCROLL_PANES` 追加 `视图名 → (pane容器id, 位置行id)` 即自动获得滚动键闸控＋实时位置指示（`ScrollMarkMixin` 一族已覆盖滚动/内容尺寸/面板尺寸三路）；② `action_back` 追加回退分支（如 plan-detail→plans）；③ 任务书列表如需「`o` 打开」同族语义，参照 `_tb_target_id()` 的单点目标解析；④ 列表类视图重建请走 **async + `await lv.clear()`**（否则 DuplicateIds）。
