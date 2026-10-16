# 交付全文留档：devops-20261016-193042-tui-overhaul-20261016-s6-plans-ui

- member: devops
- kind: result
- code: 0
- plan: tui-overhaul-20261016
- step: s6-plans-ui
- task_file: orchestration/state/plans/tui-overhaul-20261016-s6-plans-ui.task.md
- sid: 73d2478d-8fd1-4bb5-978f-65b55b82aeef
- duration_ms: 1495002
- ts: 2026-10-16T19:55:40+08:00

## 交付正文

任务书 tui-overhaul-20261016-s6-plans-ui（计划与派发：列表→详情：可折叠 Tree＋Markdown）已完成：main merge commit **0799ce2**（feat commit 7547c8b），push 成功（e73429e..0799ce2）。全量 **3782 passed × 连跑两次稳定**。

**改动清单**（6 文件，+707/−31）：

1. `tui/app.py` —— **`#plans` 列表视图**：menu「计划与派发」Enter → 本视图（`on_list_view_selected` 按模块 key 分派；`queues/system/pending` 仍走通用 `#detail`）。行模型 `self.plan_rows`（**只走 `plans.overview_rows()`＝state.json 缓存**，不解析 YAML）；行＝`make_row(mark, "id  done/total", meta=goal[:40])`（徽标图标带空格）；序＝新→旧；async 重建（`await lv.clear()`，s5 范式）；空态「（暂无计划）」；`#plans-title` 一行操作说明（↑↓ 选｜Enter 看详情｜Esc 返回）；`#plans-scroll` 位置行。
2. `tui/app.py` —— **`plan-detail` 视图**（上→下）：**头卡**（`card()` 形态 Static：goal／状态徽标／进度 done/total／created／completed／abandoned）＋ **可折叠 `MarkedTree`**（数据 `plans.plan_tree_nodes(pid)`：superstep label `〔serial|parallel〕 id`；叶子 `图标 id [status] member｜brief`，done→✔/failed→✗/其余·；**默认全展开**）＋ **`Markdown(id="plan-detail-md")` 联动面板**（树节点选中→渲染该节点详情：id/路径/类型/状态/成员/brief 全文＋叶子运行留痕 任务书/会话/job/evidence/复核；superstep 附子节点概览含点路径）。默认选中＝**当前进行中环节**（无则首叶/根）；焦点给树（↑↓ 走树光标＝联动；PgUp/PgDn/Home/End 归全局滚动，Tree 不绑它们）；`_SCROLL_PANES["plan-detail"]=("#plan-detail-body","#plan-detail-scroll")`、`#plan-detail-title`=plan id、`action_back`：plan-detail→plans→menu；`_hide_all_views` 收编两新容器。
3. `tui/app.py` —— 三处踩坑修正（均被测试或冒烟抓到）：① **`on_mount` 漏藏新面板**（两面板 `height:1fr` 抢走布局，`#detail-body` 只剩 1 行 → 既有滚动测试立刻变红「翻页 200 次未到底」）：改为统一 `_hide_all_views()` 再显菜单；② **Rich markup 吃掉 label 方括号**（`✔ s1 [failed] …` 渲染成 `✗ s1  …`）：Tree label 改传 `rich.text.Text`；③ **`select_node` 在首帧前静默无效**（行号未算，光标落根）：默认选中延后到 `call_after_refresh`。另新增 `_widget_id()`（plan id 可能含点号如 `…-20261015.plan`，Textual widget id 非法 → 折 `-`；任务书列表条目同用）。
4. 测试：新增 `tests/skills/orchestration/test_tui_plans_panel.py`（10 测：列表序/内容/空态、头卡、Tree 节点数与 `plan_tree_nodes` 一致＋label 形态、Markdown 选中联动、superstep 子节点概览、缺 state/unknown plan 不崩、**性能钉**、**禁语钉**、md 组件在场）；同步 `test_tui_spike.py`（plans→plans 视图 + queues 仍 detail）、`test_tui_visual_structure.py`（广告键矩阵/反例/全走一遍：plans 分支）、`test_tui_user_paths.py`（模块遍历跳过 plans、index 1→2）、`test_tui_app_frame_s4.py`（index 1→2）。

**测试/验证证据**：
- `PYTHONPATH=<wt>/src uv run --no-sync pytest <wt>/tests -q` → **3782 passed, 42 skipped, 6 deselected**，**连跑两次均全绿**（5m39s / 5m24s）；TUI 相关子集 **280 passed × 两连**；
- 验收②：新增 10 测全过（含「Tree 节点数与 `plan_tree_nodes` 一致」「选中节点 → 面板含该节点 brief」）；
- 验收③**手感自证（46 plan 量级）**：临时造 46 个 plan 实测 `打开 plans 视图 ≈ 304 ms`（含 Pilot pause 一轮事件循环）、`进详情（含 YAML 树）≈ 243 ms`，无卡顿；性能钉以「`_load_yaml`/`plan_tree_nodes` 抛异常后列表仍正常打开」实证列表只走 state.json 缓存；
- 验收④：图标空格判据（s3 测试）与常量单一出处判据（s2 目录扫描）仍绿；
- ruff format/check 全过、pre-commit 全过；`pysci-dev worktree merge` 交集为空、会话数不减（76→76）、自动摘面。

**给 s7–s8 的接口注记**：① 新视图注册三件套＝compose 容器 + `_SCROLL_PANES[视图]=(pane_id, mark_id)` + `action_back` 分支（再在 `_hide_all_views` 名单加上容器——**漏了会让隐藏面板照样占布局**，本环节实锤）；② 列表类复用 `_RowMarkedList`（`ScrollMarkMixin`＋`RowListView`，滚动位置三路重算已内建）与 async `await lv.clear()` 重建范式；③ 树/selections 类若要在首帧前定位光标，务必延后到 `call_after_refresh`；④ 任何用户数据进 widget id 前过 `_widget_id()`。
