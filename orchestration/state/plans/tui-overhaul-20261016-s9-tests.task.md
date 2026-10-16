# 任务书 tui-overhaul-20261016 · s9-tests：测试收口（跨面一致性钉＋全量回归）

## 背景 / 目标

TUI 大改版 9 环节串行链**最后一环**。s1–s8 已全部合并 main（数据层缓存/家族归组/盲审读数/偏差队列 → theme/widgets 共享层 → modules 廉价指纹+图标空格 → app 键位收敛+滚动三路重算+概览板 → 任务书分组+Markdown 详情+VSCode → 计划列表+Tree+Markdown 联动 → 决策列表+三看板卡片化 → 盲审反转交互+偏差报告闭环；各环自测已随环落地：test_tui_taskbook_groups / test_tui_plans_panel / test_tui_decisions_panel / test_tui_boards_s7 / test_tui_review_blind 等）。

本环节**不加新功能**，只做测试收口：
1. **跨面一致性钉**（各环局部测过、缺全局缝合判据的）；
2. **全量回归**（连跑两次稳定）；
3. 清理测试自身债（如有）。

## 输入（只读）

- `tests/skills/orchestration/test_tui_*.py`（既有 15+ 文件）与 `src/pysci/skills/orchestration/tools/tui/`（app/modules/theme/widgets/cards/watermark）。
- 裁决语义文档 `data/others/original_designs/2_REFACTOR.md` §5.4/§10（如需补盲审语义断言）。

## 交付（新增 `tests/skills/orchestration/test_tui_overall_s9.py` 或按主题拆 2 个文件，含以下跨面钉）

1. **footer 键位矩阵全视图**：menu 广告 t/j；全部非 menu 视图（taskbooks/tb-detail/plans/plan-detail/decisions/decision-detail/review-blind/review-rulings/ruling-detail/journal/tb-new/tb-reject）不广告 t/j；`o` 只在 taskbooks；「广告的键必有可观察效果」对新增键全覆盖。
2. **滚动指示全覆盖**：`_SCROLL_PANES` 每个条目对应视图打开后位置行在场，且 `pane.scroll_y` 赋值（滚轮/滚动条路径）**立即**刷新指示；taskbooks 列表 `#tb-scroll` 同钉。
3. **禁语全仓钉**：全部可断言读面（`detail_text`/`detail_body`/`tb_detail_*`/`decision_detail_*`/`blind_*`/`ruling_*`/各列表 title）不含「Esc/← 返回」「批复入口」「口述动作」「推进：」「多 plan 并行」「reviewer 附属队列」「审查面（inspector 队列）超时项」「审查入口」。
4. **图标空格全仓钉**（如 s3 判据未覆盖 tui/ 全目录则补）：`tui/` 与 `tools/` 用户可见字符串无「图标紧贴文字」形态（豁免：`[▶]` 成对括号、句末）。
5. **徽标语义回归**（§2.2）：plans/system 指纹模块 `cursor()` 不调 `detail()`（防回归回全量重建）；system 倒计时行不进指纹（徽标不常亮）；`_mark_seen` 同 cursor 不写盘。
6. **导航一致性冒烟**：menu 七模块逐一 Enter→Esc 回菜单（含 plans/decisions/review 新分支），`self.view` 状态机全程正确、无崩溃。
7. **TUI 契约读面**：`tb_rows`/`tb_expanded`/`plan_rows`/`decision_rows`/`blind_count` 等状态面与渲染一致（各环已有局部钉，此处补一跳「重进视图状态重建正确」——展开态/选中态/计数在 Esc 往返后行为符合预期）。

## 白名单（写权限）

- 可写：src/pysci/skills/orchestration/tools/tui、tests/skills/orchestration

## 验收预期

1. `uv run --no-sync pytest tests -q` **全量绿且连跑两次稳定**（附两次输出计数）。
2. §1–7 跨面钉全过；无对裸下标/固定 id 顺序的脆弱依赖。
3. 交付正文附**测试总量变化**（前值 3815 passed → 新值）与各钉的行号/用例名清单，供组长终验对照 13 项需求。
