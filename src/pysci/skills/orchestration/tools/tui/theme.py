"""TUI 主题常量——**唯一出处**（f2 美学统一；s2 起从 app.py 迁出）。

全 TUI 面板（app.py 的 CSS 与 widgets.py 的 DEFAULT_CSS）一律 **f-string 插值引用**
本文件常量：同一视觉参数在 ``tui/`` 全目录源码中只允许硬编码一次（tests/
test_tui_visual_structure.py 的运行期拼接 grep 判据钉住）——各面板各写一套就是
「美学不一致」的根。
"""

from __future__ import annotations

_T_BORDER = "border: round $accent;"
_T_BORDER_SOFT = "border: round $warning;"
_T_INSET = "padding: 0 1;"
#: 标题条用色：**亮黄**（高对比）。原用 `$accent`（橙）——实测标题面最刺眼的是
#: Markdown 各级标题（Textual 默认 `$primary` 蓝，深底几乎不可辨，见 _T_HEADING）。
_T_TITLE = "text-style: bold; color: $warning; background: $panel;"
_T_SUBTLE = "color: $text-muted;"
_T_BODY = "color: $text;"
_T_HINT = "color: $warning; text-style: italic;"
_T_HL = "background: $accent; text-style: bold;"
_T_BG = "background: $panel;"

#: 高亮底（`_T_HL` 橙）上的**前景色**：橙底上白字与橙色徽标都不可辨（用户 2026-10-16
#: 实机反馈：「选中后标题白色难辨、状态徽标直接不可见」）——选中态一律压深色。
_T_HL_FG = "color: black;"

#: Markdown 各级标题（h1–h6）覆写用色：Textual 默认 h1–h3 是 `$primary`（#0178D4 蓝），
#: 深灰蓝底上几乎不可辨（用户点名「全流程报告：…」「任务概述」等标题）——统一亮黄。
_T_HEADING = "color: $warning; text-style: bold;"

#: 复选标记（Checkbox/SelectionList 共用；用户 2026-10-17「要对勾不要错叉」）。
#: Textual 8.2.8 的 ToggleButton/SelectionList **无论选中与否都渲染 `▐X▌`**（选中只靠
#: 颜色区分——实测 `str(render())` 两态同字形）→ 改**值感知字形**：选中 ``√``、未选
#: 空白。实现见 widgets.Check / widgets.CheckList（1 字符宽与原 X 同宽，无布局漂移）。
_T_CHECK_ON = "√"
_T_CHECK_OFF = " "
