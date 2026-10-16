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
_T_TITLE = "text-style: bold; color: $accent; background: $panel;"
_T_SUBTLE = "color: $text-muted;"
_T_BODY = "color: $text;"
_T_HINT = "color: $warning; text-style: italic;"
_T_HL = "background: $accent; text-style: bold;"
_T_BG = "background: $panel;"
