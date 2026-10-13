"""`pysci` 裸名交互式控制台的 TUI 层（REFACTOR.md §2.5：`tools/tui/`）。

`app.py` 是门面：入口 `main()` 按 §2.1 分派（子命令→静态通道 / tty→TUI / 非 tty→status），
`modules()` 供主菜单使用、读数复用 `..mirror` 的纯函数（同一能力不留两份实现）。
"""
