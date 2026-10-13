"""`pysci` 裸名交互式控制台——TUI 骨架（REFACTOR.md §2.1/§2.2/§2.3/§2.5，M1 spike）。

**这是 m2 的地基，不是一次性脚本**：本环节只做最小骨架（模块列表 + ↑↓ 选择高亮 + Enter 进
详情占位 + Esc 返回），但结构按蓝图定位——数据层复用 :mod:`..mirror` 的纯读数函数（§2.5：
同一能力不留两份实现），TUI 只做表现；详情视图的具体工作入口是 m2 的事。

入口分派（§2.1）：
- ``pysci <subcommand> …`` → 转交 :func:`mirror.main` 的静态命令面（LLM 专用通道，本环节不动）；
- ``pysci``（无子命令）+ **tty** → 本 TUI；
- ``pysci``（无子命令）+ **非 tty** → 等价 ``pysci status`` 静态输出（复用
  :func:`mirror.cmd_status`，防 agent 会话误入交互）。

交互模型（§2.2）：主菜单每模块一行（标题；**更新徽标**与展开动画是 m2）；↑↓ 切换、选中项
下方展开 3–5 行概览板；Enter 进全屏详情、Esc/← 返回主菜单。
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Footer, Header, ListItem, ListView, Static

from .. import mirror


@dataclass(frozen=True)
class Module:
    """主菜单一行：稳定 ``key`` + ``title`` + 概览行构造器（复用 mirror 纯读数函数）。"""

    key: str
    title: str
    summary: Callable[[], list[str]]


def modules() -> list[Module]:
    """主菜单七模块（REFACTOR.md §2.4）；概览行全部复用 mirror 的现成读数，不重写。"""
    return [
        # 1 任务书 / 2 计划与派发：数据面（§3.4 队列、active plan）m2 建，先占位。
        Module(
            "taskbooks",
            "任务书",
            lambda: ["（m2）大任务列表 / 新增多行输入 / 批复返工"],
        ),
        Module(
            "plans",
            "计划与派发",
            lambda: ["（m2）active plan 清单 / 环节进度 / 多 plan 并行"],
        ),
        Module(
            "queues",
            "队列与运行",
            lambda: [mirror._line_queues(), mirror._line_defer()],
        ),
        Module("review", "盲审与审查", lambda: [mirror._line_review()]),
        Module(
            "system",
            "系统与路由",
            lambda: [
                mirror._line_switches(),
                mirror._line_routing(),
                mirror._line_jury(),
                mirror._line_fallback(),
            ],
        ),
        Module(
            "decisions",
            "决策审计",
            lambda: [mirror._line_inspector_decisions(), mirror._line_trend()],
        ),
        Module(
            "pending",
            "待决与验收",
            lambda: [mirror._line_pending(), mirror._line_acceptance()],
        ),
    ]


def _menu_label(mod: Module) -> str:
    """主菜单一行文案。§2.2 的**更新徽标**（`state/mirror-seen.json` 水位）是 m2 的事；
    骨架先只出标题（不伪造徽标）。"""
    return mod.title


class PysciTUI(App):
    """`pysci` 交互式控制台骨架：模块列表 ↔ 详情视图（可无头测试：见 tests 的 Pilot 用例）。"""

    TITLE = "pysci"
    BINDINGS = [("escape", "back", "返回/退出"), ("q", "quit", "退出")]
    CSS = """
    #menu { height: 1fr; }
    #summary { height: auto; padding: 0 1; color: $text-muted; }
    #detail { height: 1fr; padding: 0 1; }
    """

    def __init__(self, *, mods: list[Module] | None = None) -> None:
        super().__init__()
        self._mods = list(mods) if mods is not None else modules()
        # 可断言的状态面（测试不依赖 widget 内部）：当前视图 + 选中下标 + 详情正文。
        self.view = "menu"  # "menu" | "detail"
        self.selected = 0
        self.detail_text = ""

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="body"):
            yield ListView(
                *[
                    ListItem(Static(_menu_label(m)), id=f"m-{m.key}")
                    for m in self._mods
                ],
                id="menu",
            )
            yield Static("", id="summary")
            yield Static("", id="detail")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#detail", Static).display = False  # 详情初始隐藏
        self._render_summary(0)
        self.query_one("#menu", ListView).focus()

    # ---- 交互 -------------------------------------------------------------
    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        """↑↓ 切换 → 记选中态 + 在选中项下方展开概览板（§2.2）。"""
        if self.view != "menu":
            return
        self.selected = int(event.list_view.index or 0)
        self._render_summary(self.selected)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Enter → 全屏详情视图。"""
        self._open_detail(int(event.list_view.index or 0))

    def action_back(self) -> None:
        """Esc/← ：详情 → 主菜单；主菜单 → 退出。"""
        if self.view == "detail":
            self.view = "menu"
            self.detail_text = ""
            self.query_one("#detail", Static).display = False
            self.query_one("#menu", ListView).display = True
            self.query_one("#summary", Static).display = True
            self.query_one("#menu", ListView).focus()
        else:
            self.exit()

    # ---- 渲染 -------------------------------------------------------------
    def _render_summary(self, idx: int) -> None:
        mod = self._mods[idx]
        lines = [ln for ln in mod.summary() if ln][:5]  # §2.2：3–5 行概览板
        self.query_one("#summary", Static).update("\n".join(lines) or "（无读数）")

    def _open_detail(self, idx: int) -> None:
        mod = self._mods[idx]
        self.view = "detail"
        self.selected = idx
        self.detail_text = "\n".join(
            [f"{mod.title}　详情（m2 细化）", *mod.summary(), "", "Esc/← 返回主菜单"]
        )
        self.query_one("#detail", Static).update(self.detail_text)
        self.query_one("#detail", Static).display = True
        self.query_one("#menu", ListView).display = False
        self.query_one("#summary", Static).display = False


def _is_tty() -> bool:
    """交互面可用性：stdin 与 stdout **都是** tty 才算（管道/重定向下绝不阻塞等输入）。"""
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except (AttributeError, ValueError):
        return False


def main(argv: list[str] | None = None) -> int:
    """`pysci` 入口——按 §2.1 分派（子命令静态通道 / tty→TUI / 非 tty→status）。"""
    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        # 有子命令（status/journal/decisions/…）：静态命令面，本环节原样转交 mirror。
        return mirror.main(args)
    if _is_tty():
        PysciTUI().run()
        return 0
    # 非 tty（agent/管道/重定向）：等价 `pysci status`，复用 mirror.cmd_status，零交互。
    return mirror.main(["status"])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
