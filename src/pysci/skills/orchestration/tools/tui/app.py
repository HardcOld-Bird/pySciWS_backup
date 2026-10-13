"""`pysci` 裸名交互式控制台——主菜单框架（REFACTOR.md §2.2/§2.4/§2.5，M2）。

在 m1 骨架（模块列表 + ↑↓/Enter/Esc）之上长出**主菜单交互模型**（§2.2）与**更新水位**：
每模块一行「标题（左）＋ 更新徽标（右）」、选中项下方展开 3–5 行概览板、Enter 进全屏详情、
Esc/← 返回；**进入详情即写该模块水位**（真源 ``state/mirror-seen.json``，见
:mod:`.watermark`）。`journal` 的编号分流菜单迁进 TUI 口述动作（主菜单 `j` 与「待决与验收」
详情内的入口；偏好/指令二分法理不变）。

入口分派（§2.1）沿用 m1：``pysci <子命令> …`` → :func:`mirror.main` 静态通道（LLM 专用）；
``pysci``（无子命令）tty → 本 TUI；非 tty → 等价 ``pysci status``（复用 :func:`mirror.cmd_status`）。

数据层全在 :mod:`..mirror`（§2.5 不复制读数）；模块定义在 :mod:`.modules`。
"""

from __future__ import annotations

import sys

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Footer,
    Header,
    Input,
    ListItem,
    ListView,
    SelectionList,
    Static,
)
from textual.widgets.selection_list import Selection

from .. import mirror
from . import watermark
from .modules import Module, modules  # re-export：m1 的 `app.modules()` 仍在

#: 无更新时的灰点（§2.2：`● NEW` / 数字 / 灰点）。
_BADGE_NONE = "·"


class PysciTUI(App):
    """`pysci` 交互式控制台（主菜单 ↔ 详情 ↔ 口述）；可无头测试（tests 的 Pilot 用例）。"""

    TITLE = "pysci 控制台"
    BINDINGS = [
        ("escape", "back", "返回/退出"),
        ("j", "journal", "口述"),
        ("q", "quit", "退出"),
    ]
    CSS = """
    #menu { height: 1fr; border: round $accent; }
    #menu > ListItem.-highlight { background: $accent; text-style: bold; }
    .row { width: 100%; height: 1; }
    .row-title { width: 1fr; }
    .row-badge { width: auto; color: $warning; text-style: bold; }
    #summary { height: auto; padding: 0 1; color: $text-muted; }
    #detail { height: 1fr; padding: 0 1; }
    #journal { height: 1fr; padding: 0 1; border: round $warning; }
    #journal-result { color: $success; }
    """

    def __init__(self, *, mods: list[Module] | None = None, seen_path=None) -> None:
        super().__init__()
        self._mods = list(mods) if mods is not None else modules()
        self._seen_path = seen_path
        self._seen = watermark.load_seen(seen_path)
        # 可断言的状态面（测试不依赖 widget 内部）：视图 + 选中下标 + 详情正文 + 徽标/口述读数。
        self.view = "menu"  # "menu" | "detail" | "journal"
        self.selected = 0
        self.detail_text = ""
        self.journal_result = ""
        self.badge_texts: dict[str, str] = {
            m.key: self._badge_text(m) for m in self._mods
        }

    # ---- 组合 -------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="body"):
            yield ListView(*self._rows(), id="menu")
            yield Static("", id="summary")
            yield Static("", id="detail")
            with Vertical(id="journal"):
                yield Static(
                    "口述：这条内容是——（空格勾选①/②，回车在输入框里提交）",
                    id="journal-title",
                )
                yield SelectionList[int](
                    Selection(
                        "① 偏好陈述 → 决策日志（twin 训练语料；不要求行动）",
                        mirror.CHOICE_PREFERENCE,
                        False,
                    ),
                    Selection(
                        "② 行动指令 → 强制处理队列（组长须派 devops，不得 override）",
                        mirror.CHOICE_DIRECTIVE,
                        False,
                    ),
                    id="journal-select",
                )
                yield Input(
                    placeholder="口述内容…（Enter 记录；Esc 返回）", id="journal-text"
                )
                yield Static("", id="journal-result")
        yield Footer()

    def _rows(self) -> list[ListItem]:
        rows: list[ListItem] = []
        for m in self._mods:
            rows.append(
                ListItem(
                    Horizontal(
                        Static(m.title, classes="row-title"),
                        Static(
                            self.badge_texts.get(m.key, _BADGE_NONE),
                            classes="row-badge",
                            id=f"badge-{m.key}",
                        ),
                        classes="row",
                    ),
                    id=f"m-{m.key}",
                )
            )
        return rows

    def on_mount(self) -> None:
        self.query_one("#detail", Static).display = False
        self.query_one("#journal", Vertical).display = False
        self._render_summary(0)
        self.query_one("#menu", ListView).focus()

    # ---- 徽标与水位 -------------------------------------------------------
    def _badge_text(self, mod: Module) -> str:
        """`● NEW`（1 条）/ `● N`（N 条）/ 灰点（无）——数字与详情内容对得上（§2.2）。"""
        try:
            n = int(mod.updates_since(self._seen.get(mod.key, "")))
        except Exception:  # noqa: BLE001 - 徽标读数失败不该塌菜单
            n = 0
        if n <= 0:
            return _BADGE_NONE
        return "● NEW" if n == 1 else f"● {n}"

    def _refresh_badges(self) -> None:
        for m in self._mods:
            text = self._badge_text(m)
            self.badge_texts[m.key] = text
            try:
                self.query_one(f"#badge-{m.key}", Static).update(text)
            except Exception:  # noqa: BLE001 - 渲染面容忍缺件
                continue

    def _mark_seen(self, mod: Module) -> None:
        """进入详情即写水位（§2.2）。写失败不挡浏览（观察面 fail-open）。"""
        try:
            self._seen[mod.key] = mod.cursor()
            watermark.save_seen(self._seen, self._seen_path)
        except Exception:  # noqa: BLE001
            return
        self._refresh_badges()

    # ---- 交互 -------------------------------------------------------------
    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        """↑↓ 切换 → 记选中态 + 展开概览板（§2.2）。"""
        if self.view != "menu":
            return
        self.selected = int(event.list_view.index or 0)
        self._render_summary(self.selected)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Enter → 全屏详情视图。"""
        self._open_detail(int(event.list_view.index or 0))

    def action_back(self) -> None:
        """Esc/←：详情/口述 → 主菜单；主菜单 → 退出。"""
        if self.view in ("detail", "journal"):
            self._show_menu()
        else:
            self.exit()

    def action_journal(self) -> None:
        """主菜单 `j` → 口述动作（§2.4：journal 编号分流菜单迁入 TUI）。"""
        self._open_journal()

    # ---- 渲染 -------------------------------------------------------------
    def _render_summary(self, idx: int) -> None:
        mod = self._mods[idx]
        try:
            lines = [ln for ln in mod.summary() if ln][:5]  # §2.2：3–5 行概览板
        except Exception as exc:  # noqa: BLE001 - 读数坏一项不塌菜单
            lines = [f"读数失败：{type(exc).__name__}: {exc}"]
        self.query_one("#summary", Static).update("\n".join(lines) or "（无读数）")

    def _show_menu(self) -> None:
        self.view = "menu"
        self.detail_text = ""
        self.query_one("#detail", Static).display = False
        self.query_one("#journal", Vertical).display = False
        self.query_one("#menu", ListView).display = True
        self.query_one("#summary", Static).display = True
        self._refresh_badges()
        self.query_one("#menu", ListView).focus()

    def _open_detail(self, idx: int) -> None:
        mod = self._mods[idx]
        self.view = "detail"
        self.selected = idx
        try:
            body = list(mod.detail())
        except Exception as exc:  # noqa: BLE001
            body = [f"读数失败：{type(exc).__name__}: {exc}"]
        self.detail_text = "\n".join(
            [f"{mod.title}　详情", "", *body, "", "Esc/← 返回主菜单"]
        )
        self.query_one("#detail", Static).update(self.detail_text)
        self.query_one("#detail", Static).display = True
        self.query_one("#menu", ListView).display = False
        self.query_one("#summary", Static).display = False
        self.query_one("#journal", Vertical).display = False
        self._mark_seen(mod)

    def _open_journal(self) -> None:
        self.view = "journal"
        self.query_one("#menu", ListView).display = False
        self.query_one("#summary", Static).display = False
        self.query_one("#detail", Static).display = False
        self.query_one("#journal", Vertical).display = True
        self.query_one("#journal-result", Static).update("")
        inp = self.query_one("#journal-text", Input)
        inp.value = ""
        inp.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """口述提交：勾选编号（缺省 ①，与 CLI「回车=1」同义）→ ``mirror.route`` 分流落盘。"""
        if self.view != "journal":
            return
        text = (event.value or "").strip()
        if not text:
            self._set_journal_result("✗ 缺少口述内容（Enter 前请先输入）")
            return
        picks = list(self.query_one("#journal-select", SelectionList).selected)
        picks = [int(p) for p in picks] or [mirror.CHOICE_PREFERENCE]
        try:
            r = mirror.route(text, picks)
        except Exception as exc:  # noqa: BLE001
            self._set_journal_result(f"✗ 记录失败：{type(exc).__name__}: {exc}")
            return
        marks: list[str] = []
        if r.get("preference"):
            marks.append("① 偏好 → 决策日志（训练语料）")
        if r.get("directive"):
            marks.append("② 行动指令 → 强制队列（须派发）")
        self._set_journal_result("已记录：" + "；".join(marks))

    def _set_journal_result(self, text: str) -> None:
        self.journal_result = text
        self.query_one("#journal-result", Static).update(text)


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
