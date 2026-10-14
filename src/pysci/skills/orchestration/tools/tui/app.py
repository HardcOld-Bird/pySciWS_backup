"""`pysci` 裸名交互式控制台——主菜单框架（README §12，M2）。

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
    Button,
    Footer,
    Header,
    Input,
    ListItem,
    ListView,
    SelectionList,
    Static,
    TextArea,
)
from textual.widgets.selection_list import Selection

from .. import mirror
from . import watermark
from .modules import Module, modules  # re-export：m1 的 `app.modules()` 仍在

#: 无更新时的灰点（§2.2：`● NEW` / 数字 / 灰点）。
_BADGE_NONE = "·"

#: 各视图下**有可观察效果**的 action（f1：footer 只广告这些，广告的键必须真有效果）。
#: 未列出的受管 action 在当前视图返回 `False`（Textual 语义：禁用**且不显示**）。
#: **输入框视图（journal / tb-new）留空**：焦点在 `Input` 上时可打印键属于文本输入
#: （实测 `screen.active_bindings` 不含 `t`/`n`/`a`/`r`），标成「可用」是撒谎——
#: 用户打字要的是字符，不是快捷键。
_ACTIONS_BY_VIEW: dict[str, frozenset[str]] = {
    "menu": frozenset({"taskbooks", "journal"}),
    "detail": frozenset({"taskbooks", "journal"}),
    "journal": frozenset(),
    "taskbooks": frozenset(
        {"taskbooks", "tb_new", "tb_accept", "tb_reject", "journal"}
    ),
    "tb-new": frozenset(),
}

#: 受本类视图闸管的 action（其余——`back`/`quit` 与 widget 内部动作——一律放行，不干预导航）。
_GATED_ACTIONS: frozenset[str] = frozenset(
    {"taskbooks", "tb_new", "tb_accept", "tb_reject", "journal"}
)

# ---------------------------------------------------------------------------
# 主题常量（f2 美学统一）：**一处定义**，全 TUI 面板的 CSS 一律插值引用。
# 同一视觉参数在本文件里只允许出现一次（tests/test_tui_visual_structure.py 用运行期拼接
# 的 grep 判据钉住）——各面板各写一套就是「美学不一致」的根。
# ---------------------------------------------------------------------------
_T_BORDER = "border: round $accent;"
_T_BORDER_SOFT = "border: round $warning;"
_T_INSET = "padding: 0 1;"
_T_TITLE = "text-style: bold; color: $accent; background: $panel;"
_T_SUBTLE = "color: $text-muted;"
_T_BODY = "color: $text;"
_T_HINT = "color: $warning; text-style: italic;"
_T_HL = "background: $accent; text-style: bold;"
_T_BG = "background: $panel;"

#: 详情页里「操作提示行」的辨识标记（渲染分区用：这些行从读数区挪进提示区，**不改数据层**）。
_DETAIL_HINT_MARKERS = (
    "按 `t`",
    "Enter / `t`",
    "推进：",
    "多 plan 并行",
    "审查入口",
    "批复入口",
    "口述动作",
)


def _split_detail(lines: list[str]) -> tuple[list[str], list[str]]:
    """把详情行分成 **读数区** 与 **操作提示区**（只在渲染时分，数据层原样不动）。

    尾部连续命中 :data:`_DETAIL_HINT_MARKERS` 的行归提示区；一行都不命中则提示区只有
    「Esc/← 返回主菜单」。
    """
    body = list(lines)
    hints: list[str] = []
    while body and any(m in body[-1] for m in _DETAIL_HINT_MARKERS):
        hints.insert(0, body.pop())
    return body, hints


class PysciTUI(App):
    """`pysci` 交互式控制台（主菜单 ↔ 详情 ↔ 口述）；可无头测试（tests 的 Pilot 用例）。"""

    TITLE = "pysci 控制台"
    BINDINGS = [
        ("escape", "back", "返回/退出"),
        ("t", "taskbooks", "任务书"),
        ("n", "tb_new", "新增任务书"),
        ("a", "tb_accept", "接受选中"),
        ("r", "tb_reject", "驳回选中"),
        ("j", "journal", "口述"),
        ("q", "quit", "退出"),
    ]
    CSS = f"""
    #menu {{ {_T_BORDER} height: 1fr; {_T_BG} }}
    #menu > ListItem.-highlight {{ {_T_HL} }}
    .row {{ width: 100%; height: 1; }}
    .row-title {{ width: 1fr; }}
    .row-badge {{ width: auto; color: $warning; text-style: bold; }}
    #summary {{ height: auto; {_T_INSET} {_T_SUBTLE} }}
    #detail {{ height: 1fr; {_T_INSET} {_T_BORDER} {_T_BG} }}
    #detail-title {{ {_T_TITLE} }}
    #detail-body {{ {_T_BODY} height: 1fr; }}
    #detail-hint {{ {_T_HINT} margin-top: 1; }}
    #journal {{ height: 1fr; {_T_INSET} {_T_BORDER} {_T_BG} }}
    #journal-title {{ {_T_TITLE} }}
    #journal-result {{ color: $success; }}
    #taskbook {{ height: 1fr; {_T_INSET} {_T_BORDER} {_T_BG} }}
    #tb-title {{ {_T_TITLE} }}
    #tb-list {{ height: 1fr; }}
    #tb-list > ListItem.-highlight {{ {_T_HL} }}
    #tb-msg {{ color: $success; }}
    #tb-new {{ height: 1fr; {_T_INSET} {_T_BORDER_SOFT} {_T_BG} }}
    #tb-new-title {{ {_T_TITLE} }}
    #tb-new-body {{ height: 1fr; }}
    #tb-new-msg {{ color: $success; }}
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
        # 详情三区（f2 分区渲染）——textual 8 的 Static 没有 .renderable，测试读这里
        self.detail_title = ""
        self.detail_body = ""
        self.detail_hint = ""
        self.journal_result = ""
        self.tb_items: list[dict] = []
        self.tb_msg = ""
        self.tb_new_msg = ""
        self.last_created = ""
        self.badge_texts: dict[str, str] = {
            m.key: self._badge_text(m) for m in self._mods
        }

    # ---- 组合 -------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="body"):
            yield ListView(*self._rows(), id="menu")
            yield Static("", id="summary")
            with Vertical(id="detail"):
                yield Static("", id="detail-title")
                yield Static("", id="detail-body")
                yield Static("", id="detail-hint")
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
            with Vertical(id="taskbook"):
                yield Static(
                    "任务书：↑↓ 选；n 新增；a 接受选中（delivered→accepted）；"
                    "r 驳回选中（批语取下方输入框）",
                    id="tb-title",
                )
                yield ListView(id="tb-list")
                yield Input(placeholder="驳回批语（按 r 驳回选中项）", id="tb-note")
                yield Static("", id="tb-msg")
            with Vertical(id="tb-new"):
                yield Static(
                    "新增任务书（标题 + 多行正文；点「投递」落盘）", id="tb-new-title"
                )
                yield Input(placeholder="标题", id="tb-new-title-input")
                yield TextArea(id="tb-new-body")
                yield Button("投递", id="tb-new-submit")
                yield Static("", id="tb-new-msg")
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
        self.query_one("#detail", Vertical).display = False
        self.query_one("#journal", Vertical).display = False
        self.query_one("#taskbook", Vertical).display = False
        self.query_one("#tb-new", Vertical).display = False
        self._render_summary(0)
        self.query_one("#menu", ListView).focus()
        # 详情倒计时**每秒跳动**（「距下次切换还有 X 分 X 秒」，用户 2026-10-14 实测反馈）：
        # 只重算详情文本、不走菜单/徽标面——徽标指纹用**去掉倒计时行**的稳定读数，
        # 否则每秒一次的读数变化会让徽标常亮（§2.2「数字与详情对得上」）。
        self.set_interval(1.0, self._tick_detail)

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
        """Enter：主菜单 → 详情；「任务书」模块 → **专用面板**（＝`t`，f1 用户路径修复：
        菜单→Enter→`n` 必须可用）；任务书列表 → 看正文（批复走 `a`/`r`）。"""
        if event.list_view.id == "menu":
            idx = int(event.list_view.index or 0)
            if self._mods[idx].key == "taskbooks":
                self._open_taskbooks()
            else:
                self._open_detail(idx)
        elif event.list_view.id == "tb-list":
            self._tb_show_body(int(event.list_view.index or 0))

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool:
        """footer 键位广告与守卫一致化（f1）：受管 action 在当前视图无效果 → 返回 `False`
        （Textual 语义：`False` = 禁用且**不显示**于 Footer；`None` 只是禁用但仍显示）。
        其余 action（`back`/`quit` 与 widget 内部导航/提交）一律放行，绝不干预光标与输入。"""
        if action not in _GATED_ACTIONS:
            return True
        return action in _ACTIONS_BY_VIEW.get(self.view, frozenset())

    def _tb_show_body(self, idx: int) -> None:
        from .. import taskbooks as _tb

        if idx >= len(self.tb_items):
            return
        tid = str(self.tb_items[idx].get("id"))
        self._set_tb_msg(f"—— {tid} 正文 ——\n{_tb.read_body(tid)[:400]}")

    def action_back(self) -> None:
        """Esc/←：任何子视图（详情/口述/任务书）→ 主菜单；主菜单 → 退出。"""
        if self.view != "menu":
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

    def _hide_all_views(self) -> None:
        for sel in (
            "#menu",
            "#summary",
            "#detail",
            "#journal",
            "#taskbook",
            "#tb-new",
        ):
            try:
                self.query_one(sel).display = False
            except Exception:  # noqa: BLE001 - 组合面容忍缺件
                continue

    def _show_menu(self) -> None:
        self.view = "menu"
        self.detail_text = ""
        self.detail_title = ""
        self.detail_body = ""
        self.detail_hint = ""
        self._hide_all_views()
        self.query_one("#menu", ListView).display = True
        self.query_one("#summary", Static).display = True
        self._refresh_badges()
        self.query_one("#menu", ListView).focus()

    def _open_detail(self, idx: int) -> None:
        mod = self._mods[idx]
        self.view = "detail"
        self.selected = idx
        try:
            raw = list(mod.detail())
        except Exception as exc:  # noqa: BLE001
            raw = [f"读数失败：{type(exc).__name__}: {exc}"]
        body, hints = _split_detail(raw)
        # `detail_text` 保留整段（测试与静态通道都要它）；渲染按标题/读数/提示三区
        self.detail_text = "\n".join(
            [f"{mod.title}　详情", "", *body, "", *hints, "Esc/← 返回主菜单"]
        )
        self.detail_title = f"{mod.title}　详情"
        self.detail_body = "\n".join(body)
        self.detail_hint = "  ".join([*hints, "Esc/← 返回主菜单"])
        self.query_one("#detail-title", Static).update(self.detail_title)
        self.query_one("#detail-body", Static).update(self.detail_body or "（无读数）")
        self.query_one("#detail-hint", Static).update(self.detail_hint)
        self.query_one("#detail", Vertical).display = True
        self.query_one("#menu", ListView).display = False
        self.query_one("#summary", Static).display = False
        self.query_one("#journal", Vertical).display = False
        self._mark_seen(mod)

    def _tick_detail(self) -> None:
        """逐秒刷新详情读数（倒计时用）：只在**系统与路由**详情在场时重算，别处不动。"""
        if self.view != "detail":
            return
        mod = self._mods[self.selected]
        if mod.key != "system":
            return
        try:
            raw = list(mod.detail())
        except Exception:  # noqa: BLE001 - 刷新失败保持上一帧，不塌详情
            return
        body, hints = _split_detail(raw)
        self.detail_body = "\n".join(body)
        self.detail_text = "\n".join(
            [f"{mod.title}　详情", "", *body, "", *hints, "Esc/← 返回主菜单"]
        )
        self.detail_hint = "  ".join([*hints, "Esc/← 返回主菜单"])
        self.query_one("#detail-body", Static).update(self.detail_body or "（无读数）")
        self.query_one("#detail-hint", Static).update(self.detail_hint)
        self._mark_seen(mod)  # 用户正看着：把最新读数记为已见，徽标不自亮

    def _open_journal(self) -> None:
        self.view = "journal"
        self.query_one("#menu", ListView).display = False
        self.query_one("#summary", Static).display = False
        self.query_one("#detail", Vertical).display = False
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

    # ---- 任务书面板（§3.4 / §2.4 #1）---------------------------------------
    def action_taskbooks(self) -> None:
        self._open_taskbooks()

    def _open_taskbooks(self) -> None:
        from .. import taskbooks as _tb  # 延迟导入：数据层叶子

        self.view = "taskbooks"
        try:
            self.tb_items = _tb.list_all()
        except Exception:  # noqa: BLE001 - 读数失败不该塌面板
            self.tb_items = []
        lv = self.query_one("#tb-list", ListView)
        lv.clear()
        for d in self.tb_items:
            lv.append(ListItem(Static(f"[{d.get('status')}] {d.get('title')}")))
        lv.index = 0  # 重建后选中态回到首项（clear/append 后 index 可能为 None）
        self._set_tb_msg("")
        self._hide_all_views()
        self.query_one("#taskbook", Vertical).display = True
        lv.focus()
        # 进面板 = 进「任务书」模块视图 → 写该模块水位（§2.2「进入详情即写水位」；f1 起
        # Enter 也走这里，故水位在这里写，不再依赖 _open_detail）。
        mod = next((m for m in self._mods if m.key == "taskbooks"), None)
        if mod is not None:
            self._mark_seen(mod)

    def action_tb_new(self) -> None:
        if self.view != "taskbooks":
            return
        self.view = "tb-new"
        self._hide_all_views()
        self.query_one("#tb-new", Vertical).display = True
        self.query_one("#tb-new-title-input", Input).value = ""
        self.query_one("#tb-new-body", TextArea).text = ""
        self._set_tb_new_msg("")
        self.query_one("#tb-new-title-input", Input).focus()

    def action_tb_accept(self) -> None:
        self._tb_act(accept=True)

    def action_tb_reject(self) -> None:
        self._tb_act(accept=False)

    def _tb_act(self, *, accept: bool) -> None:
        """对**选中项**批复：accept / reject＋批语（批语取 ``#tb-note``）。"""
        if self.view != "taskbooks":
            return
        from .. import taskbooks as _tb

        idx = int(self.query_one("#tb-list", ListView).index or 0)
        if idx >= len(self.tb_items):
            return
        tid = str(self.tb_items[idx].get("id"))
        try:
            if accept:
                _tb.accept(tid)
                msg = f"[√] {tid} → accepted"
            else:
                note = self.query_one("#tb-note", Input).value.strip()
                _tb.reject(tid, note)
                msg = f"[√] {tid} → rejected（批语已记），已回 new 待重拆解"
        except ValueError as exc:
            self._set_tb_msg(f"[!] {exc}")
            return
        self._open_taskbooks()  # 重建列表（状态已变）——随后再落读数（重建会清空）
        self._set_tb_msg(msg)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """「投递」：程序化组装元数据并落盘（TASKBOOK-NEW 由 watch 增量报出）。"""
        if event.button.id != "tb-new-submit" or self.view != "tb-new":
            return
        from .. import taskbooks as _tb

        title = self.query_one("#tb-new-title-input", Input).value.strip()
        body = self.query_one("#tb-new-body", TextArea).text
        if not title:
            self._set_tb_new_msg("✗ 标题不能为空")
            return
        try:
            st = _tb.create(title, body, source="mirror-tui")
        except ValueError as exc:
            self._set_tb_new_msg(f"✗ {exc}")
            return
        self.last_created = str(st["id"])
        self._set_tb_new_msg(f"已投递：{st['id']}（TASKBOOK-NEW 唤醒组长）")

    def _set_tb_msg(self, text: str) -> None:
        self.tb_msg = text
        self.query_one("#tb-msg", Static).update(text)

    def _set_tb_new_msg(self, text: str) -> None:
        self.tb_new_msg = text
        self.query_one("#tb-new-msg", Static).update(text)


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
