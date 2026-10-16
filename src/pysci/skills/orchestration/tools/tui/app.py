"""`pysci` 裸名交互式控制台——主菜单框架（README §12，M2）。

在 m1 骨架（模块列表 + ↑↓/Enter/Esc）之上长出**主菜单交互模型**（§2.2）与**更新水位**：
每模块一行「标题（左）＋ 更新徽标（右）」、选中项下方展开 3–5 行概览板、Enter 进全屏详情、
Esc 返回上级；**进入详情即写该模块水位**（真源 ``state/mirror-seen.json``，见
:mod:`.watermark`）。`journal` 的编号分流菜单迁进 TUI 口述动作（主菜单 `j` 与「待决与验收」
详情内的入口；偏好/指令二分法理不变）。

入口分派（§2.1）沿用 m1：``pysci <子命令> …`` → :func:`mirror.main` 静态通道（LLM 专用）；
``pysci``（无子命令）tty → 本 TUI；非 tty → 等价 ``pysci status``（复用 :func:`mirror.cmd_status`）。

数据层全在 :mod:`..mirror`（§2.5 不复制读数）；模块定义在 :mod:`.modules`。
"""

from __future__ import annotations

import shutil
import subprocess
import sys

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    ListItem,
    ListView,
    Markdown,
    RadioButton,
    RadioSet,
    SelectionList,
    Static,
    TextArea,
)
from textual.widgets.selection_list import Selection

from .. import mirror
from . import watermark
from .modules import Module, modules  # re-export：m1 的 `app.modules()` 仍在
from .theme import (
    _T_BG,
    _T_BODY,
    _T_BORDER,
    _T_BORDER_SOFT,
    _T_HINT,
    _T_HL,
    _T_INSET,
    _T_SUBTLE,
    _T_TITLE,
)
from .widgets import (
    ChainListItem,
    MarkedVerticalScroll,
    RowActivated,
    RowListView,
    ScrollMarkMixin,
    make_row,
)

#: 无更新时的灰点（§2.2：`● NEW` / 数字 / 灰点）。
_BADGE_NONE = "·"

#: 各视图下**有可观察效果**的 action（f1：footer 只广告这些，广告的键必须真有效果）。
#: 未列出的受管 action 在当前视图返回 `False`（Textual 语义：禁用**且不显示**）。
#: **输入框视图（journal / tb-new）留空**：焦点在 `Input` 上时可打印键属于文本输入
#: （实测 `screen.active_bindings` 不含 `t`/`n`/`a`/`r`），标成「可用」是撒谎——
#: 用户打字要的是字符，不是快捷键。
_ACTIONS_BY_VIEW: dict[str, frozenset[str]] = {
    "menu": frozenset({"taskbooks", "journal"}),
    # 详情/任务书面板不再是 `t`/`j` 的阵地（用户 2026-10-16：t/j 只在主菜单）——
    # 跨面板跳转由 Esc 回菜单再进，footer 不广告别处无效的键。
    "detail": frozenset(),
    "journal": frozenset(),
    "taskbooks": frozenset({"tb_new", "tb_accept", "tb_reject", "tb_open_latest"}),
    # 任务书详情页（Enter）：**批复就在读报告处**——`a`/`r` 在此可用（`n` 新增不可用：
    # 详情页不是列表，新增无处落）。
    "tb-detail": frozenset({"tb_accept", "tb_reject"}),
    "tb-new": frozenset(),
    "tb-reject": frozenset(),
}

#: 受本类视图闸管的 action（其余——`back`/`quit` 与 widget 内部动作——一律放行，不干预导航）。
_GATED_ACTIONS: frozenset[str] = frozenset(
    {"taskbooks", "tb_new", "tb_accept", "tb_reject", "tb_open_latest", "journal"}
)

#: 详情页**滚动** action（用户 2026-10-15：「详情页无法滚动，最大化窗口仍看不全」）。
#: 单独一张闸（不进 `_GATED_ACTIONS`）：滚动键不进 footer 广告面（那会把提示行挤满），
#: 只在 :data:`_SCROLL_PANES` 所列视图生效——位置指示行本身就是它的可见性线索。
_SCROLL_ACTIONS: frozenset[str] = frozenset(
    {
        "detail_scroll_up",
        "detail_scroll_down",
        "detail_page_up",
        "detail_page_down",
        "detail_scroll_home",
        "detail_scroll_end",
    }
)

#: 可滚动视图：``视图名 → (滚动容器 id, 位置指示 id)``。
#: **一处同构**（模块详情 / 任务书详情 / 任务书列表），故用一张表而不是多套方法（§2.5 不留两份）。
#: s5–s8 各自视图落地时**追加本表条目**（新面板自带位置行即自动获得滚动键与实时指示）。
_SCROLL_PANES: dict[str, tuple[str, str]] = {
    "detail": ("#detail-body", "#detail-scroll"),
    "tb-detail": ("#tb-detail-body", "#tb-detail-scroll"),
    "taskbooks": ("#tb-list", "#tb-scroll"),
}

#: 位置指示前缀（测试与用户读同一串）。
_SCROLL_MARK_HEAD = "滚动：↑↓/PgUp/PgDn/Home/End ｜ 位置 "

#: 返工三档模式 / 优先级在 TUI 里的**显示序**（RadioSet 下标 → 数据层取值）。
_REWORK_MODE_BY_INDEX: tuple[str, ...] = ("full", "no_inspector", "rework_only")
_REWORK_MODE_LABELS: tuple[str, ...] = (
    "① 完整（综合项目改进＋监察改进＋返工执行）",
    "② 无监察改进（A→B）",
    "③ 仅返工（B，快速返工）",
)
_PRIORITY_BY_INDEX: tuple[str, ...] = ("high", "normal", "low")
_PRIORITY_LABELS: tuple[str, ...] = ("优先级 高", "优先级 中（默认）", "优先级 低")

# ---------------------------------------------------------------------------
# 主题常量已迁入 :mod:`.theme`（s2 共享层：唯一出处，本文件只插值引用）——
# 同一视觉参数在 tui/ 全目录源码只硬编码一次（tests/test_tui_visual_structure.py
# 的运行期拼接 grep 判据钉住）。
#
# 提示机制已退役（s4）：数据层详情不再含提示行（s1/s3 清理），逐行辨识提示行的
# 机制（标记表 + 切分函数）已删——操作提示改由 TUI **生成**纯文本（见
# `_render_detail`/`_open_tb_detail_by_id`），Esc 语义由 Footer 广告承担。
# ---------------------------------------------------------------------------


class _TbList(ScrollMarkMixin, RowListView):
    """任务书分组列表：s2 滚动知情（``ScrollMarkMixin``）＋ s2 行激活归一（``RowListView``）。

    组合点是本环节：分组列表既要报滚动位置（`#tb-scroll`）又要区分键 / 单击 / 双击
    （``RowActivated.source/chain``）。两个 mixin 都来自 s2 共享层，此处只做组合。
    """


#: 单击组行后等待「是否双击」的窗口（秒）——双击到达即取消单击的展开/收起。
_TB_CLICK_TOGGLE_DELAY = 0.35


class PysciTUI(App):
    """`pysci` 交互式控制台（主菜单 ↔ 详情 ↔ 口述）；可无头测试（tests 的 Pilot 用例）。"""

    TITLE = "pysci 控制台"
    BINDINGS = [
        ("escape", "back", "返回/退出"),
        ("t", "taskbooks", "任务书"),
        ("n", "tb_new", "新增任务书"),
        ("a", "tb_accept", "接受选中"),
        ("r", "tb_reject", "驳回选中"),
        ("o", "tb_open_latest", "打开最新"),
        ("j", "journal", "口述"),
        ("q", "quit", "退出"),
        # 详情页滚动（只在 detail/tb-detail 生效，见 check_action；不占 footer 广告面）
        Binding("up", "detail_scroll_up", "上滚", show=False),
        Binding("down", "detail_scroll_down", "下滚", show=False),
        Binding("pageup", "detail_page_up", "上翻页", show=False),
        Binding("pagedown", "detail_page_down", "下翻页", show=False),
        Binding("home", "detail_scroll_home", "到顶", show=False),
        Binding("end", "detail_scroll_end", "到底", show=False),
        # vim 风格到顶/到底（与 TUI 单字母键位同族）
        Binding("g", "detail_scroll_home", "到顶", show=False),
        Binding("G", "detail_scroll_end", "到底", show=False),
    ]
    CSS = f"""
    #menu {{ {_T_BORDER} height: 1fr; {_T_BG} }}
    #menu > ListItem.-highlight {{ {_T_HL} }}
    .row {{ width: 100%; height: 1; }}
    .row-title {{ width: 1fr; }}
    .row-badge {{ width: auto; color: $warning; text-style: bold; }}
    #summary {{ height: auto; {_T_BORDER_SOFT} {_T_BG} {_T_INSET} }}
    #summary-title {{ {_T_TITLE} }}
    #summary-body {{ {_T_BODY} }}
    #detail {{ height: 1fr; {_T_INSET} {_T_BORDER} {_T_BG} }}
    #detail-title {{ {_T_TITLE} }}
    #detail-body {{ {_T_BODY} height: 1fr; overflow-y: auto; scrollbar-gutter: stable; }}
    #detail-body-text {{ height: auto; }}
    #detail-hint {{ {_T_HINT} margin-top: 1; }}
    #detail-scroll {{ {_T_SUBTLE} height: 1; }}
    #journal {{ height: 1fr; {_T_INSET} {_T_BORDER} {_T_BG} }}
    #journal-title {{ {_T_TITLE} }}
    #journal-result {{ color: $success; }}
    #taskbook {{ height: 1fr; {_T_INSET} {_T_BORDER} {_T_BG} }}
    #tb-title {{ {_T_TITLE} }}
    #tb-list {{ height: 1fr; }}
    #tb-list > ListItem.-highlight {{ {_T_HL} }}
    #tb-msg {{ color: $success; }}
    #tb-scroll {{ {_T_SUBTLE} height: 1; }}
    #tb-new {{ height: 1fr; {_T_INSET} {_T_BORDER_SOFT} {_T_BG} }}
    #tb-new-title {{ {_T_TITLE} }}
    #tb-new-body {{ height: 1fr; }}
    #tb-new-msg {{ color: $success; }}
    #tb-reject {{ height: 1fr; {_T_INSET} {_T_BORDER_SOFT} {_T_BG} }}
    #tb-reject-title {{ {_T_TITLE} }}
    #tb-reject-note-label {{ {_T_SUBTLE} height: 1; }}
    #tb-note {{ height: 5; overflow-y: auto; scrollbar-gutter: stable; }}
    #tb-reject-msg {{ color: $success; }}
    #tb-detail {{ height: 1fr; {_T_INSET} {_T_BORDER} {_T_BG} }}
    #tb-detail-title {{ {_T_TITLE} {_T_BORDER} {_T_INSET} }}
    #tb-detail-body {{ {_T_BODY} height: 1fr; overflow-y: auto; scrollbar-gutter: stable; }}
    #tb-detail-md {{ {_T_BODY} height: auto; }}
    #tb-detail-hint {{ {_T_HINT} margin-top: 1; }}
    #tb-detail-msg {{ color: $success; }}
    #tb-open-vscode {{ margin-top: 1; }}
    #tb-detail-scroll {{ {_T_SUBTLE} height: 1; }}
    """

    def __init__(self, *, mods: list[Module] | None = None, seen_path=None) -> None:
        super().__init__()
        self._mods = list(mods) if mods is not None else modules()
        self._seen_path = seen_path
        self._seen = watermark.load_seen(seen_path)
        # 可断言的状态面（测试不依赖 widget 内部）：视图 + 选中下标 + 详情正文 + 徽标/口述读数。
        self.view = (
            "menu"  # "menu" | "detail" | "journal" | "taskbooks" | "tb-detail" | …
        )
        self.selected = 0
        self.detail_text = ""
        # 详情三区（f2 分区渲染）——textual 8 的 Static 没有 .renderable，测试读这里
        self.detail_title = ""
        self.detail_body = ""
        self.detail_hint = ""
        # 任务书详情页三区（同上：三区分离，测试读属性不读 widget 内部）
        self.tb_detail_id = ""
        self.tb_detail_title = ""
        self.tb_detail_body = ""
        self.tb_detail_hint = ""
        # 详情页滚动态（用户 2026-10-15「详情页无法滚动」）：行偏移 + 位置指示文案
        self.detail_scroll = 0
        self.detail_scroll_mark = ""
        self.tb_detail_scroll = 0
        self.tb_detail_scroll_mark = ""
        # 任务书列表滚动态（s4 收编：ListView 也报位置）
        self.tb_scroll = 0
        self.tb_scroll_mark = ""
        self.journal_result = ""
        # 任务书分组列表（s5 用户核心需求）：**行模型**（列表按它重建）＋ 家族展开态
        # （初始全部 False＝收起，主列表一屏只见每组最新/本体）。
        self.tb_rows: list[dict] = []
        self.tb_expanded: dict[str, bool] = {}
        self._tb_timer = None  # 单击组行的延迟切换定时器（双击到达即取消）
        self.tb_msg = ""
        self.tb_new_msg = ""
        self.tb_reject_msg = ""
        self.tb_detail_msg = ""
        self.last_created = ""
        self.last_rework = ""
        self.reject_target = ""
        self.badge_texts: dict[str, str] = {
            m.key: self._badge_text(m) for m in self._mods
        }

    # ---- 组合 -------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="body"):
            yield ListView(*self._rows(), id="menu")
            with Vertical(id="summary"):
                yield Static("", id="summary-title")
                yield Static("", id="summary-body")
            with Vertical(id="detail"):
                yield Static("", id="detail-title")
                yield MarkedVerticalScroll(
                    Static("", id="detail-body-text"), id="detail-body", can_focus=False
                )
                yield Static("", id="detail-hint")
                yield Static("", id="detail-scroll")
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
                    "任务书：↑↓ 选｜Enter 展开/收起｜双击或 o 进详情｜n 新增｜a 接受｜r 驳回",
                    id="tb-title",
                )
                yield _TbList(id="tb-list")
                yield Static("", id="tb-msg")
                yield Static("", id="tb-scroll")
            with Vertical(id="tb-new"):
                yield Static(
                    "新增任务书（① 选优先级 → 标题 + 多行正文 → 点「投递」落盘）",
                    id="tb-new-title",
                )
                yield RadioSet(
                    *(
                        RadioButton(lbl, value=lbl.endswith("（默认）"))
                        for lbl in _PRIORITY_LABELS
                    ),
                    id="tb-new-priority",
                )
                yield Input(placeholder="标题", id="tb-new-title-input")
                yield TextArea(id="tb-new-body")
                yield Button("投递", id="tb-new-submit")
                yield Static("", id="tb-new-msg")
            with Vertical(id="tb-detail"):
                yield Static("", id="tb-detail-title")
                yield MarkedVerticalScroll(
                    Markdown(id="tb-detail-md"), id="tb-detail-body", can_focus=False
                )
                yield Static("", id="tb-detail-hint")
                yield Button("用 VSCode 打开原文", id="tb-open-vscode")
                yield Static("", id="tb-detail-msg")
                yield Static("", id="tb-detail-scroll")
            with Vertical(id="tb-reject"):
                yield Static(
                    "驳回交付（返工=新大任务）：① 返工模式 → ② 优先级 → ③ 批语",
                    id="tb-reject-title",
                )
                yield RadioSet(
                    *(
                        RadioButton(lbl, value=(i == 0))
                        for i, lbl in enumerate(_REWORK_MODE_LABELS)
                    ),
                    id="tb-reject-mode",
                )
                yield RadioSet(
                    *(
                        RadioButton(lbl, value=lbl.endswith("（默认）"))
                        for lbl in _PRIORITY_LABELS
                    ),
                    id="tb-reject-priority",
                )
                yield Static(
                    "③ 批语（多行：Enter 换行，填完点按钮提交）",
                    id="tb-reject-note-label",
                )
                yield TextArea("", id="tb-note")
                yield Button("确认驳回并生成返工任务书", id="tb-reject-submit")
                yield Static("", id="tb-reject-msg")
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
        self.query_one("#tb-reject", Vertical).display = False
        self.query_one("#tb-detail", Vertical).display = False
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
        """进入详情即写水位（§2.2）。**cursor 与已存水位相同则不写盘**——倒计时每秒
        tick 不必每秒重写 ``mirror-seen.json``。写失败不挡浏览（观察面 fail-open）。"""
        try:
            cur = mod.cursor()
        except Exception:  # noqa: BLE001 - 读数失败不塌浏览
            return
        if self._seen.get(mod.key) == cur:
            return
        try:
            self._seen[mod.key] = cur
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

    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Enter：主菜单 → 详情；「任务书」模块 → **专用面板**（＝`t`，f1 用户路径修复：
        菜单→Enter→`n` 必须可用）。**任务书列表已改走 `RowActivated`**（s5 分组列表：
        Enter 是展开/收起、双击/`o` 才是进详情），故此处不再处理 tb-list。"""
        if event.list_view.id == "menu":
            idx = int(event.list_view.index or 0)
            if self._mods[idx].key == "taskbooks":
                await self._open_taskbooks()
            else:
                self._open_detail(idx)

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool:
        """footer 键位广告与守卫一致化（f1）：受管 action 在当前视图无效果 → 返回 `False`
        （Textual 语义：`False` = 禁用且**不显示**于 Footer；`None` 只是禁用但仍显示）。
        其余 action（`back`/`quit` 与 widget 内部导航/提交）一律放行，绝不干预光标与输入。

        滚动 action 另走一张闸（:data:`_SCROLL_ACTIONS`）：只在 :data:`_SCROLL_PANES`
        所列视图可用——别处按上下/翻页键是列表光标与输入框的正常语义，绝不能被滚动抢走。
        """
        if action in _SCROLL_ACTIONS:
            return self.view in _SCROLL_PANES
        if action not in _GATED_ACTIONS:
            return True
        return action in _ACTIONS_BY_VIEW.get(self.view, frozenset())

    # ---- 详情页滚动（用户 2026-10-15：最大化窗口仍看不全）--------------------
    def _scroll_widgets(self) -> tuple[Widget, Static] | None:
        """当前视图的（滚动容器, 位置指示）；非可滚视图/缺件 → ``None``。"""
        pair = _SCROLL_PANES.get(self.view)
        if pair is None:
            return None
        try:
            return (
                self.query_one(pair[0]),
                self.query_one(pair[1], Static),
            )
        except Exception:  # noqa: BLE001 - 组合面容忍缺件（观察面 fail-open）
            return None

    def on_pane_scrolled(self, pane: Widget) -> None:
        """滚动知情接线（s2 ``ScrollMarkMixin`` 回调）：**任一**滚动来源——滚轮、滚动条
        拖动、键盘、程序化赋值——都经 ``scroll_y`` reactive 走到这里 → 立即刷新位置指示。

        修复用户实测缺陷：原先只有键盘路径手调 `_refresh_scroll_mark`，滚轮/滚动条导航
        时指示不更新。只认**当前视图**的面板（隐藏面板的光标跟随滚动不串台）。
        """
        got = self._scroll_widgets()
        if got is not None and got[0] is pane:
            self._refresh_scroll_mark()

    def _refresh_scroll_mark(self) -> None:
        """把滚动位置写成指示行（`顶` / `45%` / `底`）并落进可断言的态。"""
        got = self._scroll_widgets()
        if got is None:
            return
        pane, mark = got
        max_y = int(pane.max_scroll_y)
        y = int(pane.scroll_y)
        if max_y <= 0:
            text = f"{_SCROLL_MARK_HEAD}整屏（无需滚动）"
        else:
            # 中间位置夹到 1–99%：否则第 1/284 行显示「0%」（与「顶」撞车）、
            # 倒数第 1 行显示 100%（与「底」撞车）——位置指示不能自相矛盾。
            pct = min(99, max(1, round(100 * y / max_y)))
            pos = "顶" if y <= 0 else ("底" if y >= max_y else f"{pct}%")
            text = f"{_SCROLL_MARK_HEAD}{pos}（第 {y}/{max_y} 行）"
        mark.update(text)
        if self.view == "tb-detail":
            self.tb_detail_scroll, self.tb_detail_scroll_mark = y, text
        elif self.view == "taskbooks":
            self.tb_scroll, self.tb_scroll_mark = y, text
        else:
            self.detail_scroll, self.detail_scroll_mark = y, text

    def _scroll_body(self, delta: int) -> None:
        """滚动当前详情正文（delta=行数；Textual 内部会夹到有效区间）。"""
        got = self._scroll_widgets()
        if got is None:
            return
        pane = got[0]
        pane.scroll_y = max(0, min(pane.scroll_y + delta, pane.max_scroll_y))
        self._refresh_scroll_mark()

    def _scroll_body_home(self, *, end: bool) -> None:
        """跳到顶（``end=False``）或底（``end=True``）。"""
        got = self._scroll_widgets()
        if got is None:
            return
        pane = got[0]
        pane.scroll_y = pane.max_scroll_y if end else 0
        self._refresh_scroll_mark()

    def action_detail_scroll_up(self) -> None:
        self._scroll_body(-1)

    def action_detail_scroll_down(self) -> None:
        self._scroll_body(1)

    def action_detail_page_up(self) -> None:
        self._scroll_body(-self._page_rows())

    def action_detail_page_down(self) -> None:
        self._scroll_body(self._page_rows())

    def action_detail_scroll_home(self) -> None:
        self._scroll_body_home(end=False)

    def action_detail_scroll_end(self) -> None:
        self._scroll_body_home(end=True)

    def _page_rows(self) -> int:
        """翻页步长：可视区行数（一屏）减 1 行留重叠，最小 1。"""
        got = self._scroll_widgets()
        if got is None:
            return 1
        return max(1, int(got[0].size.height) - 1)

    def _reset_scroll(self) -> None:
        """打开/刷新详情：滚动回顶，并按**首帧后**的布局重算位置指示。

        首帧前 ``max_scroll_y`` 还是旧值（内容刚 update、尚未重新布局），故先算一次让
        状态面立刻可用，再 ``call_after_refresh`` 用真实布局覆盖（否则指示会谎报「整屏」）。
        """
        got = self._scroll_widgets()
        if got is None:
            return
        got[0].scroll_y = 0
        self._refresh_scroll_mark()
        self.call_after_refresh(self._refresh_scroll_mark)

    def _open_tb_detail_by_id(self, tb_id: str) -> None:
        """任务书**详情页**（任何状态可进）：**头卡** ＋ `Markdown` 正文（s5）。

        正文＝ :func:`taskbooks.detail_markdown`（Markdown 文档单一实现；delivered 态
        「★ 批复参考」置顶）。正文区是**可滚动**的 ``MarkedVerticalScroll``（交付报告可能
        很长），滚动位置由 :meth:`_reset_scroll` 复位到顶、底部指示行报位置（滚轮/滚动条/
        键盘任一来源都实时刷新，见 :meth:`on_pane_scrolled`）；「用 VSCode 打开原文」
        按钮见 :meth:`_open_in_vscode`。
        """
        from .. import taskbooks as _tb

        if not tb_id:
            return
        st = _tb.load_state(tb_id) or {}
        try:
            md_text = _tb.detail_markdown(tb_id)
        except Exception as exc:  # noqa: BLE001 - 读数坏一项不塌详情（观察面 fail-open）
            md_text = (
                f"# 任务书 {tb_id}\n\n[!] 详情读数失败：{type(exc).__name__}: {exc}\n"
            )
        self.view = "tb-detail"
        self.tb_detail_id = tb_id
        self.tb_detail_title = (
            f"{_tb.status_badge(str(st.get('status') or ''))}　{tb_id}"
            f"　{str(st.get('title') or '')}"
        )
        self.tb_detail_body = md_text
        # 操作提示由 TUI 生成（数据层详情已无提示行，s4 提示机制退役）；Esc 由 Footer 承担。
        self.tb_detail_hint = "a 接受　r 驳回"
        self._hide_all_views()
        self.query_one("#tb-detail-title", Static).update(self.tb_detail_title)
        self.query_one("#tb-detail-md", Markdown).update(md_text)
        self.query_one("#tb-detail-hint", Static).update(self.tb_detail_hint)
        self._set_tb_detail_msg("")
        self.query_one("#tb-detail", Vertical).display = True
        self._reset_scroll()

    def _open_in_vscode(self) -> None:
        """用 VSCode 打开任务书原文（``code`` 不在 PATH / Popen 异常 → 消息行如实报错）。

        ``Popen`` 不等待（不阻塞 UI 事件循环）。
        """
        from .. import taskbooks as _tb

        tid = self.tb_detail_id
        if not tid:
            return
        path = _tb.md_path(tid)
        if shutil.which("code") is None:
            self._set_tb_detail_msg("[!] 打开失败：PATH 中找不到 `code`（VSCode CLI）")
            return
        try:
            subprocess.Popen(["code", str(path)])
        except Exception as exc:  # noqa: BLE001 - 启动失败如实报
            self._set_tb_detail_msg(f"[!] 打开失败：{type(exc).__name__}: {exc}")
            return
        self._set_tb_detail_msg(f"[√] 已在 VSCode 打开 {path}")

    async def action_back(self) -> None:
        """Esc/←：任务书详情页 → 任务书列表；其余子视图 → 主菜单；主菜单 → 退出。"""
        if self.view == "tb-detail":
            await self._open_taskbooks()
        elif self.view != "menu":
            self._show_menu()
        else:
            self.exit()

    def action_journal(self) -> None:
        """主菜单 `j` → 口述动作（§2.4：journal 编号分流菜单迁入 TUI）。"""
        self._open_journal()

    # ---- 渲染 -------------------------------------------------------------
    def _render_summary(self, idx: int) -> None:
        """概览板（§2.2）：**标题行**「〔模块名〕概览」＋读数正文（3–5 行，截行逻辑不变）。"""
        mod = self._mods[idx]
        try:
            lines = [ln for ln in mod.summary() if ln][:5]  # §2.2：3–5 行概览板
        except Exception as exc:  # noqa: BLE001 - 读数坏一项不塌菜单
            lines = [f"读数失败：{type(exc).__name__}: {exc}"]
        try:
            self.query_one("#summary-title", Static).update(f"〔{mod.title}〕概览")
            self.query_one("#summary-body", Static).update(
                "\n".join(lines) or "（无读数）"
            )
        except Exception:  # noqa: BLE001 - 组合面容忍缺件（无头测试无面板时不塌）
            return

    def _hide_all_views(self) -> None:
        for sel in (
            "#menu",
            "#summary",
            "#detail",
            "#journal",
            "#taskbook",
            "#tb-new",
            "#tb-reject",
            "#tb-detail",
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
        self.query_one("#summary", Vertical).display = True
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
        self._render_detail(mod, body)
        self.query_one("#detail", Vertical).display = True
        self.query_one("#menu", ListView).display = False
        self.query_one("#summary", Vertical).display = False
        self.query_one("#journal", Vertical).display = False
        self._reset_scroll()
        self._mark_seen(mod)

    def _render_detail(self, mod: Module, body: list[str]) -> None:
        """渲染模块详情三区（标题 / 读数 / 操作提示）＋静态读面 ``detail_text``。

        操作提示由 **TUI 生成**（数据层详情无提示行，s1/s3 已清；s4 退役逐行辨识机制）：
        模块详情无专属操作可广告 → 留空。**一律不写「Esc/← 返回…」**——Esc 语义由
        Footer「返回/退出」广告承担（消除「← 无绑定却出现在提示里」的假广告）。
        """
        self.detail_title = f"{mod.title}　详情"
        self.detail_body = "\n".join(body)
        self.detail_hint = ""
        parts = [self.detail_title, "", *body]
        if self.detail_hint:
            parts += ["", self.detail_hint]
        self.detail_text = "\n".join(parts)
        self.query_one("#detail-title", Static).update(self.detail_title)
        self.query_one("#detail-body-text", Static).update(
            self.detail_body or "（无读数）"
        )
        self.query_one("#detail-hint", Static).update(self.detail_hint)

    def _tick_detail(self) -> None:
        """逐秒刷新详情读数（倒计时用）：只在**系统与路由**详情在场时重算，别处不动。"""
        if self.view != "detail":
            return
        mod = self._mods[self.selected]
        if mod.key != "system":
            return
        try:
            body = list(mod.detail())
        except Exception:  # noqa: BLE001 - 刷新失败保持上一帧，不塌详情
            return
        self._render_detail(mod, body)
        self._refresh_scroll_mark()  # 每秒重算只更读数，**不动滚动位置**（用户正读的那屏不跳）
        self._mark_seen(mod)  # 用户正看着：把最新读数记为已见，徽标不自亮（同值不写盘）

    def _open_journal(self) -> None:
        self.view = "journal"
        self.query_one("#menu", ListView).display = False
        self.query_one("#summary", Vertical).display = False
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
    async def action_taskbooks(self) -> None:
        await self._open_taskbooks()

    async def _open_taskbooks(self, *, keep_root: str | None = None) -> None:
        self.view = "taskbooks"
        await self._rebuild_tb_list(
            keep_root=keep_root
        )  # 行模型 + 列表（含滚动位置复位）
        self._set_tb_msg("")
        self._hide_all_views()
        self.query_one("#taskbook", Vertical).display = True
        self.query_one("#tb-list", _TbList).focus()
        # 进面板 = 进「任务书」模块视图 → 写该模块水位（§2.2「进入详情即写水位」；f1 起
        # Enter 也走这里，故水位在这里写，不再依赖 _open_detail）。
        mod = next((m for m in self._mods if m.key == "taskbooks"), None)
        if mod is not None:
            self._mark_seen(mod)

    # ---- 分组列表（s5：返工家族，收起/展开 + 键鼠归一）------------------------
    def _build_tb_rows(self) -> list[dict]:
        """行模型：**每组一行**（最新/本体）；展开的组下逐行列出成员（新→旧、本体殿后）。

        行 = ``{"kind": "group"|"sub", "tb_id", "root", …}``；``#tb-list`` **只按本函数
        重建**（测试亦只准按行模型选行，禁裸下标）。家族来自 s1 ``taskbooks.families()``。
        """
        from .. import taskbooks as _tb  # 延迟导入：数据层叶子

        try:
            fams = _tb.families()
        except Exception:  # noqa: BLE001 - 读数失败不该塌面板
            return []
        rows: list[dict] = []
        for fam in fams:
            members = fam["members"]
            if not members:
                continue
            latest = members[0]
            root = str(fam["root"])
            rows.append(
                {
                    "kind": "group",
                    "tb_id": str(latest.get("id") or ""),
                    "root": root,
                    "title": str(latest.get("title") or ""),
                    "badge": _tb.status_badge(str(latest.get("status") or "")),
                    "size": int(fam["size"]),
                    "round": int(latest.get("rework_round") or 0),
                }
            )
            if not self.tb_expanded.get(root):
                continue
            for m in members:
                rows.append(
                    {
                        "kind": "sub",
                        "tb_id": str(m.get("id") or ""),
                        "root": root,
                        "label": _tb.family_label(m),
                        "indent": 1,
                        "title": str(m.get("title") or ""),
                        "badge": _tb.status_badge(str(m.get("status") or "")),
                    }
                )
        return rows

    def _tb_list_item(self, row: dict) -> ChainListItem:
        """行 → 列表条目（``make_row`` 统一形态：徽标 | 标题 | 右对齐 meta）。"""
        if row["kind"] == "group":
            meta = f"共 {row['size']} 件" if int(row["size"]) > 1 else "本体"
            return ChainListItem(
                make_row(row["badge"], row["title"], meta=meta),
                id=f"tbrow-{row['tb_id']}-{row['kind']}",
            )
        return ChainListItem(
            make_row(
                row["badge"],
                f"└ {row['label']}　{row['title']}",
                indent=int(row.get("indent") or 0) + 1,
            ),
            id=f"tbrow-{row['tb_id']}-{row['kind']}",
        )

    async def _rebuild_tb_list(self, *, keep_root: str | None = None) -> None:
        """按行模型重建 ``#tb-list``；``keep_root`` 给出时把高亮保持在**该组行**上。

        ``ListView.clear()`` 是 **AwaitComplete**（移除在事件循环里异步完成）——必须
        ``await`` 它再挂新行，否则旧行尚未移除、新行已挂载，撞 ``DuplicateIds``（实测）。
        """
        self.tb_rows = self._build_tb_rows()
        try:
            lv = self.query_one("#tb-list", _TbList)
        except Exception:  # noqa: BLE001 - 组合面容忍缺件
            return
        await lv.clear()
        for row in self.tb_rows:
            lv.append(self._tb_list_item(row))
        keep = 0
        if keep_root is not None:
            keep = next(
                (
                    i
                    for i, r in enumerate(self.tb_rows)
                    if r["kind"] == "group" and r["root"] == keep_root
                ),
                0,
            )
        lv.index = min(keep, len(self.tb_rows) - 1) if self.tb_rows else None
        self._reset_scroll()  # 重建 → 滚动回顶并按真实布局刷新位置行（s4 已接列表重建）

    async def _toggle_tb_group(self, root: str) -> None:
        """展开/收起一族（**统一展开**：单件组同样可展开，次级列表仅「本体」一项）。"""
        self.tb_expanded[root] = not self.tb_expanded.get(root, False)
        await self._rebuild_tb_list(keep_root=root)

    def _cancel_tb_timer(self) -> None:
        timer = self._tb_timer
        self._tb_timer = None
        if timer is not None:
            try:
                timer.stop()
            except Exception:  # noqa: BLE001 - 已停/已触发都算取消成功
                pass

    def _schedule_tb_toggle(self, root: str) -> None:
        """组行单击：延迟 ``_TB_CLICK_TOGGLE_DELAY`` 秒再展开/收起——双击到达即取消。"""
        self._cancel_tb_timer()
        self._tb_timer = self.set_timer(
            _TB_CLICK_TOGGLE_DELAY, lambda: self._toggle_tb_group(root)
        )

    async def on_row_activated(self, event: RowActivated) -> None:
        """任务书列表行激活（s2 ``RowListView``：键/单击/双击**同型**消息，键鼠可辨）。

        - 组行 Enter（``key``）→ **立即**展开/收起；组行单击（``click`` chain=1）→ 延迟
          展开/收起（双击到达即取消）；组行双击（chain≥2）→ 取消延迟、直接进最新项详情；
        - 次级行（单击/Enter）→ 立即进该项详情。
        """
        if self.view != "taskbooks":
            return
        index = int(event.index)
        if not (0 <= index < len(self.tb_rows)):
            return
        row = self.tb_rows[index]
        if row["kind"] != "group":
            self._open_tb_detail_by_id(str(row["tb_id"]))
            return
        if event.source == "key":
            await self._toggle_tb_group(str(row["root"]))
        elif int(event.chain) >= 2:
            self._cancel_tb_timer()  # 双击：取消单击的延迟切换
            self._open_tb_detail_by_id(str(row["tb_id"]))
        else:
            self._schedule_tb_toggle(str(row["root"]))

    def action_tb_open_latest(self) -> None:
        """`o`：进选中行目标详情（组行→最新项；次级行→该项）。"""
        if self.view != "taskbooks":
            return
        tid = self._tb_target_id()
        if tid:
            self._open_tb_detail_by_id(tid)

    def _tb_target_id(self) -> str:
        """批复/打开目标：组行/次级行 → 该行 ``tb_id``（组行即最新项）；详情页 → 详情项。

        ``a``/``r``/``o``（含详情页内）都走它——目标语义单点，别处不许再猜下标含义。
        """
        if self.view == "tb-detail" and self.tb_detail_id:
            return self.tb_detail_id
        idx = self._tb_selected_index()
        if 0 <= idx < len(self.tb_rows):
            return str(self.tb_rows[idx].get("tb_id") or "")
        return ""

    def _tb_root_of(self, tb_id: str) -> str:
        """行模型里该任务书所在家族的 root（找不到回空串）。"""
        for r in self.tb_rows:
            if str(r.get("tb_id")) == str(tb_id):
                return str(r.get("root") or "")
        return ""

    def _tb_selected_index(self) -> int:
        """当前选中行下标（``#tb-list`` 高亮行；禁止外部按它猜语义——目标解析见
        :meth:`_tb_target_id`）。"""
        try:
            return int(self.query_one("#tb-list", _TbList).index or 0)
        except Exception:  # noqa: BLE001 - 组合面容忍缺件
            return 0

    def action_tb_new(self) -> None:
        if self.view != "taskbooks":
            return
        self.view = "tb-new"
        self._hide_all_views()
        self.query_one("#tb-new", Vertical).display = True
        self._set_radio("#tb-new-priority", 1)  # 默认「中」
        self.query_one("#tb-new-title-input", Input).value = ""
        self.query_one("#tb-new-body", TextArea).text = ""
        self._set_tb_new_msg("")
        self.query_one("#tb-new-title-input", Input).focus()

    async def action_tb_accept(self) -> None:
        await self._tb_act(accept=True)

    async def action_tb_reject(self) -> None:
        """`r`：对**选中行目标**开三段式驳回面板（组行→最新项；次级行→该项；PSEP_3 §4.1）。"""
        if self.view not in ("taskbooks", "tb-detail"):
            return
        tid = self._tb_target_id()
        if tid:
            self._open_reject_panel(tid)

    def _open_reject_panel(self, tb_id: str) -> None:
        self.view = "tb-reject"
        self.reject_target = tb_id
        self._hide_all_views()
        self.query_one("#tb-reject", Vertical).display = True
        # 复位：模式回①、优先级回中（默认）、批语清空
        self._set_radio("#tb-reject-mode", 0)
        self._set_radio("#tb-reject-priority", 1)
        self.query_one("#tb-note", TextArea).text = ""
        self._set_tb_reject_msg(f"驳回 {tb_id}：请依次选 ① 模式 → ② 优先级 → ③ 批语")
        self.query_one("#tb-reject-mode", RadioSet).focus()

    async def _tb_act(self, *, accept: bool) -> None:
        """对**选中行目标**批复：accept（驳回走三段式面板，见 :meth:`action_tb_reject`）。

        列表与**详情页**都可批复（用户 2026-10-15：读完流程报告就地处批复）；批准后列表
        重建并把高亮保持在**被操作家族**的组行上。
        """
        if self.view not in ("taskbooks", "tb-detail"):
            return
        del accept
        from .. import taskbooks as _tb

        tid = self._tb_target_id()
        if not tid:
            return
        keep_root = self._tb_root_of(tid) or None
        try:
            _tb.accept(tid)
        except ValueError as exc:
            self._set_tb_msg(f"[!] {exc}")
            return
        await self._open_taskbooks(keep_root=keep_root)  # 重建列表（状态已变）
        self._set_tb_msg(f"[√] {tid} → accepted")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """按钮：``#tb-new-submit`` 投递；``#tb-reject-submit`` 三段式驳回；
        ``#tb-open-vscode`` 用 VSCode 打开任务书原文。"""
        if event.button.id == "tb-new-submit" and self.view == "tb-new":
            await self._submit_new_taskbook()
            return
        if event.button.id == "tb-open-vscode" and self.view == "tb-detail":
            self._open_in_vscode()
            return
        if event.button.id == "tb-reject-submit" and self.view == "tb-reject":
            await self._submit_reject()

    def _radio_index(self, selector: str, default: int) -> int:
        """RadioSet 选中下标（`pressed_index` 为 None 时回落默认位）。"""
        idx = self.query_one(selector, RadioSet).pressed_index
        return default if idx is None else int(idx)

    def _set_radio(self, selector: str, index: int) -> None:
        """程序化选中 RadioSet 第 ``index`` 项。

        ``pressed_index`` 只读，故改各按钮的 ``value``；**必须两段式**（backlog 20261015-230248
        ④）：先把所有按钮清成 False 再把目标置 True——直接写 ``value = (i == index)`` 时，
        置 True 会让 RadioSet 内部重排，**上一轮选中的按钮会把 True 留在自己身上**（实测重开
        面板后 ``values=[True,False,True]``：残留值与默认值同时被选中=双绿）。
        """
        btns = list(self.query_one(selector, RadioSet).query(RadioButton))
        for btn in btns:
            btn.value = False
        if 0 <= index < len(btns):
            btns[index].value = True

    async def _submit_new_taskbook(self) -> None:
        """「投递」：程序化组装元数据（含优先级）并落盘（TASKBOOK-NEW 由 watch 增量报出）。"""
        from .. import taskbooks as _tb

        title = self.query_one("#tb-new-title-input", Input).value.strip()
        body = self.query_one("#tb-new-body", TextArea).text
        priority = _PRIORITY_BY_INDEX[self._radio_index("#tb-new-priority", 1)]
        if not title:
            self._set_tb_new_msg("✗ 标题不能为空")
            return
        try:
            st = _tb.create(title, body, source="mirror-tui", priority=priority)
        except ValueError as exc:
            self._set_tb_new_msg(f"✗ {exc}")
            return
        self.last_created = str(st["id"])
        # ③ 成功即**关表单回列表**（backlog 20261015-230248）：留在表单上可被重复点「投递」
        # → 同一份任务书被投两遍；回列表后确认结果一眼可见（新件最新 → 高亮首行即它）。
        await self._open_taskbooks()
        self._set_tb_msg(
            f"[√] 已投递：{st['id']}（优先级 {priority}；TASKBOOK-NEW 唤醒组长）"
        )

    async def _submit_reject(self) -> None:
        """三段式驳回落盘（模式→优先级→批语）→ 生成返工任务书（PSEP_3 §4.1）。"""
        from .. import taskbooks as _tb

        tb_id = self.reject_target
        mode = _REWORK_MODE_BY_INDEX[self._radio_index("#tb-reject-mode", 0)]
        priority = _PRIORITY_BY_INDEX[self._radio_index("#tb-reject-priority", 1)]
        note = self.query_one("#tb-note", TextArea).text.strip()
        try:
            new = _tb.create_rework(tb_id, mode=mode, priority=priority, note=note)
        except ValueError as exc:
            self._set_tb_reject_msg(f"[!] {exc}")
            return
        self.last_rework = str(new["id"])
        # 落账即给下一步命令（用户 2026-10-15：返工引导缺口——逐字写出 plan rework，
        # 不让组长去 grep 才知道有这个入口）
        msg = (
            f"[√] {tb_id} → rejected（留档）；已生成返工任务书 {new['id']}"
            f"（模式 {mode}，优先级 {priority}，第 {new['rework_round']} 轮）"
            f"　下一步：orch plan rework {new['id']} --mode {mode} 生成返工骨架"
        )
        self._set_tb_reject_msg(msg)
        await self._open_taskbooks()  # 回列表并重建（原项状态已变）
        self._set_tb_msg(msg)

    def _set_tb_msg(self, text: str) -> None:
        self.tb_msg = text
        self.query_one("#tb-msg", Static).update(text)

    def _set_tb_new_msg(self, text: str) -> None:
        self.tb_new_msg = text
        self.query_one("#tb-new-msg", Static).update(text)

    def _set_tb_reject_msg(self, text: str) -> None:
        self.tb_reject_msg = text
        self.query_one("#tb-reject-msg", Static).update(text)

    def _set_tb_detail_msg(self, text: str) -> None:
        self.tb_detail_msg = text
        self.query_one("#tb-detail-msg", Static).update(text)


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
