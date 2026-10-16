"""TUI 共享组件（s2 共享层）：s3–s8 各视图复用的**命名接口**。

两类件：

- **知情/溯源件**（``ScrollMarkMixin`` 一族、``ChainListItem``、``RowListView``）：
  把 Textual 8.2.8 无法直接区分的输入路径（滚动来源、键鼠激活）归一成可断言的
  信号；机制事实（已核实，勿重推）：

  - Textual 8.2.8 **没有** ``Scroll`` 事件类——滚轮/滚动条拖动/键盘/程序化赋值全部经
    ``scroll_y`` reactive，唯一可靠接线 = 覆写 ``Widget.watch_scroll_y``；
  - ``ListView.Selected`` 由 Enter（``action_select_cursor``）和单击
    （``ListItem._on_click`` → ``_ChildClicked`` → ``_on_list_item__child_clicked``）
    **都会发、无法区分键鼠**；``Click.chain``（2=双击，``events.py:641``）。
    ``Pilot.click(sel, times=2)`` 按 chain=1、2 发两次。

- **视觉件**（``card``/``section_title``/``hint``/``make_row``/``ChoiceCard``）：
  统一美学的积木，DEFAULT_CSS 一律 f-string 插值引用 :mod:`.theme` 常量
  （**不得重复 CSS 字面量**——唯一出处判据钉在 theme.py）。
"""

from __future__ import annotations

from textual import events
from textual.containers import Horizontal, VerticalScroll
from textual.message import Message
from textual.widgets import ListItem, ListView, Static, Tree

from .theme import (
    _T_BG,
    _T_BORDER,
    _T_HINT,
    _T_HL,
    _T_INSET,
    _T_SUBTLE,
    _T_TITLE,
)


# ---------------------------------------------------------------------------
# 滚动知情：scroll_y 任一来源变化 → 宿主 App（on_pane_scrolled，s4 接线）
# ---------------------------------------------------------------------------
class ScrollMarkMixin:
    """滚动知情 mixin：``scroll_y`` 变化即通知宿主 App。

    Textual 8.2.8 没有 ``Scroll`` 事件类（机制事实，见模块 docstring）——本 mixin
    覆写 :meth:`~textual.widget.Widget.watch_scroll_y`（唯一可靠接线），调
    ``getattr(self.app, "on_pane_scrolled", None)``：存在则 ``self.app.on_pane_scrolled(self)``。
    **fail-soft**：宿主未实现或调用抛异常一律吞掉——知情通道绝不塌滚动本体。
    """

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)  # 基座行为（滚动条同步/重绘）保持
        try:
            notify = getattr(self.app, "on_pane_scrolled", None)
            if notify is not None:
                notify(self)
        except Exception:  # noqa: BLE001 - 知情通道 fail-soft（缺失/异常不塌滚动）
            pass


class MarkedVerticalScroll(ScrollMarkMixin, VerticalScroll):
    """可滚动容器（带滚动知情）。"""


class MarkedList(ScrollMarkMixin, ListView):
    """列表（带滚动知情）。"""


class MarkedTree(ScrollMarkMixin, Tree):
    """树（带滚动知情）。"""


# ---------------------------------------------------------------------------
# 键鼠归一：ChainListItem 记 Click.chain → RowListView 发同型 RowActivated
# ---------------------------------------------------------------------------
class ChainListItem(ListItem):
    """记录 ``Click.chain`` 的 ``ListItem``（单击=1，双击=2…）。

    ``ListItem._on_click`` 原本不读事件（只发 ``_ChildClicked``）——本类在放行原逻辑
    前把 ``event.chain`` 落到 ``self.click_chain``，供 :class:`RowListView` 溯源。
    """

    def __init__(
        self, *children: Static, id: str | None = None, classes: str = ""
    ) -> None:
        super().__init__(*children, id=id, classes=classes)
        self.click_chain: int = 1

    def _on_click(self, event: events.Click) -> None:
        self.click_chain = int(event.chain)
        # Textual 消息派发沿 MRO **逐个类**调同名 handler（message_pump._get_dispatch_methods）
        # ——prevent_default 拦住再派 ListItem._on_click（否则 _ChildClicked 双发）；
        # 基座行为仍由下面的显式 super() 调一次。
        event.prevent_default()
        super()._on_click(event)


class RowActivated(Message):
    """行激活消息：Enter 与单击/双击**同型**，键鼠经 ``source`` 可辨。

    ``source`` ∈ ``{"key", "click"}``；``chain`` = 点击链（``Click.chain``：2=双击，
    3=三击…；键盘激活恒 1）；``item_id`` = 行 ListItem 的 id（无 id → ``None``）。
    """

    def __init__(
        self,
        list_view: RowListView,
        index: int,
        item_id: str | None,
        source: str,
        chain: int,
    ) -> None:
        super().__init__()
        self.list_view: RowListView = list_view
        self.index: int = index
        self.item_id: str | None = item_id
        self.source: str = source
        self.chain: int = chain

    @property
    def control(self) -> RowListView:
        """触发本消息的列表（Textual 消息惯例的 ``control`` 别名）。"""
        return self.list_view


class RowListView(ListView):
    """行式列表：激活统一发 :class:`RowActivated`（**不发** ``ListView.Selected``）。

    - Enter → 覆写 :meth:`~textual.widgets.ListView.action_select_cursor`：发
      ``RowActivated(source="key", chain=1)``；
    - 单击/双击 → ``_ChildClicked`` 通路：读 :class:`ChainListItem` 记录的
      ``click_chain``（普通 ListItem 回落 1）→ 发 ``RowActivated(source="click", chain=…)``
      并照基座语义聚焦/置 index。
    """

    def action_select_cursor(self) -> None:
        item = self.highlighted_child
        if item is None:
            return
        self.post_message(RowActivated(self, int(self.index or 0), item.id, "key", 1))

    def _on_list_item__child_clicked(self, event: ListItem._ChildClicked) -> None:
        event.stop()
        # MRO 逐类派发（见 ChainListItem._on_click 注释）：拦住基座的同名 handler——
        # 否则它会再发一条 ListView.Selected（键鼠归一后 Selected 不该出现在本类动线上）。
        event.prevent_default()
        self.focus()
        self.index = self._nodes.index(event.item)
        chain = int(getattr(event.item, "click_chain", 1) or 1)
        self.post_message(
            RowActivated(self, int(self.index or 0), event.item.id, "click", chain)
        )


# ---------------------------------------------------------------------------
# 视觉件：卡片 / 分区标题 / 提示行 / 统一列表行 / 盲审大选项卡
# ---------------------------------------------------------------------------
class Card(Static):
    """圆角边框卡（``card()`` 工厂的载体；边框/底色/内距全走 theme 常量）。"""

    DEFAULT_CSS = f"""
    Card {{ {_T_BORDER} {_T_BG} {_T_INSET} height: auto; }}
    """


def card(*lines: str, id: str | None = None, classes: str = "") -> Static:
    """圆角边框卡：多行文本 ``Static``（组合面按内容定高）。"""
    return Card("\n".join(str(ln) for ln in lines), id=id, classes=classes)


class SectionTitle(Static):
    """分区标题行（``section_title()`` 的载体；视觉语言同 ``_T_TITLE``）。"""

    DEFAULT_CSS = f"""
    SectionTitle {{ {_T_TITLE} height: auto; }}
    """


def section_title(text: str) -> Static:
    """分区标题行。"""
    return SectionTitle(str(text))


class HintLine(Static):
    """弱化提示行（``hint()`` 的载体；视觉语言同 ``_T_HINT``）。"""

    DEFAULT_CSS = f"""
    HintLine {{ {_T_HINT} height: auto; }}
    """


def hint(text: str) -> Static:
    """弱化提示行（操作提示/从属说明）。"""
    return HintLine(str(text))


class Row(Horizontal):
    """统一列表行（``make_row()`` 的载体）：徽标 | 标题（1fr）| 右对齐 meta。"""

    DEFAULT_CSS = f"""
    Row {{ height: 1; }}
    Row .row-badge-cell {{ width: auto; {_T_HINT} }}
    Row .row-title-cell {{ width: 1fr; }}
    Row .row-meta-cell {{ width: auto; {_T_SUBTLE} text-align: right; }}
    """


def make_row(badge: str, title: str, meta: str = "", indent: int = 0) -> Horizontal:
    """统一列表行：徽标（auto 宽）+ 标题（1fr）+ 右对齐 meta（auto 宽）。

    ``indent`` 控标题左侧留白（家族缩进等场景）。
    """
    prefix = " " * max(0, int(indent))
    return Row(
        Static(str(badge), classes="row-badge-cell"),
        Static(f"{prefix}{title}", classes="row-title-cell"),
        Static(str(meta), classes="row-meta-cell"),
    )


class ChoiceCard(Static):
    """盲审大选项卡：``-selected`` class 高亮切换 + 点击发 :class:`Chosen`。

    选中态由交互流程代码经 :meth:`toggle_selected`/:meth:`set_selected` 驱动
    （点击本身只上报 ``value``，不自作主张改态）。
    """

    DEFAULT_CSS = f"""
    ChoiceCard {{ {_T_BORDER} {_T_BG} {_T_INSET} height: auto; }}
    ChoiceCard.-selected {{ {_T_HL} }}
    """

    class Chosen(Message):
        """选项卡被点击（携带构造时传入的 ``value``）。"""

        def __init__(self, card: ChoiceCard, value: str) -> None:
            super().__init__()
            self.card: ChoiceCard = card
            self.value: str = value

        @property
        def control(self) -> ChoiceCard:
            return self.card

    def __init__(
        self, value: str, label: str, *, id: str | None = None, classes: str = ""
    ) -> None:
        super().__init__(label, id=id, classes=classes)
        self.value: str = str(value)
        self._selected: bool = False

    @property
    def selected(self) -> bool:
        """当前是否高亮（与 ``-selected`` class 同步）。"""
        return self._selected

    def set_selected(self, value: bool) -> None:
        self._selected = bool(value)
        self.set_class(self._selected, "-selected")

    def toggle_selected(self) -> None:
        self.set_selected(not self._selected)

    def on_click(self, event: events.Click) -> None:
        self.post_message(self.Chosen(self, self.value))


__all__ = [
    "Card",
    "ChainListItem",
    "ChoiceCard",
    "HintLine",
    "MarkedList",
    "MarkedTree",
    "MarkedVerticalScroll",
    "Row",
    "RowActivated",
    "RowListView",
    "ScrollMarkMixin",
    "SectionTitle",
    "card",
    "hint",
    "make_row",
    "section_title",
]
