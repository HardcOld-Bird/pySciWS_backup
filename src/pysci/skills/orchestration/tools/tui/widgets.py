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

from typing import TypeVar

from textual import events
from textual.containers import Horizontal, VerticalScroll
from textual.content import Content
from textual.geometry import Size
from textual.message import Message
from textual.style import Style
from textual.widgets import Checkbox, ListItem, ListView, SelectionList, Static, Tree

from .theme import (
    _T_BG,
    _T_BORDER,
    _T_CHECK_OFF,
    _T_CHECK_ON,
    _T_HINT,
    _T_HL,
    _T_HL_FG,
    _T_INSET,
    _T_SUBTLE,
    _T_TITLE,
)


# ---------------------------------------------------------------------------
# 滚动知情：scroll_y 任一来源变化 → 宿主 App（on_pane_scrolled，s4 接线）
# ---------------------------------------------------------------------------
class ScrollMarkMixin:
    """滚动知情 mixin：``scroll_y`` / 内容尺寸变化即通知宿主 App。

    Textual 8.2.8 没有 ``Scroll`` 事件类（机制事实，见模块 docstring）——本 mixin
    覆写 :meth:`~textual.widget.Widget.watch_scroll_y`（滚动来源）与
    ``watch_virtual_size``（**内容尺寸**：Markdown 异步渲染、条目增减、窗口变化），
    调 ``getattr(self.app, "on_pane_scrolled", None)``：存在则
    ``self.app.on_pane_scrolled(self)``。
    **fail-soft**：宿主未实现或调用抛异常一律吞掉——知情通道绝不塌滚动本体。
    """

    def _notify_pane_changed(self) -> None:
        try:
            notify = getattr(self.app, "on_pane_scrolled", None)
            if notify is not None:
                notify(self)
        except Exception:  # noqa: BLE001 - 知情通道 fail-soft（缺失/异常不塌滚动）
            pass

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)  # 基座行为（滚动条同步/重绘）保持
        self._notify_pane_changed()

    def watch_virtual_size(self, old_value: Size, new_value: Size) -> None:
        """内容尺寸变了（布局落定/内容增减）→ 位置指示需重算（读的是 max_scroll_y）。"""
        self._notify_pane_changed()

    def on_resize(self, event: events.Resize) -> None:
        """面板尺寸变化（隐藏→显示、窗口缩放）→ ``max_scroll_y`` 随之变，指示需重算。"""
        self._notify_pane_changed()


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


def card(
    *lines: str, id: str | None = None, classes: str = "", markup: bool = True
) -> Static:
    """圆角边框卡：多行文本 ``Static``（组合面按内容定高）。

    ``markup=False`` 关 Rich markup 解析——卡内是**纯文本**：含 ``[sim]``/``[!]`` 这类
    方括号 token 时会被当标签吃掉（s7 实测），看板卡一律显式关闭。
    """
    return Card(
        "\n".join(str(ln) for ln in lines), id=id, classes=classes, markup=markup
    )


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
    Row .row-lead-cell {{ width: 2; }}
    Row .row-badge-cell {{ width: auto; {_T_HINT} padding-right: 1; }}
    Row .row-title-cell {{ width: 1fr; }}
    Row .row-meta-cell {{ width: auto; {_T_SUBTLE} text-align: right; }}
    """


def make_row(
    badge: str,
    title: str,
    meta: str = "",
    indent: int = 0,
    markup: bool = True,
    lead: str = "",
) -> Horizontal:
    """统一列表行：〔连线格〕＋ 徽标（auto 宽）＋ 标题（1fr）＋ 右对齐 meta（auto 宽）。

    ``lead``：行首**树状连线格**（``├``/``└``），格宽由 CSS 固定（``.row-lead-cell``）——
    同族各行连线落在**同一显示列**（徽标串长度不一也不会把连线顶歪；用户 2026-10-16 实机反馈）。
    ``indent`` 控标题左侧留白（家族缩进等场景）；``markup=False`` 关闭 Rich markup 解析
    ——徽标形如 ``[i]`` 时会被当**斜体标签**吃掉（s7 实测），纯文本旗标须显式关。
    """
    prefix = " " * max(0, int(indent))
    cells = []
    if lead:
        cells.append(Static(str(lead), classes="row-lead-cell", markup=markup))
    cells += [
        Static(str(badge), classes="row-badge-cell", markup=markup),
        Static(f"{prefix}{title}", classes="row-title-cell", markup=markup),
        Static(str(meta), classes="row-meta-cell", markup=markup),
    ]
    return Row(*cells)


class ChoiceCard(Static):
    """盲审大选项卡：``-selected`` class 高亮切换 + 点击发 :class:`Chosen`。

    选中态由交互流程代码经 :meth:`toggle_selected`/:meth:`set_selected` 驱动
    （点击本身只上报 ``value``，不自作主张改态）。
    """

    DEFAULT_CSS = f"""
    ChoiceCard {{ {_T_BORDER} {_T_BG} {_T_INSET} height: auto; }}
    ChoiceCard.-selected {{ {_T_HL} {_T_HL_FG} }}
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
        self,
        value: str,
        label: str,
        *,
        id: str | None = None,
        classes: str = "",
        markup: bool = True,
    ) -> None:
        super().__init__(label, id=id, classes=classes, markup=markup)
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


# ---------------------------------------------------------------------------
# 值感知复选（用户 2026-10-17「要对勾不要错叉」）
# ---------------------------------------------------------------------------
_SelT = TypeVar("_SelT")


class Check(Checkbox):
    """Checkbox：勾选标记**值感知**（选中 ``√`` / 未选空白）。

    Textual 8.2.8 的 ``ToggleButton._button`` 无论值与否恒渲染 ``▐X▌``（选中只靠
    颜色区分）——覆写为按 ``value`` 取 :data:`theme._T_CHECK_ON`/``_T_CHECK_OFF``。
    """

    @property
    def _button(self) -> Content:
        button_style = self.get_visual_style("toggle--button")
        side_style = Style(
            foreground=button_style.background,
            background=self.background_colors[1],
        )
        inner = _T_CHECK_ON if self.value else _T_CHECK_OFF
        return Content.assemble(
            (self.BUTTON_LEFT, side_style),
            (inner, button_style),
            (self.BUTTON_RIGHT, side_style),
        )


class CheckList(SelectionList[_SelT]):
    """SelectionList：勾选标记**值感知**（选中 ``√`` / 未选空白）。

    覆写 ``render_line`` 在基类 Strip 上**只换第 2 段**（左沿/内芯/右沿三段里的内芯）
    ——基类把它硬编为 ``ToggleButton.BUTTON_INNER``（恒 ``X``），字形与选中态无关。
    继承须**参数化**（``SelectionList[_SelT]``）：裸继承会丢泛型性，
    ``CheckList[int](…)`` 构造即 TypeError。
    """

    def render_line(self, y: int):
        strip = super().render_line(y)
        segs = list(strip)
        if len(segs) < 3:
            return strip
        _, scroll_y = self.scroll_offset
        try:
            sel = self.get_option_at_index(scroll_y + y)
        except Exception:  # noqa: BLE001 - 越界行照基类渲染（半行/空行）
            return strip
        on = sel.value in self.selected
        segs[1] = type(segs[1])(_T_CHECK_ON if on else _T_CHECK_OFF, segs[1].style)
        return type(strip)(segs)


__all__ = [
    "Card",
    "ChainListItem",
    "Check",
    "CheckList",
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
