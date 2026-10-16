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

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from rich.text import Text
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
    _T_HEADING,
    _T_HINT,
    _T_HL,
    _T_HL_FG,
    _T_INSET,
    _T_SUBTLE,
    _T_TITLE,
)
from .widgets import (
    ChainListItem,
    ChoiceCard,
    MarkedTree,
    MarkedVerticalScroll,
    RowActivated,
    RowListView,
    ScrollMarkMixin,
    card,
    make_row,
    section_title,
)

#: VS Code 启动器候选名：Windows 上 `code` 实为 **`code.cmd`**——`subprocess.Popen` 直呼
#: `.cmd` 会 WinError 2（CreateProcess 不经 cmd 解释器），故解析到 `.cmd`/`.bat` 须经 `cmd /c`。
_VSCODE_NAMES: tuple[str, ...] = ("code", "code.cmd", "code.exe", "code.bat")

#: VS Code 常见安装位（`which` 与 PATH 目录扫描都落空时的**兜底**）。`{VAR}` 按环境展开；
#: 最后一条是**非标准安装位**（用户 2026-10-16 实测环境：VSCode 在 D 盘自定目录）——
#: 它命中过才说明「只有它可得」，删不得。
_VSCODE_FALLBACK_DIRS: tuple[str, ...] = (
    r"{LOCALAPPDATA}\Programs\Microsoft VS Code\bin",
    r"{ProgramFiles}\Microsoft VS Code\bin",
    r"{ProgramFiles(x86)}\Microsoft VS Code\bin",
    r"D:\XiGPrograms\vsCode\base\bin",
)


def _expand_env(template: str, env: dict[str, str]) -> str:
    """`{VAR}` 占位按环境展开；缺少该变量则回空串（跳过该候选）。"""
    out = template
    while "{" in out and "}" in out:
        head, _, rest = out.partition("{")
        name, _, tail = rest.partition("}")
        value = str(env.get(name) or "")
        if not value:
            return ""
        out = f"{head}{value}{tail}"
    return out


def resolve_vscode_launcher(
    *,
    which=None,
    env: dict[str, str] | None = None,
    is_file=None,
) -> tuple[str, list[str]]:
    """解析 VS Code 启动器 → ``(启动器路径, 尝试过的候选清单)``；找不到回 ``("", 清单)``。

    解析链（用户 2026-10-16 实测：PATH 里有 VSCode 的 `bin` 目录，但旧实现
    `which("code")` 落空后仍裸名 Popen → WinError 2）：
    ① ``which(<候选名>)``（PATH × PATHEXT，含 .cmd/.exe/.bat）；② 逐个 PATH 目录找候选文件；
    ③ 常见安装位兜底。``which``/``is_file``/``env`` 可注入（测试用；缺省读真实环境）。
    """
    find = which if which is not None else shutil.which
    e = env if env is not None else dict(os.environ)
    check = is_file if is_file is not None else (lambda p: p.is_file())
    tried: list[str] = []
    for name in _VSCODE_NAMES:
        try:
            hit = find(name)
        except Exception:  # noqa: BLE001 - which 抛异常按未命中（观察面不炸）
            hit = None
        if hit:
            return str(hit), [*tried, f"{name}（PATH/PATHEXT）"]
        tried.append(f"{name}（PATH/PATHEXT）")
    for raw in str(e.get("PATH") or "").split(os.pathsep):
        d = raw.strip()
        if not d:
            continue
        for name in ("code.cmd", "code.exe", "code.bat"):
            cand = Path(d) / name
            if check(cand):
                return str(cand), [*tried, f"{cand}（PATH 目录扫描）"]
        tried.append(f"{d}（PATH 目录扫描）")
    for template in _VSCODE_FALLBACK_DIRS:
        d = _expand_env(template, e)
        if not d:
            continue
        for name in ("code.cmd", "code.exe"):
            cand = Path(d) / name
            if check(cand):
                return str(cand), [*tried, f"{cand}（常见安装位）"]
        tried.append(f"{d}（常见安装位）")
    return "", tried


def launch_editor(launcher: str, target: Path, *, popen=None) -> tuple[bool, str]:
    """拉起启动器打开 ``target``：``.cmd``/``.bat`` 经 **`cmd /c`**，其余直呼。

    Returns:
        ``(成功?, argv 串或错误串)``——失败串取 ``类型: 消息``（调用方转消息行）。
    """
    argv = (
        ["cmd", "/c", launcher, str(target)]
        if Path(launcher).suffix.lower() in (".cmd", ".bat")
        else [launcher, str(target)]
    )
    spawn = popen if popen is not None else subprocess.Popen
    try:
        spawn(argv)
    except Exception as exc:  # noqa: BLE001 - 启动失败如实报（不抛给 UI 事件循环）
        return False, f"{type(exc).__name__}: {exc}"
    return True, " ".join(argv)


#: Textual widget id 只允许字母/数字/下划线/连字符——plan id 可能含点号等（如
#: `rework-xxx.plan` 派生的 id），故列表条目 id 一律过它（非法字符折 `-`）。
_ID_SAFE_RE = re.compile(r"[^0-9A-Za-z_-]+")


def _widget_id(text: object) -> str:
    """任意标识串 → 合法 widget id 片段。"""
    return _ID_SAFE_RE.sub("-", str(text)).strip("-") or "x"


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
    "plans": ("#plans-list", "#plans-scroll"),
    "plan-detail": ("#plan-detail-body", "#plan-detail-scroll"),
    "decisions": ("#decisions-list", "#decisions-scroll"),
    "decision-detail": ("#decision-detail-body", "#decision-detail-scroll"),
    "review-blind": ("#blind-body", "#blind-scroll"),
    "review-rulings": ("#rulings-list", "#rulings-scroll"),
    "ruling-detail": ("#ruling-detail-body", "#ruling-detail-scroll"),
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


class _RowMarkedList(ScrollMarkMixin, RowListView):
    """行式列表（滚动知情＋行激活归一）：s2 ``ScrollMarkMixin`` ＋ s2 ``RowListView``。

    组合理由：列表既要报滚动位置（`#tb-scroll` / `#plans-scroll`）又要区分键 / 单击 /
    双击（``RowActivated.source/chain``）。两个 mixin 都来自 s2 共享层，此处只做组合。
    用于：任务书分组列表（s5）、计划列表（s6）。
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
        # 盲审选项键盘可达（只在 review-blind 视图内生效；输入框聚焦时按键归文本输入）
        Binding("1", "blind_pick_a", "选项 A", show=False),
        Binding("2", "blind_pick_b", "选项 B", show=False),
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
    /* 选中态可读性（用户 2026-10-16 实机反馈）：橙底上白字/橙徽标不可辨 → 一律压深色。
       覆盖面＝**全部列表**（主菜单/任务书/计划/决策/已裁决；行内三格与旧 class 名都管）。 */
    ListItem.-highlight .row-lead-cell,
    ListItem.-highlight .row-title,
    ListItem.-highlight .row-badge,
    ListItem.-highlight .row-title-cell,
    ListItem.-highlight .row-badge-cell,
    ListItem.-highlight .row-meta-cell {{ {_T_HL_FG} }}
    /* Markdown 各级标题：Textual 默认 h1–h3 是 $primary 蓝（#0178D4），深底不可辨 → 亮黄 */
    MarkdownH1, MarkdownH2, MarkdownH3, MarkdownH4, MarkdownH5, MarkdownH6 {{ {_T_HEADING} }}
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
    #plans {{ height: 1fr; {_T_INSET} {_T_BORDER} {_T_BG} }}
    #plans-title {{ {_T_TITLE} }}
    #plans-list {{ height: 1fr; }}
    #plans-list > ListItem.-highlight {{ {_T_HL} }}
    #plans-scroll {{ {_T_SUBTLE} height: 1; }}
    #plan-detail {{ height: 1fr; {_T_INSET} {_T_BORDER} {_T_BG} }}
    #plan-detail-title {{ {_T_TITLE} {_T_BORDER} {_T_INSET} }}
    #plan-detail-body {{ {_T_BODY} height: 1fr; overflow-y: auto; scrollbar-gutter: stable; }}
    #plan-detail-head {{ {_T_BODY} height: auto; }}
    #plan-detail-tree {{ height: auto; background: transparent; }}
    #plan-detail-md {{ {_T_BODY} height: auto; margin-top: 1; }}
    #plan-detail-scroll {{ {_T_SUBTLE} height: 1; }}
    #detail-cards {{ height: auto; }}
    #detail-cards > Card {{ margin-bottom: 1; }}
    #decisions {{ height: 1fr; {_T_INSET} {_T_BORDER} {_T_BG} }}
    #decisions-title {{ {_T_TITLE} }}
    #decisions-list {{ height: 1fr; }}
    #decisions-list > ListItem.-highlight {{ {_T_HL} }}
    #decisions-scroll {{ {_T_SUBTLE} height: 1; }}
    #decision-detail {{ height: 1fr; {_T_INSET} {_T_BORDER} {_T_BG} }}
    #decision-detail-title {{ {_T_TITLE} {_T_BORDER} {_T_INSET} }}
    #decision-detail-body {{ {_T_BODY} height: 1fr; overflow-y: auto; scrollbar-gutter: stable; }}
    #decision-detail-head {{ {_T_BODY} height: auto; }}
    #decision-detail-md {{ {_T_BODY} height: auto; margin-top: 1; }}
    #decision-detail-scroll {{ {_T_SUBTLE} height: 1; }}
    #review-blind {{ height: 1fr; {_T_INSET} {_T_BORDER} {_T_BG} }}
    #blind-count {{ {_T_TITLE} }}
    #blind-body {{ {_T_BODY} height: 1fr; overflow-y: auto; scrollbar-gutter: stable; }}
    #blind-question {{ {_T_BODY} height: auto; }}
    #blind-evidence {{ {_T_SUBTLE} height: auto; margin-top: 1; }}
    #blind-options {{ height: auto; margin-top: 1; }}
    #blind-a {{ width: 1fr; margin-right: 1; }}
    #blind-b {{ width: 1fr; }}
    #blind-ask {{ height: auto; {_T_INSET} {_T_BORDER_SOFT} margin-top: 1; }}
    #blind-reason-label, #blind-remedy-label {{ {_T_SUBTLE} height: 1; }}
    #blind-reason {{ height: 4; }}
    #blind-requirements {{ height: 3; }}
    #blind-msg {{ color: $success; }}
    #blind-scroll {{ {_T_SUBTLE} height: 1; }}
    #review-rulings {{ height: 1fr; {_T_INSET} {_T_BORDER} {_T_BG} }}
    #rulings-title {{ {_T_TITLE} }}
    #rulings-list {{ height: 1fr; }}
    #rulings-list > ListItem.-highlight {{ {_T_HL} }}
    #rulings-scroll {{ {_T_SUBTLE} height: 1; }}
    #ruling-detail {{ height: 1fr; {_T_INSET} {_T_BORDER} {_T_BG} }}
    #ruling-detail-title {{ {_T_TITLE} {_T_BORDER} {_T_INSET} }}
    #ruling-detail-body {{ {_T_BODY} height: 1fr; overflow-y: auto; scrollbar-gutter: stable; }}
    #ruling-detail-head {{ {_T_BODY} height: auto; }}
    #ruling-detail-md {{ {_T_BODY} height: auto; margin-top: 1; }}
    #ruling-detail-scroll {{ {_T_SUBTLE} height: 1; }}
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
        # 计划与派发（s6）：列表行模型 + 详情三区（头卡/Tree/Markdown）
        self.plan_rows: list[dict] = []
        self.plan_detail_id = ""
        self.plan_detail_title = ""
        self.plan_detail_head = ""
        self.plan_detail_md = ""
        self.plan_detail_scroll = 0
        self._plan_default_node = (
            None  # 首帧后要落到的树节点（见 _apply_plan_default_selection）
        )
        # 决策审计（s7）：列表行模型 + 详情三区
        self.decision_rows: list[dict] = []
        self.decision_detail_id = ""
        self.decision_detail_title = ""
        self.decision_detail_head = ""
        self.decision_detail_md = ""
        self.decision_detail_scroll = 0
        self.decision_detail_scroll_mark = ""
        # 系统看板三处动态读数文本（每秒 tick 只 update 文本；首帧由卡片栈写入）
        self.detail_countdown_text = ""
        self.detail_switches_text = ""
        self.detail_config_text = ""
        # 盲审与审查（s8）：逐案状态机 + 已裁决列表/详情
        self.blind_case_id = ""
        self.blind_count = 0
        self.blind_selected = ""
        self.blind_ask_visible = False
        self.blind_remedy = ""
        self.blind_reason = ""
        self.blind_requirements = ""
        self.blind_msg = ""
        self.blind_last_deviation: dict = {}
        self.ruling_rows: list[dict] = []
        self.ruling_detail_id = ""
        self.ruling_detail_title = ""
        self.ruling_detail_head = ""
        self.ruling_detail_md = ""
        self.ruling_detail_scroll = 0
        self.ruling_detail_scroll_mark = ""
        # Esc 路由（polish-b）：子面板（tb-reject/tb-new）的来源视图——Esc 回「进入前视图」
        self._panel_return = ""
        # VSCode 启动器读数（polish-b）：实际启动器路径 / 尝试过的候选清单
        self.tb_launcher_path = ""
        self.tb_launcher_tried: list[str] = []
        self._plan_default_node = (
            None  # 首帧后要落到的树节点（见 _apply_plan_default_selection）
        )
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
                    Static("", id="detail-body-text"),
                    Vertical(id="detail-cards"),
                    id="detail-body",
                    can_focus=False,
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
                yield _RowMarkedList(id="tb-list")
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
            with Vertical(id="plans"):
                yield Static(
                    "计划与派发：↑↓ 选｜Enter 看详情｜Esc 返回", id="plans-title"
                )
                yield _RowMarkedList(id="plans-list")
                yield Static("", id="plans-scroll")
            with Vertical(id="plan-detail"):
                yield Static("", id="plan-detail-title")
                yield MarkedVerticalScroll(
                    card("", id="plan-detail-head"),
                    MarkedTree("计划", id="plan-detail-tree"),
                    Markdown(id="plan-detail-md"),
                    id="plan-detail-body",
                    can_focus=False,
                )
                yield Static("", id="plan-detail-scroll")
            with Vertical(id="decisions"):
                yield Static(
                    "决策审计：↑↓ 选（[!] 置顶需审计）｜Enter 看详情｜Esc 返回",
                    id="decisions-title",
                )
                yield _RowMarkedList(id="decisions-list")
                yield Static("", id="decisions-scroll")
            with Vertical(id="decision-detail"):
                yield Static("", id="decision-detail-title")
                yield MarkedVerticalScroll(
                    card("", id="decision-detail-head"),
                    Markdown(id="decision-detail-md"),
                    id="decision-detail-body",
                    can_focus=False,
                )
                yield Static("", id="decision-detail-scroll")
            with Vertical(id="review-blind"):
                yield Static("", id="blind-count", markup=False)
                yield MarkedVerticalScroll(
                    Static("", id="blind-question", markup=False),
                    Static("", id="blind-evidence", markup=False),
                    Horizontal(
                        ChoiceCard("A", "", id="blind-a", markup=False),
                        ChoiceCard("B", "", id="blind-b", markup=False),
                        id="blind-options",
                    ),
                    id="blind-body",
                    can_focus=False,
                )
                yield Button("提交", id="blind-submit")
                with Vertical(id="blind-ask"):
                    yield Static(
                        "① 原因批复（与 twin 不一致时必填）", id="blind-reason-label"
                    )
                    yield TextArea(id="blind-reason")
                    yield Static("② 是否需恢复更改", id="blind-remedy-label")
                    yield RadioSet(
                        RadioButton("需恢复", value=True),
                        RadioButton("不需恢复"),
                        RadioButton("其他"),
                        id="blind-remedy",
                    )
                    yield TextArea(id="blind-requirements")
                    yield Button("确认", id="blind-confirm")
                yield Static("", id="blind-msg", markup=False)
                yield Static("", id="blind-scroll")
            with Vertical(id="review-rulings"):
                yield Static(
                    "已裁决事项：↑↓ 选｜Enter 看详情｜Esc 返回", id="rulings-title"
                )
                yield _RowMarkedList(id="rulings-list")
                yield Static("", id="rulings-scroll")
            with Vertical(id="ruling-detail"):
                yield Static("", id="ruling-detail-title")
                yield MarkedVerticalScroll(
                    card("", id="ruling-detail-head"),
                    Markdown(id="ruling-detail-md"),
                    id="ruling-detail-body",
                    can_focus=False,
                )
                yield Static("", id="ruling-detail-scroll")
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
        # 除主菜单外一律先藏（含 s5/s6 新视图）——否则隐藏面板照样占布局（§2.2 首帧）。
        self._hide_all_views()
        self.query_one("#menu", ListView).display = True
        self.query_one("#summary", Vertical).display = True
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
            key = self._mods[idx].key
            if key == "taskbooks":
                await self._open_taskbooks()
            elif key == "plans":
                await self._open_plans()
            elif key == "decisions":
                await self._open_decisions()
            elif key == "review":
                await self._open_review()
            else:
                await self._open_detail(idx)

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
        elif self.view == "plan-detail":
            self.plan_detail_scroll, self.plan_detail_scroll_mark = y, text
        elif self.view == "decision-detail":
            self.decision_detail_scroll, self.decision_detail_scroll_mark = y, text
        elif self.view == "ruling-detail":
            self.ruling_detail_scroll, self.ruling_detail_scroll_mark = y, text
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
        """用 VSCode 打开任务书原文：**多级启动器解析链** ＋ `.cmd` 经 `cmd /c`（2026-10-16 修）。

        失败消息列出**尝试过的候选清单**（一眼看出卡在哪）；成功消息显示**实际启动器路径**。
        ``Popen`` 不等待（不阻塞 UI 事件循环）。
        """
        from .. import taskbooks as _tb

        tid = self.tb_detail_id
        if not tid:
            return
        path = _tb.md_path(tid)
        launcher, tried = resolve_vscode_launcher()
        self.tb_launcher_tried = list(tried)
        if not launcher:
            shown = "、".join(tried[:6]) + (
                f"…（共 {len(tried)} 项）" if len(tried) > 6 else ""
            )
            self._set_tb_detail_msg(
                f"[!] 打开失败：未找到 VS Code 启动器（code/code.cmd/code.exe）"
                f"——已尝试：{shown}"
            )
            return
        self.tb_launcher_path = launcher
        ok, info = launch_editor(launcher, path)
        if not ok:
            self._set_tb_detail_msg(f"[!] 打开失败（启动器 {launcher}）：{info}")
            return
        self._set_tb_detail_msg(f"[√] 已在 VSCode 打开 {path}（启动器 {launcher}）")

    async def action_back(self) -> None:
        """Esc/←：子面板 → **进入前视图**（polish-b：驳回面板→详情页/列表；新增面板→列表）；
        任务书详情页 → 任务书列表；其余子视图 → 主菜单；主菜单 → 退出。"""
        if self.view == "tb-new":
            await self._open_taskbooks()
            return
        if self.view == "tb-reject":
            if self._panel_return == "tb-detail" and self.tb_detail_id:
                self._open_tb_detail_by_id(self.tb_detail_id)
            else:
                await self._open_taskbooks()
            return
        if self.view == "tb-detail":
            await self._open_taskbooks()
        elif self.view == "plan-detail":
            await self._open_plans()
        elif self.view == "decision-detail":
            await self._open_decisions()
        elif self.view == "ruling-detail":
            await self._open_rulings()
        elif self.view in ("review-blind", "review-rulings"):
            self._show_menu()
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
            "#plans",
            "#plan-detail",
            "#decisions",
            "#decision-detail",
            "#review-blind",
            "#review-rulings",
            "#ruling-detail",
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

    async def _open_detail(self, idx: int) -> None:
        mod = self._mods[idx]
        self.view = "detail"
        self.selected = idx
        blocks = self._detail_blocks_for(mod.key)
        if blocks is None:  # 无卡片规格的模块：沿用文本面（s3 语义不变）
            try:
                body = list(mod.detail())
            except Exception as exc:  # noqa: BLE001
                body = [f"读数失败：{type(exc).__name__}: {exc}"]
            self._render_detail(mod, body)
        else:  # s7 看板：卡片栈（卡=card()、分区=section_title()）
            await self._render_detail_cards(mod, blocks)
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
        # 文本面与卡片栈互斥（s7 看板走卡片，其余模块照旧文本）
        self._show_detail_text()

    def _show_detail_text(self) -> None:
        try:
            self.query_one("#detail-body-text", Static).display = True
            self.query_one("#detail-cards", Vertical).display = False
        except Exception:  # noqa: BLE001 - 组合面容忍缺件
            return

    def _show_detail_cards(self) -> None:
        try:
            self.query_one("#detail-body-text", Static).display = False
            self.query_one("#detail-cards", Vertical).display = True
        except Exception:  # noqa: BLE001 - 组合面容忍缺件
            return

    # ---- 看板卡片栈（s7）--------------------------------------------------
    def _detail_blocks_for(self, key: str) -> list[dict] | None:
        """该模块的卡片块（无规格 → ``None``＝沿用文本面）；读数坏→单张错误卡。"""
        from . import cards as _cards

        builder = {
            "queues": _cards.queues_blocks,
            "system": _cards.system_blocks,
            "pending": _cards.pending_blocks,
        }.get(key)
        if builder is None:
            return None
        try:
            return builder()
        except Exception as exc:  # noqa: BLE001 - 看板读数坏不塌详情
            return [
                {
                    "kind": "card",
                    "lines": [f"[!] 看板读数失败：{type(exc).__name__}: {exc}"],
                    "id": "card-error",
                }
            ]

    @staticmethod
    def _blocks_text(blocks: list[dict]) -> list[str]:
        """块 → 文本面（``detail_body`` 读面；分区行 `== 分区 ==` 与 section_title 同语义）。"""
        lines: list[str] = []
        for b in blocks:
            if b.get("kind") == "section":
                lines.append(f"== {b.get('text')} ==")
            else:
                lines.extend(str(ln) for ln in (b.get("lines") or []))
        return lines

    async def _render_detail_cards(self, mod: Module, blocks: list[dict]) -> None:
        """卡片栈渲染（卡=``card()``、分区=``section_title()``）＋同步文本面 ``detail_body``。"""
        box = self.query_one("#detail-cards", Vertical)
        await box.remove_children()
        widgets: list[Widget] = []
        for b in blocks:
            if b.get("kind") == "section":
                widgets.append(section_title(str(b.get("text") or "")))
                continue
            widgets.append(
                card(
                    *[str(ln) for ln in (b.get("lines") or [])],
                    id=str(b["id"]) if b.get("id") else None,
                    classes="board-card",
                    markup=False,  # 卡内纯文本（`[sim]`/`[!]` 别被当 Rich 标签吃掉）
                )
            )
        if widgets:
            await box.mount(*widgets)
        self._show_detail_cards()
        self.detail_title = f"{mod.title}\u3000详情"
        text_lines = self._blocks_text(blocks)
        self.detail_body = "\n".join(text_lines)
        self.detail_hint = ""
        self.detail_text = "\n".join([self.detail_title, "", *text_lines])
        self.query_one("#detail-title", Static).update(self.detail_title)
        # 动态卡文本进可断言状态面（每秒 tick 只 update 文本）
        for b in blocks:
            lines = [str(ln) for ln in (b.get("lines") or [])]
            if b.get("id") == "detail-countdown":
                self.detail_countdown_text = (lines or [""])[0].strip()
            elif b.get("id") == "card-switches":
                self.detail_switches_text = "\n".join(lines).strip()
            elif b.get("id") == "card-config":
                self.detail_config_text = "\n".join(lines).strip()

    def _tick_detail(self) -> None:
        """逐秒刷新：**系统看板三处时长读数**（倒计时／距峰／距边界）——只 update 文本。

        tui-polish-c 反馈 #6：旧实现只刷倒计时卡，同页「距峰 … 还有 59.3 小时」「距边界
        3550 分」原地不动，同一屏三处倒计时不同步。现三处各由卡片文本单点重算（卡控件
        对象不动，防闪烁），并同步进可断言状态面。
        徽标指纹仍走 `_system_stable`/`_COUNTDOWN_MARK` 排除语义（倒计时跳动不常亮）。
        """
        if self.view != "detail":
            return
        mod = self._mods[self.selected]
        if mod.key != "system":
            return
        self._refresh_countdown_card()
        self._refresh_system_dynamic_cards()
        self._mark_seen(mod)  # 用户正看着：把最新读数记为已见，徽标不自亮（同值不写盘）

    def _refresh_system_dynamic_cards(self) -> None:
        """刷「开关/路由」与「配置生效值」两卡（各含一处每秒变化的时长读数）。"""
        from . import cards as _cards

        for card_id, provider, attr in (
            ("card-switches", _cards.switches_lines, "detail_switches_text"),
            ("card-config", _cards.routing_config_lines, "detail_config_text"),
        ):
            try:
                lines = list(provider())
            except Exception:  # noqa: BLE001 - 刷新失败保持上一帧，不塌详情
                continue
            setattr(self, attr, "\n".join(lines).strip())
            try:
                self.query_one(f"#{card_id}", Static).update("\n".join(lines))
            except Exception:  # noqa: BLE001 - 组合面容忍缺件
                continue
        self._refresh_scroll_mark()

    def _refresh_countdown_card(self) -> None:
        """只刷倒计时卡（其余卡与其控件对象保持不动）。"""
        from . import cards as _cards

        try:
            text = _cards.countdown_text()
        except Exception:  # noqa: BLE001 - 刷新失败保持上一帧，不塌详情
            return
        self.detail_countdown_text = text.strip()
        try:
            self.query_one("#detail-countdown", Static).update(text)
        except Exception:  # noqa: BLE001 - 组合面容忍缺件
            return
        self._refresh_scroll_mark()  # 每秒只更读数，**不动滚动位置**（用户正读的那屏不跳）

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
        self.query_one("#tb-list", _RowMarkedList).focus()
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
                    "no": int(fam.get("no") or 0),  # 家族编号（按 root 创建序，稳定）
                    "title": str(latest.get("title") or ""),
                    "badge": _tb.status_badge(str(latest.get("status") or "")),
                    "size": int(fam["size"]),
                    "round": int(latest.get("rework_round") or 0),
                }
            )
            if not self.tb_expanded.get(root):
                continue
            for i, m in enumerate(members):
                rows.append(
                    {
                        "kind": "sub",
                        "tb_id": str(m.get("id") or ""),
                        "root": root,
                        "label": _tb.family_label(m),
                        "indent": 1,
                        # 末子项用 `└`、其余用 `├`（树状连线；展示序即 members 序）
                        "lead": "└" if i == len(members) - 1 else "├",
                        "title": str(m.get("title") or ""),
                        "badge": _tb.status_badge(str(m.get("status") or "")),
                    }
                )
        return rows

    def _tb_list_item(self, row: dict) -> ChainListItem:
        """行 → 列表条目（``make_row`` 统一形态：徽标 | 标题 | 右对齐 meta）。"""
        if row["kind"] == "group":
            meta = f"共 {row['size']} 件" if int(row["size"]) > 1 else "本体"
            no = int(row.get("no") or 0)
            # 家族编号只挂**主项**（子项不重复；tui-polish-c 反馈 #9）——前缀进标题格，
            # 不碰 lead/徽标列，故 polish-a 的列宽纪律（定宽连线列）不受影响。
            title = f"#{no} {row['title']}" if no else row["title"]
            return ChainListItem(
                make_row(row["badge"], title, meta=meta),
                id=f"tbrow-{_widget_id(row['tb_id'])}-{row['kind']}",
            )
        return ChainListItem(
            make_row(
                row["badge"],
                f"{row['label']}\u3000{row['title']}",
                lead=str(row.get("lead") or ""),
                markup=False,
            ),
            id=f"tbrow-{_widget_id(row['tb_id'])}-{row['kind']}",
        )

    async def _rebuild_tb_list(self, *, keep_root: str | None = None) -> None:
        """按行模型重建 ``#tb-list``；``keep_root`` 给出时把高亮保持在**该组行**上。

        ``ListView.clear()`` 是 **AwaitComplete**（移除在事件循环里异步完成）——必须
        ``await`` 它再挂新行，否则旧行尚未移除、新行已挂载，撞 ``DuplicateIds``（实测）。
        """
        self.tb_rows = self._build_tb_rows()
        try:
            lv = self.query_one("#tb-list", _RowMarkedList)
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
        """行激活分派（s2 ``RowListView``：键/单击/双击**同型**消息，键鼠可辨）。

        - **计划列表**（s6）：单击/Enter → 进该项计划详情；
        - **任务书分组列表**（s5）：组行 Enter→立即展开/收起；组行单击→延迟展开/收起
          （双击到达即取消）；组行双击（chain≥2）→ 取消延迟、直接进最新项详情；
          次级行（单击/Enter）→ 立即进该项详情。
        """
        if event.list_view.id == "plans-list":
            if self.view != "plans":
                return
            index = int(event.index)
            if 0 <= index < len(self.plan_rows):
                self._open_plan_detail(str(self.plan_rows[index].get("id") or ""))
            return
        if event.list_view.id == "rulings-list":
            if self.view != "review-rulings":
                return
            index = int(event.index)
            if 0 <= index < len(self.ruling_rows):
                self._open_ruling_detail(self.ruling_rows[index])
            return
        if event.list_view.id == "decisions-list":
            if self.view != "decisions":
                return
            index = int(event.index)
            if 0 <= index < len(self.decision_rows):
                self._open_decision_detail(self.decision_rows[index])
            return
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
            return int(self.query_one("#tb-list", _RowMarkedList).index or 0)
        except Exception:  # noqa: BLE001 - 组合面容忍缺件
            return 0

    def action_tb_new(self) -> None:
        if self.view != "taskbooks":
            return
        self.view = "tb-new"
        self._panel_return = "taskbooks"  # Esc 回任务书列表（进入前视图）
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
        # 来源视图（详情页进的面板 Esc 回详情页；列表进的回列表）——须在切视图**前**读
        self._panel_return = (
            self.view if self.view in ("taskbooks", "tb-detail") else ""
        )
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
        if event.button.id == "blind-submit" and self.view == "review-blind":
            await self._submit_blind()
            return
        if event.button.id == "blind-confirm" and self.view == "review-blind":
            await self._confirm_blind_ask()
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

    # ---- 盲审与审查（s8：盲审反转 §10 ＋ 偏差报告 §5.4）----------------------
    async def _open_review(self) -> None:
        """入口分派：**有待盲审 → 直接进盲审界面**；否则 → 已裁决事项列表。"""
        from .. import taste as _taste

        try:
            pending = int(_taste.pending_review_count())
        except Exception:  # noqa: BLE001 - 读数坏按「无待审」处置（不塌入口）
            pending = 0
        if pending > 0:
            await self._open_blind()
        else:
            await self._open_rulings()

    # -- 盲审界面（逐案状态机；提交前**绝不出现 twin 答案**）--
    @staticmethod
    def _blind_queue() -> list[dict]:
        """待盲审队列（`taste.blind_queue()` 天然不含 choice/reason/body）。"""
        from .. import taste as _taste

        try:
            return [dict(r) for r in _taste.blind_queue()]
        except Exception:  # noqa: BLE001 - 读数坏不塌入口
            return []

    async def _open_blind(self) -> None:
        self.view = "review-blind"
        self._hide_all_views()
        self.query_one("#review-blind", Vertical).display = True
        await self._render_blind_case()

    async def _render_blind_case(self) -> None:
        """呈现队首一案：顶部计数 ＋ 案情（问题/证据）＋ 选项大卡 A/B；无待审 → 已裁决列表。"""
        queue = self._blind_queue()
        self.blind_count = len(queue)
        self.blind_selected = ""
        self.blind_ask_visible = False
        self.blind_remedy = ""
        self.blind_reason = ""
        self.blind_requirements = ""
        self.query_one("#blind-ask", Vertical).display = False
        self.query_one("#blind-requirements", TextArea).display = False
        self.query_one("#blind-reason", TextArea).text = ""
        self.query_one("#blind-requirements", TextArea).text = ""
        self._set_blind_msg("")
        if not queue:
            self.blind_case_id = ""
            await self._open_rulings()  # 处理完 → 已裁决事项列表
            return
        item = queue[0]
        self.blind_case_id = str(item.get("case_id") or "")
        self.query_one("#blind-count", Static).update(f"尚待盲审 {self.blind_count} 项")
        self.query_one("#blind-question", Static).update(
            f"问题：{str(item.get('question') or '（无）')}"
        )
        evidence = str(item.get("evidence") or "")
        self.query_one("#blind-evidence", Static).update(
            f"证据：{evidence}" if evidence else ""
        )
        for value, sel in (("A", "a"), ("B", "b")):
            option = str(item.get(f"option_{'a' if value == 'A' else 'b'}") or "（无）")
            card_w = self.query_one(f"#blind-{sel}", ChoiceCard)
            card_w.update(f"选项 {value}\n{option}")
            card_w.set_selected(False)
        self.query_one("#blind-submit", Button).disabled = False

    def _pick_blind_option(self, value: str) -> None:
        """选中选项大卡（点击 / 键盘 1、2 共用；提交前画面只有选项本身）。"""
        if self.view != "review-blind" or self.blind_ask_visible:
            return
        self.blind_selected = str(value)
        for val, sel in (("A", "a"), ("B", "b")):
            try:
                self.query_one(f"#blind-{sel}", ChoiceCard).set_selected(val == value)
            except Exception:  # noqa: BLE001 - 组合面容忍缺件
                continue

    def action_blind_pick_a(self) -> None:
        self._pick_blind_option("A")

    def action_blind_pick_b(self) -> None:
        self._pick_blind_option("B")

    def on_choice_card_chosen(self, event: ChoiceCard.Chosen) -> None:
        self._pick_blind_option(str(event.value))

    def _blind_remedy_value(self) -> str:
        """三选一读数：``需恢复``→restore｜``不需恢复``→none｜``其他``→other。"""
        idx = self._radio_index("#blind-remedy", 0)
        self.blind_remedy = ("restore", "none", "other")[min(idx, 2)]
        return self.blind_remedy

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """「其他」才显示 `#blind-requirements`（具体要求多行框）。"""
        if getattr(event.radio_set, "id", "") != "blind-remedy":
            return
        self._blind_remedy_value()
        self.query_one("#blind-requirements", TextArea).display = (
            self.blind_remedy == "other"
        )

    async def _submit_blind(self) -> None:
        """「提交」：与 twin 裁决比对——一致直接确认；不一致展开追问区（不预显答案）。"""
        from .. import taste as _taste

        cid = self.blind_case_id
        user = self.blind_selected
        if not cid or user not in ("A", "B"):
            self._set_blind_msg("请先选中 选项 A / 选项 B（点击卡片或按 1 / 2）再提交")
            return
        rec = _taste.blind_case(cid) or {}
        twin = str(rec.get("choice") or "")
        if twin and user == twin:
            rc, msg = _taste.apply_review(
                cid,
                verdict="confirmed",
                note="盲审与 twin 一致",
                blind={
                    "user_choice": user,
                    "agree": True,
                    "remedy": None,
                    "requirements": "",
                },
            )
            self._set_blind_msg(msg if rc == 0 else f"[!] {msg}")
            await self._render_blind_case()  # 自动呈现下一项
            return
        self.blind_ask_visible = True
        self.query_one("#blind-ask", Vertical).display = True
        self.query_one("#blind-reason", TextArea).focus()
        self._set_blind_msg("与 twin 不一致：填 ① 原因批复、② 是否需恢复更改后「确认」")

    async def _confirm_blind_ask(self) -> None:
        """追问确认：复核留痕 ＋ 偏差报告入强制队列；**需恢复**额外建「偏差纠正」任务书。"""
        from .. import mirror as _mirror
        from .. import taste as _taste

        cid = self.blind_case_id
        if not cid:
            return
        user = self.blind_selected
        reason = self.query_one("#blind-reason", TextArea).text.strip()
        remedy = self._blind_remedy_value()
        requirements = ""
        if remedy == "other":
            requirements = self.query_one("#blind-requirements", TextArea).text.strip()
        self.blind_reason, self.blind_requirements = reason, requirements
        rec = _taste.blind_case(cid) or {}
        twin = str(rec.get("choice") or "")
        rc, msg = _taste.apply_review(
            cid,
            verdict="rejected",
            note=reason,
            blind={
                "user_choice": user,
                "agree": False,
                "remedy": remedy,
                "requirements": requirements,
            },
        )
        if rc != 0:
            self._set_blind_msg(f"[!] {msg}")
            return
        tb_id = ""
        if remedy == "restore":
            tb_id = self._create_deviation_taskbook(
                cid, user, twin, reason, requirements
            )
        deviation = {
            "case_id": cid,
            "user_choice": user,
            "twin_choice": twin,
            "reason": reason,
            "remedy": remedy,
            "requirements": requirements,
        }
        if tb_id:
            deviation["taskbook_id"] = tb_id
        try:
            line = _mirror.append_deviation(deviation)
            self.blind_last_deviation = dict(line)
        except Exception as exc:  # noqa: BLE001 - 入队失败如实报（复核留痕已落）
            self._set_blind_msg(f"[!] 偏差报告入队失败：{type(exc).__name__}: {exc}")
            return
        tail = f"；已建偏差纠正任务书 {tb_id}" if tb_id else ""
        self._set_blind_msg(f"[√] 已记录偏差报告（{cid}）{tail}")
        await self._render_blind_case()  # 自动呈现下一项

    def _create_deviation_taskbook(
        self, cid: str, user: str, twin: str, reason: str, requirements: str
    ) -> str:
        """「需恢复」→ TUI 直接建「偏差纠正」任务书（priority=high；落盘即 TASKBOOK-NEW）。

        Returns:
            新任务书 id（建失败回空串——偏差报告照旧入队，只是没有回填）。
        """
        from .. import taskbooks as _tb

        body = (
            f"## 偏差来源\n盲审 case {cid}：用户盲审与 twin 裁决不一致，用户判定需恢复更改。\n\n"
            f"## 选择对照\n- 用户盲审选择：{user}\n- twin 裁决：{twin or '（未读出）'}\n\n"
            f"## 原因批复\n{reason or '（无）'}\n\n"
            f"## 实施要求\n{requirements or '按用户选择恢复原行为（无额外要求）'}\n"
        )
        try:
            st = _tb.create(
                f"偏差纠正：{cid}", body, source="mirror-tui", priority="high"
            )
        except ValueError as exc:  # noqa: BLE001 - 建件失败不塌流程（仍入队留痕）
            self._set_blind_msg(f"[!] 建偏差纠正任务书失败：{exc}")
            return ""
        return str(st.get("id") or "")

    def _set_blind_msg(self, text: str) -> None:
        self.blind_msg = text
        try:
            self.query_one("#blind-msg", Static).update(text)
        except Exception:  # noqa: BLE001 - 组合面容忍缺件
            return

    # -- 已裁决事项列表 ＋ 详情 --
    async def _open_rulings(self) -> None:
        self.view = "review-rulings"
        await self._rebuild_rulings_list()
        self._hide_all_views()
        self.query_one("#review-rulings", Vertical).display = True
        self.query_one("#rulings-list", _RowMarkedList).focus()

    async def _rebuild_rulings_list(self) -> None:
        from .. import taste as _taste

        try:
            self.ruling_rows = [dict(r) for r in _taste.ruling_rows()]
        except Exception:  # noqa: BLE001 - 读数失败不该塌面板
            self.ruling_rows = []
        try:
            lv = self.query_one("#rulings-list", _RowMarkedList)
        except Exception:  # noqa: BLE001 - 组合面容忍缺件
            return
        await lv.clear()
        if not self.ruling_rows:
            lv.append(ChainListItem(Static("（暂无裁决记录）"), id="rulings-empty"))
        else:
            for r in self.ruling_rows:
                rev = r.get("review") or {}
                flag = (
                    "confirmed"
                    if str(rev.get("verdict")) == "confirmed"
                    else "rejected"
                )
                lv.append(
                    ChainListItem(
                        make_row(
                            flag,
                            str(r.get("ts") or "")[:19],
                            meta=str(r.get("case_head") or "")[:40],
                            markup=False,
                        ),
                        id=f"rulrow-{_widget_id(r.get('case_id'))}",
                    )
                )
        lv.index = 0 if self.ruling_rows else None
        self._reset_scroll()

    def _open_ruling_detail(self, row: dict) -> None:
        """裁决详情：头卡（case_id/ts/结论/night/authority）＋ Markdown（含 twin 答案：盲审已过）。"""
        if not isinstance(row, dict) or not row.get("case_id"):
            return
        cid = str(row.get("case_id"))
        self.ruling_detail_id = cid
        self.ruling_detail_title = f"裁决 {cid}"
        self.ruling_detail_head = self._ruling_head_text(row)
        self.view = "ruling-detail"
        self._hide_all_views()
        self.query_one("#ruling-detail", Vertical).display = True
        self.query_one("#ruling-detail-title", Static).update(self.ruling_detail_title)
        self.query_one("#ruling-detail-head", Static).update(self.ruling_detail_head)
        self._set_ruling_detail_md(self._ruling_markdown(row))
        self._reset_scroll()

    def _ruling_head_text(self, row: dict) -> str:
        rev = row.get("review") or {}
        lines = [
            f"裁决 {row.get('case_id')}",
            f"时间：{row.get('ts') or '（无）'}"
            f"\u3000复核结论：{rev.get('verdict') or '（无）'}",
        ]
        if row.get("night"):
            lines.append(f"夜航：{row.get('authority') or '夜航预授权'}")
        if rev.get("ts"):
            lines.append(f"复核时刻：{rev.get('ts')}")
        return "\n".join(lines)

    @staticmethod
    def _ruling_case_info(row: dict) -> dict:
        """案情（问题/两选项）：读案卷解析；缺失回落 ``case_head``（不崩）。"""
        from .. import jury as _jury
        from .. import taste as _taste

        record = _taste.blind_case(str(row.get("case_id"))) or {}
        case_file = str(record.get("case_file") or "")
        try:
            case = _jury.parse_case(case_file)
            return {
                "question": case.question,
                "option_a": case.option_a,
                "option_b": case.option_b,
            }
        except (OSError, ValueError):
            return {
                "question": str(row.get("case_head") or ""),
                "option_a": "",
                "option_b": "",
            }

    def _ruling_markdown(self, row: dict) -> str:
        rev = row.get("review") or {}
        blind = rev.get("blind") if isinstance(rev.get("blind"), dict) else {}
        info = self._ruling_case_info(row)
        out = [f"# 裁决 {row.get('case_id')}", "", "## 案情", ""]
        out.append(str(info.get("question") or row.get("case_head") or "（无）"))
        if info.get("option_a") or info.get("option_b"):
            out += [
                "",
                f"- 选项 A：{info.get('option_a') or '（无）'}",
                f"- 选项 B：{info.get('option_b') or '（无）'}",
            ]
        conf = row.get("confidence")
        out += [
            "",
            "## twin 裁决（盲审已过，此处可显）",
            "",
            f"- 选择：{row.get('choice') or '（未读出）'}",
            f"- 置信度：{'-' if conf in (None, '') else conf}",
            f"- 理由：{row.get('reason') or '（无）'}",
        ]
        if blind:
            agree = bool(blind.get("agree"))
            out += [
                "",
                "## 你的盲审",
                "",
                f"- 选择：{blind.get('user_choice') or '（无）'}",
                f"- 与 twin 一致：{'是' if agree else '否'}",
            ]
            if not agree:
                out += [
                    f"- 是否需恢复更改：{mirror._remedy_label(str(blind.get('remedy') or ''))}",
                    f"- 原因批复：{rev.get('note') or '（无）'}",
                    f"- 具体要求：{blind.get('requirements') or '（无）'}",
                ]
        elif rev.get("note"):
            out += ["", "## 复核批语", "", str(rev.get("note"))]
        return "\n".join(out).rstrip() + "\n"

    def _set_ruling_detail_md(self, text: str) -> None:
        self.ruling_detail_md = text
        try:
            self.query_one("#ruling-detail-md", Markdown).update(text)
        except Exception:  # noqa: BLE001 - 组合面容忍缺件
            return

    # ---- 决策审计（s7：列表 → 详情：头卡＋Markdown）--------------------------
    def _build_decision_rows(self) -> list[dict]:
        """行模型：``inspectd.decision_rows()``（已 attention 置顶、其余 ts 倒序）。"""
        from .. import inspectd as _insp

        try:
            return [dict(r) for r in _insp.decision_rows()]
        except Exception:  # noqa: BLE001 - 读数失败不该塌面板
            return []

    async def _open_decisions(self) -> None:
        self.view = "decisions"
        await self._rebuild_decisions_list()
        self._hide_all_views()
        self.query_one("#decisions", Vertical).display = True
        self.query_one("#decisions-list", _RowMarkedList).focus()

    async def _rebuild_decisions_list(self) -> None:
        """按行模型重建 ``#decisions-list``（async：``ListView.clear()`` 异步完成，s5 注记）。"""
        self.decision_rows = self._build_decision_rows()
        try:
            lv = self.query_one("#decisions-list", _RowMarkedList)
        except Exception:  # noqa: BLE001 - 组合面容忍缺件
            return
        await lv.clear()
        if not self.decision_rows:
            lv.append(ChainListItem(Static("（暂无决策记录）"), id="decisions-empty"))
        else:
            for r in self.decision_rows:
                flag = "[!]" if r.get("attention") else "[i]"
                lv.append(
                    ChainListItem(
                        make_row(
                            flag,
                            f"{str(r.get('ts') or '')[:19]}  "
                            f"{r.get('class')}/{r.get('remedy_shape')}",
                            meta=str(r.get("title") or "")[:40],
                            markup=False,  # 旗标 `[i]` 别被 Rich 当斜体标签吃掉（s7 实测）
                        ),
                        id=f"decrow-{_widget_id(r.get('id'))}",
                    )
                )
        lv.index = 0 if self.decision_rows else None
        self._reset_scroll()

    def _open_decision_detail(self, row: dict) -> None:
        """决策详情页：**头卡**（id/ts/class/shape/attention）＋ **Markdown 正文**。"""
        if not isinstance(row, dict) or not row.get("id"):
            return
        did = str(row.get("id"))
        self.decision_detail_id = did
        self.decision_detail_title = f"决策 {did}"
        self.decision_detail_head = self._decision_head_text(row)
        self.view = "decision-detail"
        self._hide_all_views()
        self.query_one("#decision-detail", Vertical).display = True
        self.query_one("#decision-detail-title", Static).update(
            self.decision_detail_title
        )
        self.query_one("#decision-detail-head", Static).update(
            self.decision_detail_head
        )
        self._set_decision_detail_md(self._decision_markdown(row))
        self._reset_scroll()

    def _decision_head_text(self, row: dict) -> str:
        """头卡文本：id / ts / 类别 / 修复形状 / 是否需审计。"""
        attn = "〔!〕免判决+change——最需人工审计" if row.get("attention") else "常规"
        return "\n".join(
            [
                f"决策 {row.get('id')}",
                f"时间：{row.get('ts') or '（无）'}",
                f"类别：{row.get('class') or '（无）'}"
                f"\u3000修复形状：{row.get('remedy_shape') or '（缺省 change）'}",
                f"关注：{attn}",
            ]
        )

    def _decision_markdown(self, row: dict) -> str:
        """决策详情 Markdown（**有啥写啥**，无值字段省略）。"""
        out = [f"# 决策 {row.get('id')}", ""]
        out += ["## 发现", "", str(row.get("title") or "（无）")]
        ruling = str(row.get("ruling") or "")
        verdict_ref = str(row.get("verdict_ref") or "")
        if verdict_ref or ruling:
            out += ["", "## 判决", ""]
            if verdict_ref:
                out.append(f"- 依据：{verdict_ref}")
            if ruling:
                out.append(f"- 裁示：{ruling}")
        impl: list[str] = []
        if row.get("commit"):
            impl.append(f"- 实施 commit：`{row['commit']}`")
        if row.get("verify"):
            impl.append(f"- 核验：{row.get('verify')}")
        if row.get("backlog_status"):
            impl.append(f"- backlog：{row.get('backlog_status')}")
        if row.get("finding_status"):
            impl.append(f"- 发现状态：{row.get('finding_status')}")
        if row.get("channel"):
            impl.append(f"- 通道：{row.get('channel')}")
        if impl:
            out += ["", "## 实施与核验", "", *impl]
        if row.get("note"):
            out += ["", "## 留痕", "", str(row.get("note"))]
        return "\n".join(out).rstrip() + "\n"

    def _set_decision_detail_md(self, text: str) -> None:
        self.decision_detail_md = text
        try:
            self.query_one("#decision-detail-md", Markdown).update(text)
        except Exception:  # noqa: BLE001 - 组合面容忍缺件
            return

    # ---- 计划与派发（s6：列表 → 详情：可折叠 Tree＋Markdown 面板）--------------
    def _build_plan_rows(self) -> list[dict]:
        """行模型：全部 plan（``plans.overview_rows()`` 序＝新→旧；**只走 state.json 缓存**）。"""
        from .. import plans as _plans

        try:
            return [dict(r) for r in _plans.overview_rows()]
        except Exception:  # noqa: BLE001 - 读数失败不该塌面板
            return []

    async def _open_plans(self) -> None:
        self.view = "plans"
        await self._rebuild_plans_list()
        self._hide_all_views()
        self.query_one("#plans", Vertical).display = True
        self.query_one("#plans-list", _RowMarkedList).focus()

    async def _rebuild_plans_list(self) -> None:
        """按行模型重建 ``#plans-list``（async：``ListView.clear()`` 异步完成，s5 注记）。"""
        self.plan_rows = self._build_plan_rows()
        try:
            lv = self.query_one("#plans-list", _RowMarkedList)
        except Exception:  # noqa: BLE001 - 组合面容忍缺件
            return
        await lv.clear()
        if not self.plan_rows:
            lv.append(ChainListItem(Static("（暂无计划）"), id="plans-empty"))
        else:
            for r in self.plan_rows:
                lv.append(
                    ChainListItem(
                        make_row(
                            str(r.get("mark") or ""),
                            f"{r.get('id')}  {r.get('done')}/{r.get('total')}",
                            meta=str(r.get("goal") or "")[:40],
                        ),
                        id=f"planrow-{_widget_id(r.get('id'))}",
                    )
                )
        lv.index = 0 if self.plan_rows else None
        self._reset_scroll()  # 重建 → 位置行复位（s4 范式）

    def _open_plan_detail(self, pid: str) -> None:
        """计划详情页：**头卡** ＋ **可折叠 Tree 环节树**（s1 ``plan_tree_nodes``）＋
        树下 **Markdown 文本面板**（树节点选中联动；树与 md 冲突时优先保 Tree）。"""
        if not pid:
            return
        row = next((r for r in self.plan_rows if str(r.get("id")) == pid), None)
        self.plan_detail_id = pid
        self.plan_detail_title = f"{pid}　详情"
        self.plan_detail_head = self._plan_head_text(pid, row)
        self.view = "plan-detail"
        self._hide_all_views()
        self.query_one("#plan-detail", Vertical).display = True
        self.query_one("#plan-detail-title", Static).update(self.plan_detail_title)
        self.query_one("#plan-detail-head", Static).update(self.plan_detail_head)
        cur = (row or {}).get("current") or {}
        # 焦点先给树（聚焦会把树光标归到根）：↑↓ 走树光标（选中联动 md）；翻页/首尾键
        # 归全局滚动（Tree 不绑它们）。
        self.query_one("#plan-detail-tree", MarkedTree).focus()
        self._fill_plan_tree(pid, str(cur.get("id") or ""))
        self._reset_scroll()

    def _plan_head_text(self, pid: str, row: dict | None) -> str:
        """详情头卡文本（``card()`` 形态 Static）：goal / 状态徽标 / 进度 / 时间戳。"""
        if row is None:
            return f"计划 {pid}：读数缺失（state.json 不在册？）"
        lines = [
            f"目标：{row.get('goal') or '（无）'}",
            f"状态：{row.get('mark') or '?'}　　进度：{row.get('done')}/{row.get('total')}",
            f"创建：{row.get('created') or '（未记）'}",
        ]
        if row.get("completed_at"):
            lines.append(f"完成：{row['completed_at']}")
        if row.get("abandoned_at"):
            lines.append(f"中止：{row['abandoned_at']}")
        return "\n".join(lines)

    def _plan_node_label(self, n: dict) -> str:
        """节点 label：superstep ＝ ``〔mode〕 id``；叶子 ＝ ``图标 id [status] member｜brief``。

        图标后**隔一个空格**再跟文字（全局规范）；done→``✔``、failed→``✗``、其余 ``·``。
        """
        if n.get("kind") == "super":
            return f"〔{n.get('mode') or 'serial'}〕 {n.get('id')}"
        icon = {"done": "✔", "failed": "✗", "run_failed": "✗"}.get(
            str(n.get("status") or ""), "·"
        )
        member = str(n.get("member") or "") or "self"
        return f"{icon} {n.get('id')} [{n.get('status')}] {member}｜{str(n.get('brief') or '')}"

    def _fill_plan_tree(self, pid: str, current_id: str = "") -> None:
        """填充环节树（**默认全展开**；默认选中＝当前进行中环节，无则首叶/根）。

        树节点携带结构化 dict 作 ``data``——选中联动直接读它，不必再回查。
        """
        from .. import plans as _plans

        tree = self.query_one("#plan-detail-tree", MarkedTree)
        tree.clear()  # 同步 API（重建根子树，无异步残留）
        try:
            nodes = _plans.plan_tree_nodes(pid)
        except Exception as exc:  # noqa: BLE001 - 读数坏不塌详情（观察面 fail-open）
            tree.root.add(f"[!] 环节树读数失败：{type(exc).__name__}: {exc}")
            self._set_plan_detail_md(
                f"# 计划 {pid}\n\n[!] 环节树读数失败：{type(exc).__name__}: {exc}\n"
            )
            return

        picked: dict = {"target": None, "first_leaf": None}

        def add_nodes(items: list, parent) -> None:
            for n in items:
                # Text（非 str）：label 里的 ``[status]`` 会被 Rich markup 当标签吃掉（实测）
                child = parent.add(Text(self._plan_node_label(n)), data=n)
                if n.get("kind") == "leaf":
                    if picked["first_leaf"] is None:
                        picked["first_leaf"] = child
                    if current_id and str(n.get("path")) == current_id:
                        picked["target"] = child
                if n.get("children"):
                    add_nodes(n["children"], child)

        add_nodes(nodes, tree.root)
        tree.root.expand_all()  # 默认展开全部（可折叠收起）
        selected = picked["target"] or picked["first_leaf"] or tree.root
        # ``select_node`` 依赖**已算出的行号**（首帧前为 None → 静默无效，实测）：延后一拍。
        self._plan_default_node = selected
        self.call_after_refresh(self._apply_plan_default_selection)
        node = selected.data if selected is not tree.root else None
        self._set_plan_detail_md(self._plan_node_markdown(pid, node))

    def _apply_plan_default_selection(self) -> None:
        """首帧后把树光标落到默认节点（当前进行中环节；无则该叶/根）。"""
        node = self._plan_default_node
        if node is None or self.view != "plan-detail":
            return
        try:
            tree = self.query_one("#plan-detail-tree", MarkedTree)
            tree.select_node(node)
        except Exception:  # noqa: BLE001 - 组合面容忍缺件
            return

    def _set_plan_detail_md(self, text: str) -> None:
        self.plan_detail_md = text
        try:
            self.query_one("#plan-detail-md", Markdown).update(text)
        except Exception:  # noqa: BLE001 - 组合面容忍缺件
            return

    def _plan_node_markdown(self, pid: str, node: dict | None) -> str:
        """节点详情 Markdown（**有啥写啥**）：id/路径、类型、状态、成员、brief 全文、
        运行留痕（任务书/会话/job/evidence/复核）；superstep 附**子节点概览**。"""
        from .. import plans as _plans

        if node is None:
            return f"# 计划 {pid}\n\n（未选中环节）\n"
        out = [f"# 环节 {node.get('id')}", ""]
        kind = "superstep（分组）" if node.get("kind") == "super" else "环节（叶子）"
        out.append(f"- 类型：{kind}")
        out.append(f"- 路径：`{node.get('path')}`")
        if node.get("mode"):
            out.append(f"- 模式：{node.get('mode')}")
        out.append(f"- 状态：{node.get('status') or 'pending'}")
        out.append(f"- 成员：{node.get('member') or '（组长自办/未指派）'}")
        out += ["", "## 粗描述", "", str(node.get("brief") or "") or "（无）"]
        if node.get("kind") == "leaf":
            try:
                steps = {
                    str(s.get("id")): s
                    for s in (_plans.load_state(pid).get("steps") or [])
                }
            except Exception:  # noqa: BLE001 - 读数坏不塌面板
                steps = {}
            st = steps.get(str(node.get("path"))) or {}
            rows = [
                f"- {label}：`{st[key]}`"
                for label, key in (
                    ("任务书", "task_file"),
                    ("会话", "session_id"),
                    ("job", "job_id"),
                    ("evidence", "evidence"),
                    ("复核", "review"),
                )
                if st.get(key)
            ]
            if rows:
                out += ["", "## 运行留痕", "", *rows]
        if node.get("children"):
            out += ["", "## 子节点概览", ""]
            out += [
                f"- {self._plan_node_label(ch)}（{ch.get('path')}）"
                for ch in node["children"]
            ]
        return "\n".join(out).rstrip() + "\n"

    def on_tree_node_highlighted(self, event: MarkedTree.NodeHighlighted) -> None:
        """树节点选中联动：把该节点详情渲染进 Markdown 面板（s6 核心交互）。"""
        if self.view != "plan-detail":
            return
        data = getattr(event.node, "data", None)
        if isinstance(data, dict):
            self._set_plan_detail_md(
                self._plan_node_markdown(self.plan_detail_id, data)
            )


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
