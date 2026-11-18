"""期刊样式契约 + 样式资产解析/应用（纯逻辑可测；matplotlib 懒导入）。

三类职责：

1. **尺寸契约**（数据，非代码）：APS 85/170mm、Nature 89/183mm（±2mm 为 audit 容差）；
2. **色盲安全循环**：Okabe-Ito 基表 + 家风循环（去黄，蓝打头）；
3. **样式资产** ``styles/*.mplstyle``：家风内容资产（SciencePlots 审美层基础上的
   自定义项固化为具体键值），叠加顺序后者覆盖前者，**契约层最后应用**。

ADR（2026-11-17）：样式资产随包分发，用 :mod:`importlib.resources` 定位包内数据
（不写 ``Path(__file__).parents[N]``）；解析 :func:`read_style_keys` 是纯文本解析
（mplstyle 即 ``key : value`` 行），机械断言无需 matplotlib。
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources
from pathlib import Path

from pysci.paths import PLOTTING_ROOT

#: 期刊尺寸契约（单位 mm；键 = 期刊族，值 = 单栏/双栏图宽）。
#: 来源：APS 85/170mm（PRL/PR 投稿口径）、Nature 89/183mm。
JOURNAL_WIDTHS: dict[str, dict[str, float]] = {
    "aps": {"single_mm": 85.0, "double_mm": 170.0},
    "nature": {"single_mm": 89.0, "double_mm": 183.0},
}

#: 跨度别名 → JOURNAL_WIDTHS 内键。
_SPAN_KEYS: dict[str, str] = {"single": "single_mm", "double": "double_mm"}

#: Okabe-Ito 色盲安全基表（8 色，色觉正常/色盲两可辨识；白底黑字出版面）。
OKABE_ITO: tuple[str, ...] = (
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
)

#: 家风线图循环：Okabe-Ito 去 ``#F0E442``（黄在白底线图对比度不足），蓝打头。
#: 与 ``styles/house.mplstyle`` 的 ``axes.prop_cycle`` 严格一致（audit 据此校验）。
HOUSE_CYCLE: tuple[str, ...] = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
)


def style_dir() -> Path:
    """返回样式资产目录 ``styles/``（包内数据，importlib.resources 定位）。

    Returns:
        样式资产目录的文件系统路径（editable/源码布局下为普通目录）。
    """
    return Path(str(resources.files(__package__).joinpath("styles")))


def list_styles() -> tuple[str, ...]:
    """列出可用样式资产名（不含 ``.mplstyle`` 后缀）。

    Returns:
        排序后的样式名元组，如 ``("house", "journal-aps", "journal-nature")``。
    """
    return tuple(sorted(p.stem for p in style_dir().glob("*.mplstyle")))


def style_path(name: str) -> Path:
    """解析样式资产文件路径；缺失时**报错列出可用名**（静默失效升级为可见失败）。

    Args:
        name: 样式名（不含后缀），如 ``"journal-aps"``。

    Returns:
        对应 ``styles/<name>.mplstyle`` 的路径。

    Raises:
        FileNotFoundError: 样式不存在（报错中列出可用样式）。
    """
    path = style_dir() / f"{name}.mplstyle"
    if not path.is_file():
        raise FileNotFoundError(
            f"样式资产不存在：{name!r}（可用：{', '.join(list_styles()) or '无'}）"
        )
    return path


def read_style_keys(name: str) -> dict[str, str]:
    """纯文本解析一个样式资产（mplstyle 即 ``key : value`` 行；``#`` 注释）。

    Args:
        name: 样式名（不含后缀）。

    Returns:
        ``{rcParam 名: 字符串值}``（顺序同文件；重复键后者覆盖前者）。
    """
    keys: dict[str, str] = {}
    for line in style_path(name).read_text(encoding="utf-8").splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped:
            continue
        key, sep, value = stripped.partition(":")
        if sep:
            keys[key.strip()] = value.strip()
    return keys


def resolve_width_mm(journal: str, span: str) -> float:
    """解析期刊尺寸契约中的图宽（mm）。

    Args:
        journal: 期刊族名（``"aps"`` / ``"nature"``）。
        span: 栏宽（``"single"`` / ``"double"``）。

    Returns:
        契约图宽（mm），如 aps+double → 170.0。

    Raises:
        ValueError: 期刊族或栏宽未知。
    """
    try:
        spans = JOURNAL_WIDTHS[journal.lower()]
        return spans[_SPAN_KEYS[span.lower()]]
    except KeyError as exc:
        raise ValueError(
            f"未知尺寸契约：{journal!r} / {span!r}"
            f"（可用期刊：{', '.join(JOURNAL_WIDTHS)}；跨度：{', '.join(_SPAN_KEYS)}）"
        ) from exc


def mm_to_inch(mm: float) -> float:
    """mm → inch（matplotlib figsize 单位）。

    Args:
        mm: 毫米。

    Returns:
        英寸（1 inch = 25.4 mm）。
    """
    return mm / 25.4


def ensure_plotting_env(cache_home: str | Path | None = None) -> Path:
    """绘图栈运行环境引导（**必须在 import matplotlib / ultraplot 之前**调用）。

    ADR（2026-11-17，实测）：两处用户目录在 DSH 沙盒受限令牌下写入直接 PermissionError，
    且自包含原则要求配置/缓存留在项目内，故一并重定向到
    :data:`~pysci.paths.PLOTTING_ROOT` 下的 ``cache/home``（幂等，可重复调用）：

    - ``HOME`` / ``USERPROFILE``：ultraplot 首次 import 写 ``~/.ultraplot``
      （Windows 固定此路，XDG_CONFIG_HOME 仅 Linux/FreeBSD 生效，见 ultraplot
      ``_config.Configurator.user_folder``）；
    - ``MPLCONFIGDIR``：matplotlib 配置/字体缓存默认 ``%LOCALAPPDATA%\\matplotlib``，
      写不了会降级临时目录并反复重建字体缓存（实测警告）。

    Args:
        cache_home: 重定向目标；None → ``PLOTTING_ROOT/cache/home``。

    Returns:
        实际重定向到的目录。
    """
    target = Path(cache_home) if cache_home is not None else PLOTTING_ROOT / "cache" / "home"
    target.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(target)
    os.environ["USERPROFILE"] = str(target)
    os.environ["MPLCONFIGDIR"] = str(target / "matplotlib")
    return target


def load_rc_params(*names: str) -> dict[str, object]:
    """按叠加顺序合并样式资产的 rcParams（后者覆盖前者；契约层放最后）。

    Args:
        *names: 样式名序列，如 ``("house", "journal-aps")``。

    Returns:
        合并后的 rcParams 字典（普通 dict，可直接喂 :class:`matplotlib.RcParams`
        或 :func:`matplotlib.rc_context`）。
    """
    import matplotlib  # 懒导入：纯逻辑测试无需绘图栈

    merged: dict[str, object] = {}
    for name in names:
        merged.update(
            matplotlib.rc_params_from_file(str(style_path(name)), use_default_template=False)
        )
    return merged


@contextmanager
def style_context(*names: str) -> Iterator[None]:
    """叠加样式资产的上下文管理器（契约层放最后覆盖审美层）。

    用法::

        with style_context("house", "journal-aps"):
            fig, ax = plt.subplots()

    Args:
        *names: 样式名序列（后者覆盖前者）。

    Yields:
        None；退出时恢复进入前的 rcParams。
    """
    import matplotlib

    with matplotlib.rc_context(load_rc_params(*names)):
        yield
