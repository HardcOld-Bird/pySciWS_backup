"""多子图布局与面板标号工具。

科研插图常是"一个包含许多子图的大矩形"，复杂时还有子图嵌套。matplotlib 原生
GridSpec / subfigure 已足够表达，本模块只做两件高频且易错的事：

1. **面板标号**：期刊要求每个子图带 ``(a) (b) (c)`` 角标，位置/字体需统一——
   ``label_panels`` 一次性给一批 axes 打号，避免每张图手写 ``ax.text(...)``。
2. **布局速记**：``grid`` / ``nested`` 是对 ``plt.subplots`` / ``GridSpec`` 的薄封装，
   统一 ``constrained_layout``、共享轴等常用开关，减少样板代码。

这些工具与具体期刊风格解耦：字号/字体从当前 rcParams（由 style 层设定）继承。
"""

from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec


def grid(
    nrows: int,
    ncols: int,
    *,
    fig: Figure | None = None,
    width_ratios: Sequence[float] | None = None,
    height_ratios: Sequence[float] | None = None,
    sharex: bool | str = False,
    sharey: bool | str = False,
    constrained: bool = True,
    **subplot_kw,
) -> tuple[Figure, np.ndarray | Axes]:
    """创建规整子图网格（``plt.subplots`` 的薄封装）。

    Args:
        nrows, ncols: 网格行列数。
        fig: 复用已有 Figure（None 则按当前 rcParams 的 figsize 新建）。
        width_ratios / height_ratios: 列宽 / 行高比例。
        sharex / sharey: 共享轴（True/False/'row'/'col'）。
        constrained: 是否启用 constrained_layout（发表图推荐 True）。

    Returns:
        (fig, axes)。axes 形状与 matplotlib 约定一致（1x1 时为单个 Axes）。
    """
    gridspec_kw: dict = {}
    if width_ratios is not None:
        gridspec_kw["width_ratios"] = list(width_ratios)
    if height_ratios is not None:
        gridspec_kw["height_ratios"] = list(height_ratios)
    if fig is None:
        fig = plt.figure(constrained_layout=constrained)
    else:
        fig.set_constrained_layout(constrained)
    axes = fig.subplots(
        nrows,
        ncols,
        sharex=sharex,
        sharey=sharey,
        gridspec_kw=gridspec_kw or None,
        **subplot_kw,
    )
    return fig, axes


def nested(
    fig: Figure,
    outer: tuple[int, int],
    inner: Sequence[tuple[int, int]],
    *,
    outer_ratios: Sequence[float] | None = None,
    constrained: bool = True,
) -> list[np.ndarray | Axes]:
    """嵌套布局：外层网格的每个格子里再放一个内层网格。

    适合"左：概念示意；右：2x2 数据子图"这类复合图。

    Args:
        fig: 目标 Figure。
        outer: 外层 (nrows, ncols)。
        inner: 与外层格子一一对应的内层 (nrows, ncols) 序列。
        outer_ratios: 外层列宽比例。
        constrained: 是否启用 constrained_layout。

    Returns:
        与 ``inner`` 等长的 axes 列表（每项可能是单 Axes 或 ndarray）。
    """
    fig.set_constrained_layout(constrained)
    gs_kw = {"width_ratios": list(outer_ratios)} if outer_ratios else None
    outer_gs = GridSpec(outer[0], outer[1], figure=fig, **(gs_kw or {}))
    result: list[np.ndarray | Axes] = []
    for idx, (ir, ic) in enumerate(inner):
        cell = outer_gs[idx]
        sub = cell.subgridspec(ir, ic)
        result.append(fig.subplots(ir, ic, gridspec=sub))
    return result


def flatten_axes(axes) -> list[Axes]:
    """把 (可能嵌套的) axes 结构拍平成一维列表，按行优先顺序。"""
    if isinstance(axes, Axes):
        return [axes]
    arr = np.asarray(axes, dtype=object)
    flat: list[Axes] = []
    for a in arr.ravel():
        if isinstance(a, Axes):
            flat.append(a)
        elif isinstance(a, Iterable):
            flat.extend(flatten_axes(a))
    return flat


def label_panels(
    axes,
    *,
    fmt: str = "({letter})",
    start: int = 0,
    x: float = -0.02,
    y: float = 1.02,
    fontsize: float | None = None,
    fontweight: str = "bold",
    ha: str = "right",
    va: str = "bottom",
    letters: str = "abcdefghijklmnopqrstuvwxyz",
) -> list[Axes]:
    """给一批子图批量打 ``(a) (b) (c)`` 角标。

    角标以 axes 分数坐标放置（默认左上角外侧），字号默认继承 rcParams 的
    ``font.size``（可显式覆盖）。返回被打号的 axes 列表（拍平后顺序）。

    Args:
        axes: 单个 Axes 或 (嵌套) 数组。
        fmt: 标号格式，``{letter}`` 会被替换为字母。
        start: 起始字母索引（0 -> 'a'）。
        x, y: axes 分数坐标下的标号位置。
        fontsize: 覆盖字号（None 则用当前 rcParams font.size）。
        fontweight: 字重（期刊角标常加粗）。
    """
    flat = flatten_axes(axes)
    for i, ax in enumerate(flat):
        letter = letters[(start + i) % len(letters)]
        ax.text(
            x,
            y,
            fmt.format(letter=letter, index=start + i),
            transform=ax.transAxes,
            fontsize=fontsize,
            fontweight=fontweight,
            ha=ha,
            va=va,
        )
    return flat
