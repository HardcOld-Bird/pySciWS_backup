"""发散安全原语：奇点感知 1D 采样、发散遮断与发散安全探索绘图。

ADR（自研理由，2026-11-17）：EP/BIC/CPA 等奇点邻域发散的物理量在数值探索中需要
"不触奇点 + 窗外断线"的安全语义——nan 遮断优于 clip 饱和（饱和段会画成贴轴假平台，
被误读为物理平台）。社区无可替代薄件（matplotlib 只管画、不管采样/遮断语义），故保留
最小三原语；行为合同以旧项目 ``tests/skills/theoretical_computation/
test_divergent_primitives.py`` 的 7 项期望为准绳（新实现、仅对照期望值，未迁移旧代码）。
积分/级数发散检测、收敛半径/围道检查等更高层模式见
``.agents/skills/theory-calculation/references/divergence-safety.md``（文档配方，不入代码）。

机械断言：``tests/skills/theory_calculation/test_divergence_safety.py``。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def adaptive_sample_1d(
    interval: tuple[float, float],
    *,
    singularities: Sequence[float] = (),
    base_resolution: int = 200,
    approach_points: int = 8,
) -> NDArray[np.floating]:
    """奇点感知 1D 采样网格：均匀基网格 + 奇点两侧几何趋近点。

    行为合同（发散安全 7 项验收期望的采样侧）：

    - 无奇点（或奇点全部在区间外）时退化为 ``np.linspace(lo, hi, base_resolution)``；
    - **不触奇点**：奇点本身绝不入网格（基网格撞上奇点也剔除）；
    - 奇点邻域确实加密：每侧 ``approach_points`` 个按几何级数趋近的点；
    - 输出严格升序且去重；
    - 退化/反向区间抛 :class:`ValueError`。

    Args:
        interval: ``(lo, hi)``，要求 ``hi > lo``。
        singularities: 奇点位置；区间外的忽略。
        base_resolution: 均匀基网格点数（>=2）。
        approach_points: 每个奇点单侧的趋近点数（>=1）。

    Returns:
        一维严格升序采样点数组（float64）。
    """
    lo, hi = float(interval[0]), float(interval[1])
    if not hi > lo:
        raise ValueError(f"区间退化或反向：({lo}, {hi})，要求 hi > lo")
    if base_resolution < 2:
        raise ValueError(f"base_resolution 至少为 2：{base_resolution}")
    if approach_points < 1:
        raise ValueError(f"approach_points 至少为 1：{approach_points}")
    span = hi - lo
    parts = [np.linspace(lo, hi, base_resolution)]
    interior: list[float] = []
    extra: list[float] = []
    for s in singularities:
        s = float(s)
        if not lo < s < hi:
            continue  # 区间外奇点忽略（合同）
        interior.append(s)
        for d in span * np.geomspace(1e-6, 0.25, approach_points):
            for p in (s - d, s + d):
                if lo <= p <= hi:
                    extra.append(p)
    if extra:
        parts.append(np.asarray(extra))
    x = np.unique(np.concatenate(parts))  # 升序 + 去重
    for s in interior:
        x = x[x != s]  # 基网格撞上奇点时剔除：不触奇点
    return x


def mask_divergent(y: ArrayLike, window: tuple[float, float]) -> NDArray[np.floating]:
    """发散遮断：窗外与非有限值置 NaN（绘图即断线），复数先取实部。

    Args:
        y: 值数组（实数或复数；复数取实部后再判窗）。
        window: ``(lo, hi)`` 保留闭窗；窗外置 NaN。反向窗抛 :class:`ValueError`。

    Returns:
        float 数组（副本）：窗内保留原值；窗外、NaN、±Inf 一律 NaN。
    """
    arr = np.asarray(y)
    if np.iscomplexobj(arr):
        arr = arr.real
    out = np.array(arr, dtype=float, copy=True)
    lo, hi = float(window[0]), float(window[1])
    if hi < lo:
        raise ValueError(f"window 反向：({lo}, {hi})，要求 hi >= lo")
    bad = ~np.isfinite(out) | (out < lo) | (out > hi)
    out[bad] = np.nan
    return out


def quick_plot_divergent(
    x: ArrayLike,
    y: ArrayLike,
    *,
    ylim: tuple[float, float] | None = None,
    title: str | None = None,
    ax: object | None = None,
) -> Figure:
    """发散安全的一维探索绘图：先 :func:`mask_divergent` 遮断再绘制，返回 Figure。

    适用于奇点邻域发散的物理量（采样建议配 :func:`adaptive_sample_1d`）。
    ``ylim`` 给定时作为遮断窗并锁定坐标轴范围；省略时按有限值范围自动取窗
    （效果=只遮断 NaN/Inf）。matplotlib 只在本函数内 import（最小路径不背绘图后端）。
    探索图到此为止——出版图走③科研绘图域。

    Args:
        x: 横轴采样点。
        y: 纵轴值（可含发散/Inf/NaN/复数）。
        ylim: 遮断窗兼坐标轴范围；None 自动取窗。
        title: 图标题；None 不设。
        ax: 可选的 matplotlib Axes；None 时新建单图 Figure。

    Returns:
        ``matplotlib.figure.Figure``。
    """
    import matplotlib.pyplot as plt

    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y)
    if xa.shape != ya.shape:
        raise ValueError(f"x 与 y 形状不一致：{xa.shape} vs {ya.shape}")
    window = ylim if ylim is not None else _finite_window(ya)
    masked = mask_divergent(ya, window)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.plot(xa, masked, linewidth=1.8)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if title is not None:
        ax.set_title(title)
    return fig


def _finite_window(y: ArrayLike) -> tuple[float, float]:
    """取有限值范围作自动遮断窗（无有限值时抛 :class:`ValueError`）。"""
    vals = np.asarray(y)
    if np.iscomplexobj(vals):
        vals = vals.real
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        raise ValueError("y 中没有有限值，无法自动取窗")
    return float(finite.min()), float(finite.max())
