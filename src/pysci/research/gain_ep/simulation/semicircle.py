"""半圆显示区后处理：ROI 掩膜/裁剪 + 直径边以下遮盖（参数化中心/半径）。

sim2 的 1 m 旋转正方形域导出后，f2 只显示以超表面中心为圆心的上半圆感兴趣区。
本模块提供与绘图层无关的点云掩膜/裁剪，以及 Matplotlib 的 clip path 与
「直径边以下遮盖」支持——供 f2 直接消费（见 ``beam_scatter.draw_field_panel``
的旧内联实现，本轮改为调用本模块）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SemicircleROI:
    """上半圆感兴趣区：圆心 ``(cx, cy)``、半径 ``radius``（数据单位，通常为 m）。

    显示域 = ``y >= cy`` 且 ``(x-cx)^2 + (y-cy)^2 <= radius^2``；直径边为水平线
    ``y = cy``。
    """

    radius: float
    center: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if not math.isfinite(self.radius) or self.radius <= 0:
            raise ValueError(f"radius 须为正有限数：{self.radius!r}")
        if len(self.center) != 2 or not all(math.isfinite(v) for v in self.center):
            raise ValueError(f"center 须为 (cx, cy)：{self.center!r}")

    @property
    def cx(self) -> float:
        return float(self.center[0])

    @property
    def cy(self) -> float:
        return float(self.center[1])

    def mask(self, xy: np.ndarray) -> np.ndarray:
        """点云掩膜（True=在半圆内）；直径边上的点计入。"""
        pts = np.asarray(xy, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"xy 须为 Nx2：{pts.shape!r}")
        dx = pts[:, 0] - self.cx
        dy = pts[:, 1] - self.cy
        return (dy >= 0.0) & (dx * dx + dy * dy <= self.radius**2 + 1e-12)

    def outline(self, n: int = 361) -> np.ndarray:
        """半圆边界顶点（直径左端 → 弧线 → 右端），供 clip path 使用。"""
        if n < 3:
            raise ValueError(f"outline 顶点数须 >=3：{n}")
        th = np.linspace(0.0, math.pi, int(n))
        return np.column_stack(
            (self.cx + self.radius * np.cos(th), self.cy + self.radius * np.sin(th))
        )

    def bounds(self) -> tuple[float, float, float, float]:
        """数据范围 ``(x0, x1, y0, y1)``（y0=直径边）。"""
        return (
            self.cx - self.radius,
            self.cx + self.radius,
            self.cy,
            self.cy + self.radius,
        )


def clip_points(
    xy: np.ndarray, values: np.ndarray, roi: SemicircleROI
) -> tuple[np.ndarray, np.ndarray]:
    """按半圆掩膜裁剪点云与值；返回 ``(xy_roi, values_roi)``（保持原顺序）。"""
    pts = np.asarray(xy, dtype=float)
    vals = np.asarray(values, dtype=float).ravel()
    if pts.shape[0] != vals.size:
        raise ValueError(f"点数不一致：xy {pts.shape[0]} vs values {vals.size}")
    keep = roi.mask(pts)
    return pts[keep], vals[keep]


def clip_artist(artist: Any, ax: Any, roi: SemicircleROI, *, n: int = 361) -> Any:
    """把绘图 artist 的 clip path 设为半圆边界（轴外/下半区内容不渲染）。"""
    from matplotlib.patches import Polygon  # noqa: PLC0415 - 仅绘图需要

    poly = Polygon(roi.outline(n), closed=True, transform=ax.transData)
    artist.set_clip_path(poly)
    return artist


def add_below_diameter_mask(
    ax: Any,
    roi: SemicircleROI,
    *,
    color: str = "white",
    zorder: float = 20.0,
    x_pad: float = 0.0,
) -> Any | None:
    """在直径边以下铺不透明矩形，遮住管槽/域外内容。

    覆盖当前视图整宽、从视图下界到 ``y = cy``；调用前先把 xlim/ylim 固定到显示
    窗口（否则遮罩按自动范围落位）。视图中已无直径边以下区域时返回 ``None``。
    """
    from matplotlib.patches import Rectangle  # noqa: PLC0415 - 仅绘图需要

    x0, x1 = (float(v) for v in ax.get_xlim())
    y0, _ = (float(v) for v in ax.get_ylim())
    top = roi.cy
    if y0 >= top:
        return None
    left = min(x0, roi.cx - roi.radius) - x_pad
    right = max(x1, roi.cx + roi.radius) + x_pad
    bottom = min(y0, roi.cy - roi.radius)
    patch = Rectangle(
        (left, bottom),
        right - left,
        top - bottom,
        facecolor=color,
        edgecolor="none",
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def apply_semicircle(
    ax: Any,
    roi: SemicircleROI,
    artists: Any,
    *,
    mask_below: bool = True,
    mask_color: str = "white",
    mask_zorder: float = 20.0,
    n: int = 361,
) -> Any | None:
    """一步完成：逐个 artist 设半圆 clip path，并按需加直径边以下遮罩。"""
    if isinstance(artists, (list, tuple)):
        for art in artists:
            if art is not None:
                clip_artist(art, ax, roi, n=n)
    elif artists is not None:
        clip_artist(artists, ax, roi, n=n)
    if not mask_below:
        return None
    return add_below_diameter_mask(ax, roi, color=mask_color, zorder=mask_zorder)
