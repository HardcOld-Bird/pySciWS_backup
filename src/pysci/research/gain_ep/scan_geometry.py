"""gain_ep 扫描区几何单一真源（既定几何读数）。

用户第 1 轮批语 5 定：目标扫描区为 **31 cm 正方形**（菱形显示），**近边距超表面中心
30 cm**，沿 **±45° 径向**通道放置，**边平行径向/切向轴**。既定实现见
``data/research/1_gain_ep/article/figures/beam_scatter/beam_scatter.py``
（``SCAN_SIDE`` / ``SCAN_NEAR_EDGE`` / ``CH_ANG``）。

本模块把该口径登记为常量：field_schema 导出与绘图侧一律引用这里，不再各处手抄数字
（任务书引用这些读数时须附「源：本模块」或 schema 键）。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

#: 目标扫描区边长（m）。
SCAN_SIDE_M = 0.31
#: 扫描区近边到超表面中心的距离（m）。
SCAN_NEAR_EDGE_M = 0.30
#: 两组扫描区所在的径向通道（deg，自 +x 逆时针）。
SCAN_RADIAL_DEG = (135.0, 45.0)
#: 扫描正方形边与径向/切向轴平行（不是 45° 斜接）。
SCAN_EDGES_PARALLEL_RADIAL = True
#: 超表面中心（m）：X0 = 4D，D = lambda/sqrt(2)，lambda = 0.1 m。
METASURFACE_CENTER_M = (0.282842712474619, 0.0)


def radial_unit(radial_deg: float) -> np.ndarray:
    """径向单位向量（逆时针自 +x）。"""
    a = math.radians(float(radial_deg))
    return np.array([math.cos(a), math.sin(a)], dtype=float)


def scan_square_vertices(
    radial_deg: float,
    *,
    side_m: float = SCAN_SIDE_M,
    near_edge_m: float = SCAN_NEAR_EDGE_M,
    center: tuple[float, float] = METASURFACE_CENTER_M,
) -> np.ndarray:
    """扫描正方形 4 顶点（m）：近边垂直于径向、距中心 ``near_edge_m``，边平行径向/切向。

    返回顺序为「远-左、远-右、近-右、近-左」之外的无序集合语义；需要多边形顺序时用
    :func:`ordered_vertices` 按绕中心角排序。
    """
    if side_m <= 0 or near_edge_m < 0:
        raise ValueError(f"side_m 须 >0、near_edge_m 须 >=0：{side_m}, {near_edge_m}")
    u = radial_unit(radial_deg)
    v = np.array([-u[1], u[0]])
    c = np.asarray(center, dtype=float) + (near_edge_m + side_m / 2.0) * u
    half = side_m / 2.0
    return np.array(
        [
            c + su * half * u + sv * half * v
            for su, sv in ((1, 1), (1, -1), (-1, -1), (-1, 1))
        ]
    )


def ordered_vertices(vertices: np.ndarray) -> np.ndarray:
    """按绕质心角度排序（多边形顶点序），供边/面积计算。"""
    pts = np.asarray(vertices, dtype=float)
    if pts.shape != (4, 2):
        raise ValueError(f"需要 4×2 顶点：{pts.shape}")
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(ang)]


def scan_geometry_block(radial_deg: float) -> dict[str, Any]:
    """field_schema 导出用的扫描区几何块（键名见 ``field_schema`` 契约）。"""
    vertices = ordered_vertices(scan_square_vertices(radial_deg))
    return {
        "radial_deg": float(radial_deg),
        "side_m": SCAN_SIDE_M,
        "near_edge_m": SCAN_NEAR_EDGE_M,
        "edges_parallel_radial": SCAN_EDGES_PARALLEL_RADIAL,
        "center_m": [round(float(v), 12) for v in vertices.mean(axis=0)],
        "vertices_m": [[round(float(x), 12), round(float(y), 12)] for x, y in vertices],
    }
