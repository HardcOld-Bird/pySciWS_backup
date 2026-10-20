"""gain_ep 扫描区几何单一真源（用户口头反馈 2026-10-20：菱形永远正方形）。

**用户口头反馈（2026-10-20，不发返工）**：半圆场域内外的菱形**永远应该是正方形**。
完整口径：

- 扫场区＝**边长 0.31 m 正方形**（径向＝横向＝0.31 m），边平行/垂直径向；
- **近边距超表面中心 0.25 m**（远边 0.25+0.31＝**0.56 m**≈用户「约 60cm」，指远边距离，
  不是径向长度）；
- 沿 ±45° 径向对称布置两区；
- 外框（放大视图）同样正方形：0.31×1.23≈0.3813 m。

来源与覆盖关系：用户口头反馈**覆盖** twin 裁决 A 的「60 cm 径向长度（25→85 cm 矩形）」
（该读法误把「约 60cm」当径向长度）与 R1 批语 5 的旧近边 30 cm。**菱形永远正方形**是
长期不变量（见 ``DIAMOND_IS_SQUARE``），任何矩形/菱形变形口径一律错误。

本模块是该口径的单一真源：field_schema 导出与绘图侧一律引用这里，不再各处手抄数字。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

#: 扫场区边长（m）：径向＝横向（菱形→正方形）。
SCAN_SIDE_M = 0.31
#: 径向尺寸（m）＝SCAN_SIDE_M。
SCAN_RADIAL_M = SCAN_SIDE_M
#: 横向尺寸（m）＝SCAN_SIDE_M。
SCAN_TRANSVERSE_M = SCAN_SIDE_M
#: 近边到超表面中心的距离（m）；远边 = 0.25 + 0.31 = 0.56 m。
SCAN_NEAR_EDGE_M = 0.25
#: 两组扫描区所在的径向通道（deg，自 +x 逆时针）。
SCAN_RADIAL_DEG = (135.0, 45.0)
#: 正方形边与径向/切向轴对齐（不是斜接）。
SCAN_ALIGNED_RADIAL = True
#: 形状不变量：**菱形永远正方形**（用户明文，长期有效）。
SCAN_SHAPE = "square"
DIAMOND_IS_SQUARE = True
#: 超表面中心（m）：X0 = 4D，D = lambda/sqrt(2)，lambda = 0.1 m。
METASURFACE_CENTER_M = (0.282842712474619, 0.0)

#: 旧名兼容（R6 矩形读法已废弃；同 SCAN_ALIGNED_RADIAL）。
SCAN_EDGES_PARALLEL_RADIAL = SCAN_ALIGNED_RADIAL


def radial_unit(radial_deg: float) -> np.ndarray:
    """径向单位向量（逆时针自 +x）。"""
    a = math.radians(float(radial_deg))
    return np.array([math.cos(a), math.sin(a)], dtype=float)


def scan_region_vertices(
    radial_deg: float,
    *,
    radial_m: float = SCAN_RADIAL_M,
    transverse_m: float = SCAN_TRANSVERSE_M,
    near_edge_m: float = SCAN_NEAR_EDGE_M,
    center: tuple[float, float] = METASURFACE_CENTER_M,
) -> np.ndarray:
    """正方形扫场区 4 顶点（m）。

    近边垂直于径向、距 ``center`` 为 ``near_edge_m``；边长 ``radial_m``（须等于
    ``transverse_m``——**菱形永远正方形**）。返回无序集合语义；需要多边形顺序用
    :func:`ordered_vertices`。
    """
    if not math.isclose(
        float(radial_m), float(transverse_m), rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            "菱形永远正方形：radial_m 必须等于 transverse_m"
            f"（收到 {radial_m} vs {transverse_m}）"
        )
    if radial_m <= 0 or near_edge_m < 0:
        raise ValueError(
            f"radial_m 须 >0、near_edge_m 须 >=0：{radial_m}, {near_edge_m}"
        )
    u = radial_unit(radial_deg)
    v = np.array([-u[1], u[0]])
    c = np.asarray(center, dtype=float) + (near_edge_m + radial_m / 2.0) * u
    half = radial_m / 2.0
    return np.array(
        [
            c + su * half * u + sv * half * v
            for su, sv in ((1, 1), (1, -1), (-1, -1), (-1, 1))
        ]
    )


def scan_square_vertices(
    radial_deg: float,
    *,
    side_m: float | None = None,
    near_edge_m: float = SCAN_NEAR_EDGE_M,
    center: tuple[float, float] = METASURFACE_CENTER_M,
) -> np.ndarray:
    """正方形顶点旧名：``side_m`` 缺省＝SCAN_SIDE_M；显式给值亦为边长。"""
    side = SCAN_SIDE_M if side_m is None else side_m
    return scan_region_vertices(
        radial_deg,
        radial_m=side,
        transverse_m=side,
        near_edge_m=near_edge_m,
        center=center,
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
    vertices = ordered_vertices(scan_region_vertices(radial_deg))
    return {
        "radial_deg": float(radial_deg),
        "shape": SCAN_SHAPE,
        "radial_m": SCAN_RADIAL_M,
        "transverse_m": SCAN_TRANSVERSE_M,
        "near_edge_m": SCAN_NEAR_EDGE_M,
        "edges_aligned_radial": SCAN_ALIGNED_RADIAL,
        "center_m": [round(float(v), 12) for v in vertices.mean(axis=0)],
        "vertices_m": [[round(float(x), 12), round(float(y), 12)] for x, y in vertices],
    }
