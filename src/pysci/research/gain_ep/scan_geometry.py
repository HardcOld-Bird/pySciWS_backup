"""gain_ep 扫描区几何单一真源（R6 twin 裁决口径）。

**R6 批语 2（用户实测）＋ twin 裁决 A**（案卷
``data/taskbooks/tb-20261020-080213-3-5.twin案卷-R6几何口径.md``）定：

- 扫场区＝**径向对齐矩形 0.60 m（径向）× 0.31 m（横向）**；
- **近边距超表面中心 0.25 m**（远边 0.85 m）；
- 沿 **±45° 径向**对称布置两区（左右不重叠）。

覆盖 R1 批语 5 的旧口径（31 cm 正方形、近边 30 cm）。本模块是该口径的单一真源：
field_schema 导出与绘图侧一律引用这里，不再各处手抄数字（任务书引用时须附「源：本模块」
或 schema 键）。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

#: 扫场区径向长度（m）：近边 0.25 → 远边 0.85。
SCAN_RADIAL_M = 0.60
#: 扫场区横向宽度（m）。
SCAN_TRANSVERSE_M = 0.31
#: 近边到超表面中心的距离（m）。
SCAN_NEAR_EDGE_M = 0.25
#: 两组扫描区所在的径向通道（deg，自 +x 逆时针）。
SCAN_RADIAL_DEG = (135.0, 45.0)
#: 矩形边与径向/切向轴对齐（不是斜接）。
SCAN_ALIGNED_RADIAL = True
#: 超表面中心（m）：X0 = 4D，D = lambda/sqrt(2)，lambda = 0.1 m。
METASURFACE_CENTER_M = (0.282842712474619, 0.0)

#: 旧名兼容（R1「31 cm 边长」读横向宽；新代码请用 SCAN_TRANSVERSE_M）。
SCAN_SIDE_M = SCAN_TRANSVERSE_M
#: 旧名兼容（同 SCAN_ALIGNED_RADIAL）。
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
    """径向对齐矩形扫场区 4 顶点（m）。

    近边垂直于径向、距 ``center`` 为 ``near_edge_m``；径向长 ``radial_m``（远边
    ``near_edge_m + radial_m``），横向宽 ``transverse_m``（±``transverse_m/2``）。
    返回无序集合语义；需要多边形顺序用 :func:`ordered_vertices`。
    """
    if radial_m <= 0 or transverse_m <= 0 or near_edge_m < 0:
        raise ValueError(
            f"radial_m/transverse_m 须 >0、near_edge_m 须 >=0：{radial_m}, "
            f"{transverse_m}, {near_edge_m}"
        )
    u = radial_unit(radial_deg)
    v = np.array([-u[1], u[0]])
    c = np.asarray(center, dtype=float) + (near_edge_m + radial_m / 2.0) * u
    half_r = radial_m / 2.0
    half_t = transverse_m / 2.0
    return np.array(
        [
            c + su * half_r * u + sv * half_t * v
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
    """旧名兼容转发：默认返回新矩形；显式给 ``side_m`` 时返回旧正方形。"""
    if side_m is None:
        return scan_region_vertices(radial_deg, near_edge_m=near_edge_m, center=center)
    return scan_region_vertices(
        radial_deg,
        radial_m=side_m,
        transverse_m=side_m,
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
        "radial_m": SCAN_RADIAL_M,
        "transverse_m": SCAN_TRANSVERSE_M,
        "near_edge_m": SCAN_NEAR_EDGE_M,
        "edges_aligned_radial": SCAN_ALIGNED_RADIAL,
        "center_m": [round(float(v), 12) for v in vertices.mean(axis=0)],
        "vertices_m": [[round(float(x), 12), round(float(y), 12)] for x, y in vertices],
    }
