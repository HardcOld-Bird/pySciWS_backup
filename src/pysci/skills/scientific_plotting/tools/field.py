"""数据驱动场面板：从导出场（VTK/CSV）以指定 cmap+norm 渲染，使共享 colorbar 与面板颜色严格一致。

背景：把 COMSOL 渲染的 PNG 当栅格面板（raster.py）时，其颜色由 COMSOL 色标（如 'Wave'）
决定，与 matplotlib 共享 colorbar 的 cmap（如 'bwr'）只是**近似**匹配，标度读数与面板颜色
存在细微失配。本模块改为**数据驱动**：直接读场数据（点坐标 + 标量），用 matplotlib
``tripcolor`` 以显式 cmap+norm 渲染，使面板颜色与共享 colorbar（同 cmap+norm）**严格一致**。

领域无关：任何需要 truthful colorbar 的场图（声学/电磁/力学）都适用。

- :func:`load_field_points` — 读 VTK/VTU（pyvista）或 CSV/TXT → ``(xy Nx2, values N)``。
- :func:`add_field_panel` — 在给定 ax 上 ``tripcolor`` 渲染（自动 Delaunay 或指定 triangles）。
- :func:`render_field_panel` — 单面板 + colorbar + savefig 一步到位（colorbar 直接取面板 mappable）。
- :func:`independent_norms` — 每组独立 vmin/vmax 的线性 norm（多组对照禁统一量程/默认对数）。

多面板组装见 raster.compose_grid 的 ``kind: field`` 面板（与共享 colorbar 同 cmap/norm）。
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

#: 支持的场源后缀 → 读取器类别
_VTK_SUFFIX = (".vtk", ".vtu")
_CSV_SUFFIX = (".csv", ".txt")


def _read_vtk_xyv(path: Path, scalars: str | None) -> tuple[np.ndarray, np.ndarray]:
    """用 pyvista 读 VTK/VTU → (xy Nx2, values N)；scalars 缺省取首个 point_data 数组。"""
    import pyvista as pv  # noqa: PLC0415 - 仅 VTK 场需要

    mesh = pv.read(str(path))
    pts = np.asarray(mesh.points, dtype=float)
    if pts.shape[1] < 2:
        raise ValueError(f"VTK 点维度不足：{path}")
    if scalars is None:
        keys = list(mesh.point_data.keys())
        if not keys:
            raise ValueError(f"VTK 无 point_data 标量，请显式指定 scalars：{path}")
        scalars = keys[0]
    vals = np.asarray(mesh[scalars], dtype=float).ravel()
    return pts[:, :2], vals


def _read_csv_xyv(
    path: Path, cols: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """读 COMSOL/通用 CSV/TXT → (xy Nx2, values N)。

    稳健处理 COMSOL 导出格式：跳过 ``%`` 注释行、跳过非数值的列名行，数据行按逗号或
    空白分割。``cols=(xi, yi, vi)`` 指定 x/y/value 的列索引（默认前三列）。
    """
    rows: list[list[float]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        parts = [p.strip() for p in line.replace(",", " ").split()]
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue  # 列名行（非纯数值），跳过
    if not rows:
        raise ValueError(f"CSV 未解析到数值行：{path}")
    arr = np.asarray(rows, dtype=float)
    xi, yi, vi = cols
    if arr.shape[1] <= max(xi, yi, vi):
        raise ValueError(
            f"CSV 列数不足（需列索引 {cols}，实有 {arr.shape[1]} 列）：{path}"
        )
    return arr[:, [xi, yi]], arr[:, vi]


def load_field_points(
    source: str | Path,
    *,
    scalars: str | None = None,
    cols: tuple[int, int, int] = (0, 1, 2),
) -> tuple[np.ndarray, np.ndarray]:
    """读场源 → ``(xy Nx2, values N)``。

    Args:
        source: .vtk/.vtu（pyvista）或 .csv/.txt。
        scalars: VTK 标量数组名；None → 取首个 point_data 数组。
        cols: CSV 的 (x, y, value) 列索引；默认前三列。
    """
    p = Path(source)
    suf = p.suffix.lower()
    if suf in _VTK_SUFFIX:
        return _read_vtk_xyv(p, scalars)
    if suf in _CSV_SUFFIX:
        return _read_csv_xyv(p, tuple(cols))
    raise ValueError(f"不支持的场源格式：{suf}（支持 {_VTK_SUFFIX + _CSV_SUFFIX}）")


def add_field_panel(
    ax: Any,
    xy: np.ndarray,
    values: np.ndarray,
    *,
    cmap: str = "bwr",
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Any | None = None,
    triangles: Any | None = None,
    shading: str = "gouraud",
    keep_triangle: Any | None = None,
) -> Any:
    """在 ax 上以 ``tripcolor`` 渲染散点场，返回 mappable（供 colorbar 复用，保证严格一致）。

    Args:
        xy: Nx2 点坐标。
        values: N 标量。
        cmap: matplotlib colormap 名。
        vmin/vmax: 颜色范围；与共享 colorbar 用同一范围即严格一致。
        norm: 显式 matplotlib Normalize（优先于 vmin/vmax）。
        triangles: 三角连接（Mx3 索引）；None → 对 (x,y) 自动 Delaunay。
        shading: "gouraud"（平滑）| "flat"（分面）。
        keep_triangle: 可选 callable(centroids Mx2) -> bool M（True=保留）。自动 Delaunay 会在
            物理域外（如固体壁/管槽间隙）架桥产生伪影三角形；用物理域判据遮罩其质心即可去除。
    """
    from matplotlib.colors import Normalize  # noqa: PLC0415
    from matplotlib.tri import Triangulation  # noqa: PLC0415

    xy = np.asarray(xy, dtype=float)
    vals = np.asarray(values, dtype=float).ravel()
    triang = Triangulation(
        xy[:, 0], xy[:, 1], None if triangles is None else np.asarray(triangles)
    )
    if keep_triangle is not None:
        cent = xy[triang.triangles].mean(axis=1)
        keep = np.asarray(keep_triangle(cent), dtype=bool)
        triang.set_mask(~keep)
    return ax.tripcolor(
        triang, vals, cmap=cmap, norm=norm or Normalize(vmin, vmax), shading=shading
    )


def independent_norms(
    values_by_group: Mapping[str, Any],
    *,
    vmax: Mapping[str, float] | None = None,
    vmin: Mapping[str, float] | None = None,
    symmetric: bool = False,
) -> dict[str, Any]:
    """每组**独立**的线性 ``Normalize``（多组对照禁统一量程、禁默认对数）。

    Args:
        values_by_group: ``{组名: 值数组}``；缺省量程按该组自身 min/max 取。
        vmax/vmin: 逐组显式量程映射（键须属于 ``values_by_group``）。传标量会
            raise——统一量程会掩盖弱组，须逐组给值。
        symmetric: True 时缺省量程取对称 ``[-max|v|, +max|v|]``。

    Returns:
        ``{组名: matplotlib.colors.Normalize}``。需要对数/对称对数标度时由调用方
        显式自建 norm——本函数不提供非线性入口（禁默认对数）。

    Raises:
        ValueError: 标量量程、未知组名、空数据或量程不可分。
    """
    from matplotlib.colors import Normalize  # noqa: PLC0415

    for label, spec in (("vmax", vmax), ("vmin", vmin)):
        if spec is not None and not isinstance(spec, Mapping):
            raise ValueError(
                f"{label} 须为逐组映射（禁统一量程）：{type(spec).__name__}"
            )
        unknown = set(spec or {}) - set(values_by_group)
        if unknown:
            raise ValueError(f"{label} 含未知组：{sorted(unknown)}")
    out: dict[str, Any] = {}
    for key, raw in values_by_group.items():
        vals = np.asarray(raw, dtype=float).ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            raise ValueError(f"组 {key} 无有限数据")
        hi = (
            vmax[key]
            if vmax and key in vmax
            else float(np.max(np.abs(vals)) if symmetric else np.max(vals))
        )
        lo = (
            vmin[key]
            if vmin and key in vmin
            else float(-hi if symmetric else np.min(vals))
        )
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo >= hi:
            raise ValueError(f"组 {key} 量程不可分：vmin={lo}, vmax={hi}")
        out[key] = Normalize(vmin=lo, vmax=hi)
    return out


def render_field_panel(
    source: str | Path,
    out: str | Path,
    *,
    scalars: str | None = None,
    cols: tuple[int, int, int] = (0, 1, 2),
    cmap: str = "bwr",
    vmin: float | None = None,
    vmax: float | None = None,
    triangles: Any | None = None,
    shading: str = "gouraud",
    figsize: tuple[float, float] = (6, 5),
    dpi: int = 150,
    colorbar: bool = True,
    cbar_label: str | None = None,
    title: str | None = None,
    axis_off: bool = False,
) -> Path:
    """单场面板渲染 + colorbar + savefig 一步到位（colorbar 直接取面板 mappable → 严格一致）。"""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    xy, vals = load_field_points(source, scalars=scalars, cols=cols)
    fig, ax = plt.subplots(figsize=figsize)
    mappable = add_field_panel(
        ax,
        xy,
        vals,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        triangles=triangles,
        shading=shading,
    )
    ax.set_aspect("equal")
    if axis_off:
        ax.set_axis_off()
    if title:
        ax.set_title(title)
    if colorbar:
        cb = fig.colorbar(mappable, ax=ax)
        if cbar_label:
            cb.set_label(cbar_label)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out
