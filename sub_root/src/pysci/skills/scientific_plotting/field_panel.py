"""数据驱动场面板 truthful colorbar（用户裁决 2026-11-17：保留，薄化）。

真值语义：面板着色与共享 colorbar 用**同一个 Normalize**（cmap+norm 严格一致），
vmax/端标按 **v2 阶梯向上取整**（:func:`ceil_label`，绝不裁峰）——truthful by
construction，不做两套归一化。

端标取整 v2 规则（SKILL.md 规则文字，机械实现 :func:`ceil_label`）：
合法值 ``d1.d2 × 10^k``——``d1`` 取 1–9 任意，``d2 ∈ {0,5}``，``d1=1`` 时另含
``d2 ∈ {2,4,6,8}``；第三位起全为 0。返回 ``>= x`` 的最小合法值
（65.48→70、1.4655→1.5、2.11→2.5、0.6436→0.65）。**径向刻度 rmax 不适用**。

数据通路：CSV（x,y,v 三列）或 VTK/VTU（pyvista 懒导入，仅场数据需要）→
matplotlib tripcolor。ADR（2026-11-17）：旧 field.py 23KB 薄化为本模块；
COMSOL 栅格 sidecar 叠加**不建**（归②仿真域，③只文档化 imshow 用法）。
"""

from __future__ import annotations

import math
from decimal import Decimal
from pathlib import Path

# v2 阶梯尾数表（升序）：d1.d2，d2∈{0,5}，d1=1 另含 2/4/6/8。
_LABEL_MANTISSAS: tuple[Decimal, ...] = tuple(
    sorted(
        {Decimal(f"{d1}.{d2}") for d1 in range(1, 10) for d2 in ("0", "5")}
        | {Decimal(f"1.{d2}") for d2 in ("2", "4", "6", "8")}
    )
)


def ceil_label(x: float) -> float:
    """colorbar 端标 v2 阶梯向上取整（场图 vmax/端标默认口径）。

    Args:
        x: 正有限数（端标量值；负端标由调用方对称处理 ``-ceil_label(abs(x))``）。

    Returns:
        ``>= x`` 的最小 v2 合法值（已合规原样返回）：
        ``65.48→70``、``1.4655→1.5``、``1.8277→2.0``、``2.11→2.5``、
        ``1.1968→1.2``、``0.6436→0.65``、``0.98→1.0``、``9.7→10``、``0.151→0.16``。

    Raises:
        ValueError: 非正或非有限输入。
    """
    if not math.isfinite(x) or x <= 0:
        raise ValueError(f"ceil_label 需要正有限数：{x!r}")
    d = Decimal(repr(float(x)))
    e = d.adjusted()
    while True:
        for mant in _LABEL_MANTISSAS:
            cand = mant.scaleb(e)
            if cand >= d:
                return float(cand)
        e += 1


def read_csv_xyv(path: str | Path) -> tuple[list[float], list[float], list[float]]:
    """读 CSV 三列 ``(x, y, v)`` 场数据（纯 Python，无 numpy 依赖）。

    Args:
        path: CSV 路径（可带表头；非数字行跳过）。

    Returns:
        ``(xs, ys, values)`` 三个等长列表。

    Raises:
        ValueError: 无有效数据行或列数不足。
    """
    xs: list[float] = []
    ys: list[float] = []
    vals: list[float] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = [p.strip() for p in line.replace(",", " ").split() if p.strip()]
        if len(parts) < 3:
            continue
        try:
            x, y, v = (float(parts[i]) for i in range(3))
        except ValueError:
            continue  # 表头/注释行
        xs.append(x)
        ys.append(y)
        vals.append(v)
    if not vals:
        raise ValueError(f"CSV 无有效 (x, y, v) 数据行：{path}")
    return xs, ys, vals


def read_vtk_xyv(
    path: str | Path, scalars: str | None = None
) -> tuple[list[float], list[float], list[float]]:
    """用 pyvista 读 VTK/VTU 点位场 → ``(xs, ys, values)``（pyvista 懒导入）。

    Args:
        path: VTK/VTU 文件。
        scalars: 点数据标量名；None → 取首个 point_data 数组。

    Returns:
        三个等长列表（二维点；z 忽略）。

    Raises:
        ImportError: pyvista 未安装（按需件）。
        ValueError: 点维度不足、无 point_data 或标量名不存在。
    """
    try:
        import numpy as np
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - 依赖位
        raise ImportError("读 VTK 需要 pyvista（按需加装：uv add pyvista）") from exc

    mesh = pv.read(str(path))
    pts = np.asarray(mesh.points, dtype=float)
    if pts.shape[1] < 2:
        raise ValueError(f"VTK 点维度不足：{path}")
    if scalars is None:
        keys = list(mesh.point_data.keys())
        if not keys:
            raise ValueError(f"VTK 无 point_data 标量，请显式指定 scalars：{path}")
        scalars = keys[0]
    if scalars not in mesh.point_data:
        raise ValueError(f"VTK 标量不存在：{scalars!r}（可用：{list(mesh.point_data.keys())}）")
    values = np.asarray(mesh.point_data[scalars], dtype=float)
    return pts[:, 0].tolist(), pts[:, 1].tolist(), values.tolist()


def plot_field(
    ax: object,
    xs: list[float],
    ys: list[float],
    values: list[float],
    *,
    triangles: object | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    ceil_vmax: bool = True,
    **tripcolor_kwargs: object,
) -> object:
    """画数据驱动场面板（tripcolor，truthful：mappable 自带共享 norm）。

    Args:
        ax: Axes（``tripcolor``）。
        xs / ys / values: 等长场数据。
        triangles: 三角化（None → matplotlib 自算 Delaunay）。
        cmap: 色图。
        vmin / vmax: 归一化端点（None → 数据极值）。
        ceil_vmax: vmax 按 :func:`ceil_label` 端标取整（默认开；绝不裁峰）。
        **tripcolor_kwargs: 透传 ``tripcolor``。

    Returns:
        mappable（给 :func:`add_truthful_colorbar`，norm 严格同源）。
    """
    import matplotlib.colors as mcolors

    vmin = min(values) if vmin is None else vmin
    vmax = max(values) if vmax is None else vmax
    if ceil_vmax:
        vmax = ceil_label(vmax)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    # matplotlib ≥3.8：triangles 为 keyword-only（None → 自算 Delaunay）
    return ax.tripcolor(
        xs, ys, values, triangles=triangles, cmap=cmap, norm=norm, **tripcolor_kwargs
    )


def add_truthful_colorbar(
    fig: object,
    mappable: object,
    *,
    ax: object | None = None,
    **colorbar_kwargs: object,
) -> object:
    """给场面板加**真值** colorbar（mappable 的 norm/cmap 同源，无需二次传参）。

    Args:
        fig: Figure（``colorbar``）。
        mappable: :func:`plot_field` 返回的 mappable。
        ax: colorbar 所属面板轴（None → matplotlib 自动）。
        **colorbar_kwargs: 透传 ``fig.colorbar``。

    Returns:
        Colorbar 实例。
    """
    return fig.colorbar(mappable, ax=ax, **colorbar_kwargs)
