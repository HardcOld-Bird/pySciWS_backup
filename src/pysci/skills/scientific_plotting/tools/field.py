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
- :func:`ceil_label` — colorbar 端标/场图 vmax 的默认取整（v2 阶梯：``d1.d2×10^k``，
  第二位 ``d2∈{0,5}``，第一位为 1 时另含 2/4/5/6/8；向上取整）。旧名
  :func:`ceil_one_sig` 保留为等价包装。
- :func:`display_region_max` — 显示区（严格 y>0 气域）实测 max 的**单一真源**（vmax-check 与 field_schema 导出共用）。
- :func:`check_vmax_adaptive` — H7 硬项：vmax 对显示区实测 max|value| 合格（默认
  ``ceil-one-sig`` 口径 = :func:`ceil_label`（实测 max）期望值比对；``tight`` = 旧窗 [1.0, 1.1]）。
- :func:`check_vmax_group` — 组共享量程口径：同一 vmax 多面板时只看**组内实测最大**面板的 ratio。

多面板组装见 raster.compose_grid 的 ``kind: field`` 面板（与共享 colorbar 同 cmap/norm）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

#: 支持的场源后缀 → 读取器类别
_VTK_SUFFIX = (".vtk", ".vtu")
_CSV_SUFFIX = (".csv", ".txt")


#: 端标 v2 阶梯的尾数（按升序）：``d1.d2`` 形态，``d2∈{0,5}``；``d1=1`` 时另含 2/4/6/8。
_LABEL_MANTISSAS: tuple[Decimal, ...] = tuple(
    Decimal(s)
    for s in (
        "1.0",
        "1.2",
        "1.4",
        "1.5",
        "1.6",
        "1.8",
        "2.0",
        "2.5",
        "3.0",
        "3.5",
        "4.0",
        "4.5",
        "5.0",
        "5.5",
        "6.0",
        "6.5",
        "7.0",
        "7.5",
        "8.0",
        "8.5",
        "9.0",
        "9.5",
    )
)


def ceil_label(x: float) -> float:
    """按 colorbar 端标 **v2 阶梯**向上取整——场图 vmax 与色标端标的默认口径。

    合法值为 ``d1.d2 × 10^k``：第一位 ``d1`` 取 1–9 任意，第二位 ``d2∈{0,5}``；
    ``d1=1`` 时另含 ``d2∈{2,4,6,8}``（并集 ``{0,2,4,5,6,8}``，故 1.0 亦合法）；
    第三位及以后全为 0。返回 ``>= x`` 的**最小合法值**（向上取整，绝不裁峰）。
    已合规的值原样返回。示例：``65.48 → 70``、``1.4655 → 1.5``、``1.8277 → 2.0``、
    ``2.11 → 2.5``、``1.1968 → 1.2``、``0.6436 → 0.65``、``0.98 → 1.0``、``9.7 → 10``、
    ``0.151 → 0.16``。径向刻度（rmax）不适用本口径。

    Args:
        x: 正有限数（端标量值；负端标取 ``-ceil_label(abs(x))`` 由调用方对称处理）。

    Returns:
        ``>= x`` 的最小 v2 合法值（float）。

    Raises:
        ValueError: 非正或非有限输入。
    """
    if not np.isfinite(x) or x <= 0:
        raise ValueError(f"ceil_label 需要正有限数：{x!r}")
    d = Decimal(repr(float(x)))
    e = d.adjusted()
    while True:
        for mant in _LABEL_MANTISSAS:
            cand = mant.scaleb(e)
            if cand >= d:
                return float(cand)
        e += 1


def ceil_one_sig(x: float) -> float:
    """历史名称（一位有效数字口径）的兼容包装——现等价于 :func:`ceil_label`。

    v2 阶梯放宽了第二位（``{0,5}``，``d1=1`` 另含 2/4/5/6/8），旧调用点无需改动即随新规；
    新代码请直接用 :func:`ceil_label`。

    Args:
        x: 正有限数（端标量值）。

    Returns:
        同 :func:`ceil_label`（``>= x`` 的最小 v2 合法值）。

    Raises:
        ValueError: 非正或非有限输入。
    """
    return ceil_label(x)


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
            物理域外（如固体壁/散射体间隙）架桥产生伪影三角形；用物理域判据遮罩其质心即可去除。
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
        values_by_group: ``{组名: 值数组}``；缺省量程按该组自身 min/max 取，端标按
            :func:`ceil_label` 口径朝外取整。
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
        if vmax and key in vmax:
            hi = vmax[key]
        else:
            hi = float(np.max(np.abs(vals)) if symmetric else np.max(vals))
            if hi > 0:
                hi = ceil_label(hi)
        if vmin and key in vmin:
            lo = vmin[key]
        else:
            lo = float(-hi if symmetric else np.min(vals))
            if lo < 0:
                lo = -ceil_label(-lo)
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo >= hi:
            raise ValueError(f"组 {key} 量程不可分：vmin={lo}, vmax={hi}")
        out[key] = Normalize(vmin=lo, vmax=hi)
    return out


@dataclass(frozen=True)
class DisplayRegionMax:
    """显示区（严格 y>0 气域）实测读数——F9 单一真源。"""

    max_abs: float
    n_points: int
    region: str


#: vmax-check 判定口径（裁决 override 2026-10-27T12:24:52，preference-journal 留痕；
#: 2026-10-27 用户裁决放宽为 v2 阶梯）：``ceil-one-sig`` = 端标 = :func:`ceil_label`
#: （显示区实测 max）期望值比对（默认，D2 端标口径——裁峰与偏离取整都拦）；``tight`` =
#: 旧比值窗 ``[lo, hi]``（遗留兼容）。口径键名保留，语义已随 v2 阶梯更新。
ROUNDINGS: tuple[str, ...] = ("ceil-one-sig", "tight")


def judge_vmax(
    vmax: float,
    measured_max: float,
    ratio: float,
    *,
    rounding: str,
    lo: float = 1.0,
    hi: float = 1.1,
) -> tuple[bool, float | None]:
    """vmax 合格判定的**单一真源**（:func:`check_vmax_adaptive` 与 :func:`check_vmax_group` 共用）。

    Args:
        vmax: 色标上限。
        measured_max: 显示区实测 max|value|。
        ratio: ``vmax / measured_max``。
        rounding: :data:`ROUNDINGS` 之一。
        lo/hi: 仅 ``tight`` 口径使用的容许比值窗。

    Returns:
        ``(ok, expected_vmax)``——``expected_vmax`` 仅 ceil-one-sig 口径有值（期望端标；
        实测 max=0 时 ``None``，该情形恒不合格）。

    Raises:
        ValueError: 未知口径。
    """
    if rounding == "ceil-one-sig":
        expected = ceil_label(measured_max) if measured_max > 0 else None
        ok = expected is not None and bool(
            np.isclose(vmax, expected, rtol=1e-9, atol=1e-12)
        )
        return ok, expected
    if rounding == "tight":
        return bool(lo <= ratio <= hi), None
    raise ValueError(f"未知取整口径：{rounding!r}（合法：{ROUNDINGS}）")


@dataclass(frozen=True)
class VmaxCheck:
    """H7 vmax 自适应读数：``vmax / 显示区实测 max|value|``。"""

    ok: bool
    ratio: float
    vmax: float
    measured_max: float
    n_points: int
    region: str
    rounding: str = "tight"
    expected_vmax: float | None = None


@dataclass(frozen=True)
class VmaxGroupMember:
    """组共享量程下单个面板的读数（``role`` = driver 或 weak）。"""

    source: str
    check: VmaxCheck
    role: str

    @property
    def in_window(self) -> bool:
        """该面板自身的 ratio 是否落在容许窗内（弱侧成员必然为 False）。"""
        return self.check.ok


@dataclass(frozen=True)
class VmaxGroupCheck:
    """组共享量程判定：组内**最大实测**面板的 ratio ∈ [lo, hi] 才算通过。

    组共享口径（同一 vmax 用于多面板）下，弱侧面板 ratio 必然 >1.1；真正决定 vmax
    是否自适应的是组内显示区实测 max 最大的面板（driver）。若 vmax 低于 driver 峰值
    （ratio < lo），driver 判据同样拦得住——裁峰是比"窗偏大"更严重的错误。
    """

    ok: bool
    vmax: float
    lo: float
    hi: float
    members: tuple[VmaxGroupMember, ...]
    driver_index: int

    @property
    def driver(self) -> VmaxGroupMember:
        return self.members[self.driver_index]


def check_vmax_group(
    checks: Sequence[VmaxCheck],
    *,
    sources: Sequence[str] | None = None,
    lo: float = 1.0,
    hi: float = 1.1,
    rounding: str = "tight",
) -> VmaxGroupCheck:
    """按**组共享量程**口径判定一组面板（同一 vmax）。

    Args:
        checks: 各面板的 :class:`VmaxCheck`（须用同一个 vmax 算得）。
        sources: 与 ``checks`` 同序的源标签（缺省用序号），供输出区分 driver/weak。
        lo/hi: 容许比值区间（默认 [1.0, 1.1]；仅 ``tight`` 口径生效）。
        rounding: driver 的判定口径（:data:`ROUNDINGS`，同 :func:`judge_vmax`）。

    Returns:
        :class:`VmaxGroupCheck`；``driver_index`` 指向实测 max 最大者（并列取首个）。

    Raises:
        ValueError: 空组、vmax 不一致或区间非法。
    """
    if not checks:
        raise ValueError("check_vmax_group 需要至少一个面板读数")
    if not (0 < lo <= hi):
        raise ValueError(f"允许区间非法：lo={lo}, hi={hi}")
    labels = (
        list(sources) if sources is not None else [str(i) for i in range(len(checks))]
    )
    if len(labels) != len(checks):
        raise ValueError(
            f"sources 与 checks 数量不一致：{len(labels)} vs {len(checks)}"
        )
    vmaxes = {float(c.vmax) for c in checks}
    if len(vmaxes) != 1:
        raise ValueError(f"组共享口径要求同一 vmax，实际：{sorted(vmaxes)}")
    driver_index = max(range(len(checks)), key=lambda i: checks[i].measured_max)
    members = tuple(
        VmaxGroupMember(
            source=labels[i],
            check=c,
            role="driver" if i == driver_index else "weak",
        )
        for i, c in enumerate(checks)
    )
    driver = checks[driver_index]
    ok, _expected = judge_vmax(
        driver.vmax, driver.measured_max, driver.ratio, rounding=rounding, lo=lo, hi=hi
    )
    return VmaxGroupCheck(
        ok=ok,
        vmax=float(driver.vmax),
        lo=lo,
        hi=hi,
        members=members,
        driver_index=driver_index,
    )


def display_region_max(
    xy: np.ndarray,
    values: np.ndarray,
    *,
    center: tuple[float, float] = (0.0, 0.0),
    radius: float | None = None,
    y_min: float = 0.0,
) -> DisplayRegionMax:
    """显示区实测 max|value| 的**单一真源**（严格 ``y > y_min`` 气域）。

    排除 y≤0 的散射体域与 y=0 口径线行；给了 ``radius`` 时再取圆内
    ``(x-cx)^2 + (y-cy)^2 <= radius^2``。`check_vmax_adaptive` 与 field_schema 的
    ``display_y_positive_max_abs_real_p_t`` 导出端共用本函数。
    """
    if radius is not None and (not np.isfinite(radius) or radius <= 0):
        raise ValueError(f"radius 须为正有限数：{radius!r}")
    pts = np.asarray(xy, dtype=float)
    vals = np.asarray(values, dtype=float).ravel()
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"xy 须为 Nx2：{pts.shape!r}")
    if pts.shape[0] != vals.size:
        raise ValueError(f"点数不一致：xy {pts.shape[0]} vs values {vals.size}")
    cx, cy = float(center[0]), float(center[1])
    mask = (pts[:, 1] > y_min) & np.isfinite(vals)
    region = f"y>{y_min:g}"
    if radius is not None:
        mask &= (pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2 <= radius**2
        region += f", r<={radius:g}@({cx:g},{cy:g})"
    if not mask.any():
        raise ValueError(f"显示区无数据点（{region}）")
    return DisplayRegionMax(
        max_abs=float(np.max(np.abs(vals[mask]))),
        n_points=int(mask.sum()),
        region=region,
    )


def check_vmax_adaptive(
    xy: np.ndarray,
    values: np.ndarray,
    vmax: float,
    *,
    center: tuple[float, float] = (0.0, 0.0),
    radius: float | None = None,
    y_min: float = 0.0,
    lo: float = 1.0,
    hi: float = 1.1,
    rounding: str = "tight",
) -> VmaxCheck:
    """硬项判据：色标 ``vmax`` 对**显示区实测** max|value| 合格（口径见 ``rounding``）。

    显示区＝**严格 ``y > y_min``**（默认 y>0 气域：排除 y=0 口径线行与散射体域）且
    （给了 ``radius`` 时）``r <= radius``（圆心 ``center``）。
    实测只取该区内的有限值；必须读面板数据本身，不接受生产者声明的 vmax 说明。

    Args:
        xy: Nx2 点坐标。
        values: N 个值（可为负，按绝对值取 max）。
        vmax: 色标上限（>0）。
        center: 显示半圆中心（x, y）。
        radius: 半圆半径；None = 只按 ``y_min`` 裁。
        y_min: 显示区下界（默认 0，即只取上半域）。
        lo/hi: 允许比值区间（默认 [1.0, 1.1]；仅 ``tight`` 口径生效）。
        rounding: 判定口径（:data:`ROUNDINGS`）——``ceil-one-sig``：端标须恰为
            :func:`ceil_label`（实测 max，v2 阶梯）（裁决 2026-10-27，D2 端标口径）；
            ``tight``：比值须 ∈ ``[lo, hi]``（遗留窗，默认）。

    Returns:
        :class:`VmaxCheck`；`measured_max=0` 时 ratio 为 inf、``ok=False``。
    """
    if not np.isfinite(vmax) or vmax <= 0:
        raise ValueError(f"vmax 须为正有限数：{vmax!r}")
    if not (0 < lo <= hi):
        raise ValueError(f"允许区间非法：lo={lo}, hi={hi}")
    stats = display_region_max(xy, values, center=center, radius=radius, y_min=y_min)
    ratio = float("inf") if stats.max_abs == 0 else vmax / stats.max_abs
    ok, expected = judge_vmax(
        vmax, stats.max_abs, ratio, rounding=rounding, lo=lo, hi=hi
    )
    return VmaxCheck(
        ok=ok,
        ratio=ratio,
        vmax=float(vmax),
        measured_max=stats.max_abs,
        n_points=stats.n_points,
        region=stats.region,
        rounding=rounding,
        expected_vmax=expected,
    )


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
    """单场面板渲染 + colorbar + savefig 一步到位（colorbar 直接取面板 mappable → 严格一致）。

    ``vmin``/``vmax`` 缺省时按数据范围两端取，并按端标口径（:func:`ceil_label`）朝外取整。
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    xy, vals = load_field_points(source, scalars=scalars, cols=cols)
    if vmax is None and float(np.max(vals)) > 0:
        vmax = ceil_label(float(np.max(vals)))
    if vmin is None and float(np.min(vals)) < 0:
        vmin = -ceil_label(-float(np.min(vals)))
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
