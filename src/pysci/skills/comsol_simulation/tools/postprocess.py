"""后处理与验证：Agent「评估」仿真的程序化通道（去 GUI 的眼睛与尺子）。

两条输入通道：
- **网格/场文件**（VTK/VTU/STL/PLY，由 :mod:`export` 产出）→ pyvista 离屏渲染 + 程序化场分析。
- **COMSOL CSV**（``% key,value`` 头 + 数据列）→ numpy 数组，做数值后处理与理论比对。

能力分组：
- 渲染：:func:`render_grid`（离屏 PNG，几何/网格/场，去 GUI 的"眼睛"）。
- 场分析：:func:`field_stats` / :func:`slice_grid` / :func:`probe_point` / :func:`integrate_scalar`。
- 网格质量：:func:`mesh_quality`。
- 收敛性：:func:`convergence_order`（h-refinement 观测收敛阶）。
- 验证器：:func:`compare_to_reference` / :func:`check_conservation` / :func:`validate_field`，
  把"评估"固化为可复用检查。
- 传递函数：:func:`transfer_function`（复数比，gain_ep 类研究的通用原语）。

pyvista 为**惰性导入**：纯数值函数（CSV 解析、收敛阶、比对）不依赖 pyvista/VTK，
可在无图形栈的环境（CI、纯 Python 单测）中运行；仅渲染类函数需要 pyvista。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# COMSOL CSV 解析
# ---------------------------------------------------------------------------


@dataclass
class ComsolTable:
    """解析后的 COMSOL 导出 CSV：``% key,value`` 元信息 + 列名 + 数据矩阵。"""

    meta: dict[str, str] = field(default_factory=dict)
    columns: list[str] = field(default_factory=list)
    data: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

    def column(self, name: str) -> np.ndarray:
        """按列名取一列；不存在则抛 KeyError。"""
        idx = self.columns.index(name)
        return self.data[:, idx]

    def has(self, name: str) -> bool:
        return name in self.columns


def read_comsol_csv(path: str | Path) -> ComsolTable:
    """解析 COMSOL 导出的 CSV（``%`` 注释头 + 可选列名行 + 数值行）。

    兼容多种形态（判据统一为「``%`` 行首字段是坐标列 x/X」，避免误抓 ``% Model,...`` 等元信息行）：
    - 数据集导出（``% Expressions,0``）：列名取 ``% X,Y`` 行（大写坐标列）；
    - 表导出：列名取 ``% x,y,Value`` 行（小写坐标列 + 值列）；
    - 含场表达式：列名行形如 ``% X,Y,acpr.p_t``。
    """
    p = Path(path)
    meta: dict[str, str] = {}
    header_line: str | None = None
    rows: list[list[float]] = []

    with p.open(encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("%"):
                body = line.lstrip("%").strip()
                # 列名行：body 以坐标名前缀开头（"X" 或 "X,Y[,expr...]")
                if body.split(",")[0].strip().lower() == "x":
                    header_line = body
                    continue
                if "," in body:
                    k, _, v = body.partition(",")
                    meta[k.strip()] = v.strip()
                continue
            parts = [x for x in line.split(",")]
            if all(_is_num(x) for x in parts):
                rows.append([float(x) for x in parts])
            elif header_line is None:
                header_line = line  # 首个非数值行当作列名

    data = np.array(rows, dtype=float) if rows else np.zeros((0, 0))
    columns: list[str] = []
    if header_line:
        columns = [c.strip() for c in header_line.split(",")]
    # 列名与数据列数对齐：不足补 colN，多余截断
    ncols = data.shape[1] if data.ndim == 2 else 0
    if len(columns) < ncols:
        columns += [f"col{i}" for i in range(len(columns), ncols)]
    columns = columns[:ncols]
    return ComsolTable(meta=meta, columns=columns, data=data)


def _is_num(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# pyvista 惰性加载
# ---------------------------------------------------------------------------
def _pv() -> Any:
    """惰性导入 pyvista 并切到离屏渲染。"""
    import pyvista as pv  # noqa: PLC0415 - 渲染才需要，避免无图形栈环境导入失败

    pv.OFF_SCREEN = True
    return pv


def load_grid(path: str | Path) -> Any:
    """读取网格/场文件（VTK/VTU/STL/PLY/...）为 pyvista 数据集。"""
    pv = _pv()
    return pv.read(str(path))


# ---------------------------------------------------------------------------
# 渲染（去 GUI 的眼睛）
# ---------------------------------------------------------------------------
def render_grid(
    grid: Any,
    out_png: str | Path,
    *,
    scalars: str | None = None,
    cmap: str = "coolwarm",
    show_edges: bool = False,
    clip_normal: tuple[float, float, float] | None = None,
    clip_origin: tuple[float, float, float] | None = None,
    window_size: tuple[int, int] = (1200, 900),
    title: str | None = None,
) -> Path:
    """把网格/场离屏渲染为 PNG。

    Args:
        grid: pyvista 数据集（:func:`load_grid` 的产物）。
        out_png: 输出 PNG。
        scalars: 要着色的点/胞标量名；None → 几何/网格本色。
        cmap: matplotlib 色表名。
        show_edges: 是否画网格边（看网格质量用）。
        clip_normal/clip_origin: 若给则先沿法向剖切再渲染（看内部场）。
        window_size: 渲染窗口像素。
        title: 图标题。
    """
    pv = _pv()
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)

    target = grid
    if clip_normal is not None:
        target = grid.clip(normal=clip_normal, origin=clip_origin or (0, 0, 0))

    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    if scalars is not None and target.point_data and scalars in target.point_data:
        plotter.add_mesh(target, scalars=scalars, cmap=cmap, show_edges=show_edges)
    elif scalars is not None and target.cell_data and scalars in target.cell_data:
        plotter.add_mesh(target, scalars=scalars, cmap=cmap, show_edges=show_edges)
    else:
        plotter.add_mesh(target, show_edges=show_edges, color="lightblue")
    if title:
        plotter.add_text(title, font_size=12)
    plotter.screenshot(str(out))
    plotter.close()
    return out


# ---------------------------------------------------------------------------
# 场分析
# ---------------------------------------------------------------------------
def field_stats(grid: Any, scalars: str) -> dict[str, float]:
    """标量场的 min/max/mean/rms/L2 范数统计。"""
    arr = _scalar_array(grid, scalars)
    if arr.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "rms": 0.0, "l2": 0.0, "n": 0}
    a = np.asarray(arr, dtype=float).ravel()
    return {
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "rms": float(np.sqrt(np.mean(a**2))),
        "l2": float(np.linalg.norm(a)),
        "n": int(a.size),
    }


def _scalar_array(grid: Any, scalars: str) -> np.ndarray:
    if scalars in getattr(grid, "point_data", {}):
        return np.asarray(grid.point_data[scalars])
    if scalars in getattr(grid, "cell_data", {}):
        return np.asarray(grid.cell_data[scalars])
    raise KeyError(f"数据集无标量 '{scalars}'；可用：{list(getattr(grid, 'array_names', []))}")


def slice_grid(
    grid: Any,
    normal: tuple[float, float, float] = (0, 0, 1),
    origin: tuple[float, float, float] | None = None,
) -> Any:
    """沿法向剖切，返回剖面数据集（看内部场分布）。"""
    return grid.clip(normal=normal, origin=origin or grid.center)


def probe_point(grid: Any, point: tuple[float, float, float], scalars: str) -> float:
    """在任意点插值取标量值（探针）。"""
    sampled = grid.sample(np.array([point], dtype=float))
    arr = _scalar_array(sampled, scalars)
    return float(np.asarray(arr).ravel()[0])


def integrate_scalar(grid: Any, scalars: str) -> float:
    """对标量场做体/面积分（守恒检查用）。"""
    arr = np.asarray(_scalar_array(grid, scalars), dtype=float)
    areas = grid.cell_data.get("Area", None) if grid.cell_data else None
    if grid.point_data and scalars in grid.point_data:
        # 点标量 → 用胞体积/面积加权（若可得），否则直接求和近似
        vols = grid.compute_cell_sizes().cell_data.get("Volume", grid.compute_cell_sizes().cell_data.get("Area", None))
        if vols is not None:
            cell_avg = grid.point_data_to_cell_data().cell_data[scalars]
            return float(np.sum(np.asarray(cell_avg, dtype=float) * np.asarray(vols, dtype=float)))
        return float(np.sum(arr))
    if areas is not None:
        return float(np.sum(arr * np.asarray(areas, dtype=float)))
    return float(np.sum(arr))


# ---------------------------------------------------------------------------
# 网格质量
# ---------------------------------------------------------------------------
def mesh_quality(mesh: Any) -> dict[str, Any]:
    """网格基本质量统计：单元数、点数为必给；质量度量尽力而为。"""
    out: dict[str, Any] = {
        "n_cells": int(mesh.n_cells),
        "n_points": int(mesh.n_points),
        "bounds": tuple(float(b) for b in mesh.bounds),
    }
    try:
        q = mesh.cell_quality()  # pyvista ≥0.43 提供多种度量
        for name in ("volume", "area", "scaled_jacobian", "skew", "quality"):
            if name in q.cell_data:
                arr = np.asarray(q.cell_data[name], dtype=float)
                out[f"{name}_min"] = float(np.nanmin(arr))
                out[f"{name}_mean"] = float(np.nanmean(arr))
                out[f"{name}_max"] = float(np.nanmax(arr))
    except Exception:  # noqa: BLE001 - 度量不可用时降级
        pass
    return out


# ---------------------------------------------------------------------------
# 收敛性（h-refinement）
# ---------------------------------------------------------------------------
def convergence_order(hs: list[float], errs: list[float]) -> float:
    """由 (网格尺寸 h, 误差 err) 序列拟合观测收敛阶 p（log-log 最小二乘）。"""
    h = np.asarray(hs, dtype=float)
    e = np.asarray(errs, dtype=float)
    mask = (h > 0) & (e > 0)
    if mask.sum() < 2:
        return float("nan")
    lh = np.log(h[mask])
    le = np.log(e[mask])
    p, _ = np.polyfit(lh, le, 1)
    return float(p)


# ---------------------------------------------------------------------------
# 验证器（把"评估"固化为可复用检查）
# ---------------------------------------------------------------------------
@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str

    def report(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


def compare_to_reference(
    measured: np.ndarray,
    reference: np.ndarray,
    *,
    rtol: float = 1e-2,
    atol: float = 1e-8,
    name: str = "compare_to_reference",
) -> CheckResult:
    """与理论/参考解比对：逐点 allclose + 报告最大相对偏差。"""
    m = np.asarray(measured, dtype=complex if np.iscomplexobj(measured) else float)
    r = np.asarray(reference, dtype=complex if np.iscomplexobj(reference) else float)
    if m.shape != r.shape:
        return CheckResult(name, False, f"形状不匹配 {m.shape} vs {r.shape}")
    denom = np.maximum(np.abs(r), atol)
    rel = np.abs(m - r) / denom
    max_rel = float(np.max(rel)) if rel.size else 0.0
    ok = bool(np.allclose(m, r, rtol=rtol, atol=atol))
    return CheckResult(name, ok, f"max_rel_dev={max_rel:.3e} (rtol={rtol})")


def check_conservation(
    inflow: float,
    outflow: float,
    *,
    rtol: float = 1e-2,
    name: str = "conservation",
) -> CheckResult:
    """守恒检查：|in-out|/|in| <= rtol（能量/通量守恒）。"""
    if abs(inflow) < 1e-300:
        return CheckResult(name, abs(outflow) < 1e-300, f"inflow≈0, outflow={outflow:.3e}")
    dev = abs(inflow - outflow) / abs(inflow)
    return CheckResult(name, dev <= rtol, f"rel_dev={dev:.3e} (rtol={rtol})")


def validate_field(
    grid: Any,
    scalars: str,
    *,
    finite: bool = True,
    max_abs: float | None = None,
    name: str = "validate_field",
) -> CheckResult:
    """物理合理性检查：场有限、（可选）幅值上界。"""
    arr = np.asarray(_scalar_array(grid, scalars), dtype=float).ravel()
    if finite and not np.all(np.isfinite(arr)):
        return CheckResult(name, False, "场含 NaN/Inf")
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if max_abs is not None and peak > max_abs:
        return CheckResult(name, False, f"峰值 {peak:.3e} 超上界 {max_abs:.3e}")
    return CheckResult(name, True, f"finite ok, peak={peak:.3e}")


# ---------------------------------------------------------------------------
# 传递函数（复数比）
# ---------------------------------------------------------------------------
def transfer_function(
    output: np.ndarray | complex,
    input_: np.ndarray | complex,
) -> np.ndarray:
    """复数传递函数 H = output / input（逐元素）。"""
    o = np.asarray(output, dtype=complex)
    i = np.asarray(input_, dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = o / i
    return h


def db(x: np.ndarray | float) -> np.ndarray:
    """幅值转分贝 20*log10|x|。"""
    a = np.abs(np.asarray(x, dtype=complex if np.iscomplexobj(x) else float))
    with np.errstate(divide="ignore"):
        return 20.0 * np.log10(np.maximum(a, 1e-300))


# ---------------------------------------------------------------------------
# 解析参考解示例：刚壁硬波导平面波
# ---------------------------------------------------------------------------
def rigid_waveguide_pressure(
    x: np.ndarray,
    *,
    freq: float,
    c0: float = 343.0,
    amplitude: float = 1.0,
    direction: float = 1.0,
) -> np.ndarray:
    """刚壁波导中沿 x 的平面波解析解 p = A*exp(i*k*x*direction)（低于截止频率的单模）。"""
    k = 2.0 * math.pi * freq / c0
    return amplitude * np.exp(1j * k * np.asarray(x, dtype=float) * direction)
