"""拓扑探索：零点集、等值面、发散点/奇点、临界点检测、环绕数。

吸收并大幅扩展旧 ``pysci.common.space_curve.vis_complex_equation``，提供面向
参数空间拓扑结构分析的工具集：复变函数零点集提取、标量场等值面、奇点检测、
临界点分类（Morse 指标）以及拓扑荷（环绕数）计算。

核心函数：
- ``find_zero_set(func, param_space)``：复变函数零点集（等值面交集法）
- ``find_isosurface(func, param_space, level)``：标量场等值面
- ``find_singularities(func, param_space)``：发散点/奇点检测
- ``find_critical_points(func, param_space)``：临界点 + Morse 指标
- ``compute_winding_number(func, loop)``：环绕数（拓扑荷）

用法::

    from pysci.skills.theoretical_computation.tools import topology, numerical

    space = numerical.ParamSpace(axes=[...])
    func = numerical.lambdify_expr(expr, space)
    result = topology.find_zero_set(func, space)
    # result.curve, result.real_face, result.imag_face → pyvista PolyData
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .numerical import ParamSpace, make_meshgrid


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------
@dataclass
class ZeroSetResult:
    """复变函数零点集提取结果。

    Attributes:
        curve: 零点集交线（pyvista PolyData）。
        real_face: Re(f)=0 等值面。
        imag_face: Im(f)=0 等值面。
        grid: 原始 pyvista ImageData 网格。
        n_points: 零点集上的点数。
    """

    curve: Any  # pv.PolyData
    real_face: Any  # pv.PolyData
    imag_face: Any  # pv.PolyData
    grid: Any  # pv.ImageData
    n_points: int = 0


@dataclass
class IsosurfaceResult:
    """标量场等值面提取结果。

    Attributes:
        surface: 等值面（pyvista PolyData）。
        grid: 原始网格。
        level: 等值面值。
        n_points: 等值面上的点数。
    """

    surface: Any
    grid: Any
    level: float
    n_points: int = 0


@dataclass
class SingularityResult:
    """奇点/发散点检测结果。

    Attributes:
        locations: 奇点坐标数组，形状 (n_singularities, ndim)。
        magnitudes: 各奇点处的 |f| 值。
        threshold: 使用的检测阈值。
    """

    locations: NDArray[np.floating]
    magnitudes: NDArray[np.floating]
    threshold: float


@dataclass
class CriticalPointResult:
    """临界点检测结果。

    Attributes:
        locations: 临界点坐标，形状 (n_points, ndim)。
        values: 各临界点处的函数值。
        morse_indices: Morse 指标（Hessian 负特征值个数）。
        hessian_eigenvalues: 各临界点的 Hessian 本征值。
    """

    locations: NDArray[np.floating]
    values: NDArray[np.floating]
    morse_indices: NDArray[np.integer]
    hessian_eigenvalues: list[NDArray[np.floating]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 网格构建辅助
# ---------------------------------------------------------------------------
def _build_pyvista_grid(param_space: ParamSpace, scalars: dict[str, NDArray]) -> Any:
    """构建 pyvista ImageData 网格并附加标量场。

    Args:
        param_space: 参数空间定义。
        scalars: 标量场字典（名称 → 扁平化数组）。

    Returns:
        pyvista ImageData。
    """
    import pyvista as pv

    grid = pv.ImageData()
    dims = param_space.resolutions
    grid.dimensions = dims

    # 设置原点和间距
    origins = [ax.range[0] for ax in param_space.axes]
    spacings = [
        ax.length / (ax.resolution - 1) if ax.resolution > 1 else 1.0
        for ax in param_space.axes
    ]
    grid.origin = tuple(origins)
    grid.spacing = tuple(spacings)

    # 附加标量场（Fortran order for VTK）
    for name, data in scalars.items():
        grid[name] = data.flatten(order="F")

    return grid


# ---------------------------------------------------------------------------
# 零点集
# ---------------------------------------------------------------------------
def find_zero_set(
    func: Callable[..., NDArray],
    param_space: ParamSpace,
) -> ZeroSetResult:
    """提取复变函数在参数空间中的零点集（Re=0 与 Im=0 等值面的交集）。

    适用于 3D 参数空间中的复变方程 f(x,y,z) = 0 的解集可视化。
    零点集通常是一条空间曲线（两个等值面的交集）。

    Args:
        func: 复值数值函数（接受 N 个网格数组，返回复数数组）。
        param_space: 3D 参数空间定义。

    Returns:
        ZeroSetResult 容器（含 curve/real_face/imag_face/grid）。

    Raises:
        ValueError: 参数空间不是 3D 时。
    """
    import pyvista as pv

    if param_space.ndim != 3:
        raise ValueError(f"零点集提取需要 3D 参数空间，得到 {param_space.ndim}D")

    # 计算网格上的函数值
    mesh = make_meshgrid(param_space)
    values = func(*mesh)

    real_part = np.real(values)
    imag_part = np.imag(values)

    # 构建 pyvista 网格
    grid = _build_pyvista_grid(param_space, {"real": real_part, "imag": imag_part})

    # 提取等值面
    real_contour = grid.contour(isosurfaces=[0], scalars="real")
    imag_contour = grid.contour(isosurfaces=[0], scalars="imag")

    # 计算交线
    curve = pv.PolyData()
    first_face = real_contour
    second_face = imag_contour

    if real_contour.n_points > 0 and imag_contour.n_points > 0:
        try:
            curve, first_face, second_face = real_contour.intersection(
                imag_contour, split_first=False, split_second=False
            )
        except Exception:  # noqa: BLE001
            # 交集计算失败时回退
            curve = pv.PolyData()

    return ZeroSetResult(
        curve=curve,
        real_face=first_face,
        imag_face=second_face,
        grid=grid,
        n_points=curve.n_points if hasattr(curve, "n_points") else 0,
    )


# ---------------------------------------------------------------------------
# 等值面
# ---------------------------------------------------------------------------
def find_isosurface(
    func: Callable[..., NDArray],
    param_space: ParamSpace,
    level: float = 0.0,
    scalar_name: str = "value",
) -> IsosurfaceResult:
    """提取标量场等值面。

    Args:
        func: 实值数值函数。
        param_space: 3D 参数空间定义。
        level: 等值面值。
        scalar_name: 标量场名称。

    Returns:
        IsosurfaceResult 容器。
    """
    if param_space.ndim != 3:
        raise ValueError(f"等值面提取需要 3D 参数空间，得到 {param_space.ndim}D")

    mesh = make_meshgrid(param_space)
    values = np.real(func(*mesh))

    grid = _build_pyvista_grid(param_space, {scalar_name: values})
    surface = grid.contour(isosurfaces=[level], scalars=scalar_name)

    return IsosurfaceResult(
        surface=surface,
        grid=grid,
        level=level,
        n_points=surface.n_points if hasattr(surface, "n_points") else 0,
    )


def find_multiple_isosurfaces(
    func: Callable[..., NDArray],
    param_space: ParamSpace,
    levels: list[float],
    scalar_name: str = "value",
) -> list[IsosurfaceResult]:
    """提取多个等值面。

    Args:
        func: 实值数值函数。
        param_space: 3D 参数空间定义。
        levels: 等值面值列表。
        scalar_name: 标量场名称。

    Returns:
        IsosurfaceResult 列表。
    """
    if param_space.ndim != 3:
        raise ValueError(f"等值面提取需要 3D 参数空间，得到 {param_space.ndim}D")

    mesh = make_meshgrid(param_space)
    values = np.real(func(*mesh))
    grid = _build_pyvista_grid(param_space, {scalar_name: values})

    results = []
    for level in levels:
        surface = grid.contour(isosurfaces=[level], scalars=scalar_name)
        results.append(IsosurfaceResult(
            surface=surface,
            grid=grid,
            level=level,
            n_points=surface.n_points if hasattr(surface, "n_points") else 0,
        ))

    return results


# ---------------------------------------------------------------------------
# 奇点检测
# ---------------------------------------------------------------------------
def find_singularities(
    func: Callable[..., NDArray],
    param_space: ParamSpace,
    threshold: float | None = None,
    percentile: float = 99.5,
) -> SingularityResult:
    """检测参数空间中的发散点/奇点。

    判据：|f| 超过阈值（默认为所有值的 99.5 百分位数）的区域。

    Args:
        func: 数值函数（实值或复值）。
        param_space: 参数空间定义。
        threshold: 检测阈值。若为 None，自动用百分位数确定。
        percentile: 自动阈值的百分位数。

    Returns:
        SingularityResult 容器。
    """
    mesh = make_meshgrid(param_space)
    values = func(*mesh)

    if np.iscomplexobj(values):
        magnitude = np.abs(values)
    else:
        magnitude = np.abs(np.real(values))

    # 处理 inf/nan
    finite_mask = np.isfinite(magnitude)
    if not np.any(finite_mask):
        return SingularityResult(
            locations=np.empty((0, param_space.ndim)),
            magnitudes=np.empty(0),
            threshold=threshold or 0.0,
        )

    if threshold is None:
        finite_mags = magnitude[finite_mask]
        threshold = float(np.percentile(finite_mags, percentile))

    # 找到超过阈值的点
    sing_mask = (magnitude > threshold) | ~finite_mask
    sing_indices = np.argwhere(sing_mask)

    if len(sing_indices) == 0:
        return SingularityResult(
            locations=np.empty((0, param_space.ndim)),
            magnitudes=np.empty(0),
            threshold=threshold,
        )

    # 将网格索引转换为物理坐标
    locations = np.zeros((len(sing_indices), param_space.ndim))
    for dim, axis in enumerate(param_space.axes):
        axis_vals = axis.values
        locations[:, dim] = axis_vals[sing_indices[:, dim]]

    magnitudes = magnitude[sing_mask]

    return SingularityResult(
        locations=locations,
        magnitudes=magnitudes,
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# 临界点检测
# ---------------------------------------------------------------------------
def find_critical_points(
    func: Callable[..., NDArray],
    param_space: ParamSpace,
    gradient_tol: float = 1e-3,
    hessian_step: float | None = None,
) -> CriticalPointResult:
    """检测标量场的临界点（梯度为零）并分类（Morse 指标）。

    使用数值梯度（中心差分）在网格上搜索梯度模接近零的点，
    然后通过 Hessian 矩阵的本征值符号分类：
    - Morse 指标 0：极小值
    - Morse 指标 ndim：极大值
    - 其他：鞍点

    Args:
        func: 实值数值函数。
        param_space: 参数空间定义。
        gradient_tol: 梯度模的容差（相对于最大值归一化）。
        hessian_step: Hessian 计算的步长（默认为网格间距的 1/10）。

    Returns:
        CriticalPointResult 容器。
    """
    mesh = make_meshgrid(param_space)
    values = np.real(func(*mesh))

    # 计算数值梯度
    spacings = [
        ax.length / (ax.resolution - 1) if ax.resolution > 1 else 1.0
        for ax in param_space.axes
    ]
    gradients = np.gradient(values, *spacings)

    # 梯度模
    grad_magnitude = np.sqrt(sum(g**2 for g in gradients))
    max_grad = np.max(grad_magnitude) + 1e-15

    # 找到梯度接近零的点
    crit_mask = (grad_magnitude / max_grad) < gradient_tol
    crit_indices = np.argwhere(crit_mask)

    if len(crit_indices) == 0:
        return CriticalPointResult(
            locations=np.empty((0, param_space.ndim)),
            values=np.empty(0),
            morse_indices=np.empty(0, dtype=int),
        )

    # 转换为物理坐标
    ndim = param_space.ndim
    locations = np.zeros((len(crit_indices), ndim))
    for dim, axis in enumerate(param_space.axes):
        locations[:, dim] = axis.values[crit_indices[:, dim]]

    crit_values = values[crit_mask]

    # 计算 Hessian 并确定 Morse 指标
    if hessian_step is None:
        hessian_step = min(spacings) / 10.0

    morse_indices = np.zeros(len(crit_indices), dtype=int)
    hessian_eigs: list[NDArray] = []

    for i, loc in enumerate(locations):
        hessian = _numerical_hessian(func, loc, hessian_step, ndim)
        eigvals = np.linalg.eigvalsh(hessian)
        morse_indices[i] = int(np.sum(eigvals < 0))
        hessian_eigs.append(eigvals)

    return CriticalPointResult(
        locations=locations,
        values=crit_values,
        morse_indices=morse_indices,
        hessian_eigenvalues=hessian_eigs,
    )


def _numerical_hessian(
    func: Callable,
    point: NDArray,
    step: float,
    ndim: int,
) -> NDArray:
    """数值计算 Hessian 矩阵（中心差分）。"""
    hessian = np.zeros((ndim, ndim))
    f0 = float(np.real(func(*point)))

    for i in range(ndim):
        for j in range(i, ndim):
            # 四点中心差分
            p_pp = point.copy(); p_pp[i] += step; p_pp[j] += step
            p_pm = point.copy(); p_pm[i] += step; p_pm[j] -= step
            p_mp = point.copy(); p_mp[i] -= step; p_mp[j] += step
            p_mm = point.copy(); p_mm[i] -= step; p_mm[j] -= step

            f_pp = float(np.real(func(*p_pp)))
            f_pm = float(np.real(func(*p_pm)))
            f_mp = float(np.real(func(*p_mp)))
            f_mm = float(np.real(func(*p_mm)))

            hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * step * step)
            hessian[j, i] = hessian[i, j]

    return hessian


# ---------------------------------------------------------------------------
# 环绕数（拓扑荷）
# ---------------------------------------------------------------------------
def compute_winding_number(
    func: Callable[..., complex | NDArray],
    loop_points: NDArray,
) -> int:
    """计算复变函数沿闭合回路的环绕数（拓扑荷）。

    环绕数 = (1/2π) * ∮ d(arg(f))，即 f 沿回路的总相位变化除以 2π。

    Args:
        func: 复值函数（接受回路上的点坐标）。
        loop_points: 闭合回路上的采样点，形状 (n_points, ndim)。

    Returns:
        环绕数（整数）。
    """
    # 计算回路上各点的函数值
    values = np.array([func(*p) for p in loop_points], dtype=complex)

    # 计算相位
    phases = np.angle(values)

    # 计算相位差（处理 2π 跳跃）
    dphi = np.diff(phases, append=phases[0])
    # 将相位差归一化到 (-π, π]
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi

    # 总相位变化 / 2π = 环绕数
    winding = np.sum(dphi) / (2 * np.pi)
    return int(np.round(winding))


def compute_winding_number_2d(
    func: Callable[[NDArray, NDArray], NDArray],
    center: tuple[float, float],
    radius: float,
    n_points: int = 360,
) -> int:
    """计算 2D 参数空间中复变函数沿圆形回路的环绕数。

    Args:
        func: 复值函数 f(x, y)。
        center: 圆心坐标。
        radius: 回路半径。
        n_points: 回路采样点数。

    Returns:
        环绕数。
    """
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)

    values = func(x, y)
    phases = np.angle(values)

    dphi = np.diff(phases, append=phases[0])
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi

    winding = np.sum(dphi) / (2 * np.pi)
    return int(np.round(winding))
