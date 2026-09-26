"""数值化管线：N 维参数空间定义、lambdify 封装、网格采样与求值。

吸收并泛化旧 ``pysci.common.dtypes.ParamSpace3D`` 和 ``pysci.common.numerical.get_numpy_func``，
提供面向任意维度的参数空间定义与高效数值求值。

核心数据类：
- ``ParamAxis``：单轴定义（符号 + 范围 + 分辨率）
- ``ParamSpace``：N 维参数空间（多个 ParamAxis + 固定参数）

核心函数：
- ``lambdify_expr(expr, param_space)``：sympy 表达式 → numpy callable
- ``evaluate_on_grid(func, param_space)``：在网格上批量求值
- ``make_meshgrid(param_space)``：生成 N 维网格坐标数组

用法::

    from pysci.skills.theoretical_computation.tools.numerical import ParamAxis, ParamSpace, lambdify_expr, evaluate_on_grid

    x = sp.Symbol("x", real=True)
    y = sp.Symbol("y", real=True)
    space = ParamSpace(
        axes=[ParamAxis(x, (0, 1), 100), ParamAxis(y, (-1, 1), 80)],
        fixed={sp.Symbol("a"): 2.0},
    )
    func = lambdify_expr(x**2 + y**2 + sp.Symbol("a"), space)
    result = evaluate_on_grid(func, space)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sympy import Expr, Symbol, lambdify


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParamAxis:
    """参数空间单轴定义。

    Attributes:
        symbol: 该轴对应的 sympy 符号。
        range: 取值范围 (min, max)。
        resolution: 网格点数。
    """

    symbol: Symbol
    range: tuple[float, float]
    resolution: int = 80

    @property
    def values(self) -> NDArray[np.floating]:
        """生成该轴的等间距采样数组。"""
        return np.linspace(self.range[0], self.range[1], self.resolution)

    @property
    def length(self) -> float:
        """轴长度。"""
        return self.range[1] - self.range[0]


@dataclass(frozen=True)
class ParamSpace:
    """N 维参数空间定义。

    泛化旧 ParamSpace3D，支持任意数量的自由参数轴和固定参数。

    Attributes:
        axes: 自由参数轴列表（按维度顺序排列）。
        fixed: 固定参数字典（符号 → 数值），在 lambdify 前替换。

    Examples:
        >>> import sympy as sp
        >>> x, y, z, a = sp.symbols("x y z a", real=True)
        >>> space = ParamSpace(
        ...     axes=[ParamAxis(x, (0, 1), 50), ParamAxis(y, (0, 2), 50), ParamAxis(z, (0, 3), 50)],
        ...     fixed={a: 1.0},
        ... )
        >>> space.ndim
        3
        >>> space.symbols
        [x, y, z]
    """

    axes: tuple[ParamAxis, ...]
    fixed: dict[Symbol, Any] = field(default_factory=dict)

    def __init__(
        self,
        axes: Sequence[ParamAxis],
        fixed: dict[Symbol, Any] | None = None,
    ) -> None:
        object.__setattr__(self, "axes", tuple(axes))
        object.__setattr__(self, "fixed", fixed or {})

    @property
    def ndim(self) -> int:
        """参数空间维度。"""
        return len(self.axes)

    @property
    def symbols(self) -> list[Symbol]:
        """自由参数符号列表。"""
        return [ax.symbol for ax in self.axes]

    @property
    def resolutions(self) -> tuple[int, ...]:
        """各轴分辨率。"""
        return tuple(ax.resolution for ax in self.axes)

    @property
    def shape(self) -> tuple[int, ...]:
        """网格形状（与各轴分辨率一致）。"""
        return self.resolutions

    @property
    def grid_size(self) -> int:
        """网格总点数。"""
        return int(np.prod(self.resolutions))


# ---------------------------------------------------------------------------
# 后向兼容别名（旧代码可能引用 ParamSpace3D）
# ---------------------------------------------------------------------------
def ParamSpace3D(
    x: Symbol,
    x_range: tuple[float, float],
    y: Symbol,
    y_range: tuple[float, float],
    z: Symbol,
    z_range: tuple[float, float],
    x_resolution: int = 50,
    y_resolution: int = 50,
    z_resolution: int = 50,
    fixed_param: dict[Symbol, Any] | None = None,
) -> ParamSpace:
    """后向兼容工厂：从旧 ParamSpace3D 参数签名创建 ParamSpace。"""
    return ParamSpace(
        axes=[
            ParamAxis(x, x_range, x_resolution),
            ParamAxis(y, y_range, y_resolution),
            ParamAxis(z, z_range, z_resolution),
        ],
        fixed=fixed_param or {},
    )


# ---------------------------------------------------------------------------
# 核心函数
# ---------------------------------------------------------------------------
def lambdify_expr(
    expr: Expr,
    param_space: ParamSpace,
    modules: str | list[str] = "numpy",
) -> Callable[..., NDArray]:
    """将 sympy 表达式转换为高效的 numpy 数值函数。

    处理流程：
    1. 替换固定参数
    2. 对自由参数符号 lambdify
    3. 包装以处理常数/标量广播和复数安全

    Args:
        expr: sympy 表达式。
        param_space: 参数空间定义。
        modules: lambdify 使用的数值模块，默认 ``"numpy"``。

    Returns:
        接受 N 个数组参数（对应各轴）的 numpy 函数。
    """
    # 替换固定参数
    if param_space.fixed:
        expr = expr.subs(param_space.fixed)

    # 提取自由符号
    syms = param_space.symbols

    # lambdify
    raw_func = lambdify(syms, expr, modules=modules)

    # 包装：确保输出为数组（处理常数表达式返回标量的情况）
    def wrapped_func(*args: NDArray) -> NDArray:
        result = raw_func(*args)
        # 如果结果是标量，广播到输入形状
        if np.isscalar(result) or (isinstance(result, np.ndarray) and result.ndim == 0):
            broadcast_shape = np.broadcast_shapes(*(a.shape for a in args if hasattr(a, "shape")))
            return np.full(broadcast_shape, result, dtype=complex if np.iscomplexobj(result) else float)
        return np.asarray(result)

    return wrapped_func


def make_meshgrid(param_space: ParamSpace, indexing: str = "ij") -> tuple[NDArray, ...]:
    """生成 N 维网格坐标数组。

    Args:
        param_space: 参数空间定义。
        indexing: meshgrid 索引模式，默认 ``"ij"``（矩阵索引）。

    Returns:
        N 个形状为 param_space.shape 的坐标数组的元组。
    """
    axis_values = [ax.values for ax in param_space.axes]
    if param_space.ndim == 1:
        return (axis_values[0],)
    return tuple(np.meshgrid(*axis_values, indexing=indexing))


def evaluate_on_grid(
    func: Callable[..., NDArray],
    param_space: ParamSpace,
) -> NDArray:
    """在参数空间网格上批量求值。

    Args:
        func: 由 lambdify_expr 生成的数值函数。
        param_space: 参数空间定义。

    Returns:
        形状为 param_space.shape 的数值结果数组。
    """
    mesh = make_meshgrid(param_space)
    result = func(*mesh)
    return np.asarray(result)


def evaluate_expr_on_grid(
    expr: Expr,
    param_space: ParamSpace,
) -> NDArray:
    """一步到位：sympy 表达式 → 网格求值结果。

    等价于 ``evaluate_on_grid(lambdify_expr(expr, param_space), param_space)``。

    Args:
        expr: sympy 表达式。
        param_space: 参数空间定义。

    Returns:
        网格求值结果。
    """
    func = lambdify_expr(expr, param_space)
    return evaluate_on_grid(func, param_space)


def adaptive_sample_1d(
    func: Callable[[NDArray], NDArray],
    x_range: tuple[float, float],
    base_resolution: int = 200,
    refinement_factor: int = 4,
    gradient_threshold: float = 0.1,
) -> NDArray[np.floating]:
    """一维自适应采样：在梯度大的区域加密采样点。

    适用于检测快速变化区域（如 EP 附近、奇点附近）。

    Args:
        func: 一维数值函数。
        x_range: 采样范围。
        base_resolution: 基础均匀采样点数。
        refinement_factor: 加密倍数。
        gradient_threshold: 触发加密的归一化梯度阈值。

    Returns:
        自适应采样点数组（已排序去重）。
    """
    # 基础均匀采样
    x_base = np.linspace(x_range[0], x_range[1], base_resolution)
    y_base = func(x_base)

    # 计算归一化梯度
    if np.iscomplexobj(y_base):
        magnitude = np.abs(y_base)
    else:
        magnitude = y_base
    grad = np.abs(np.gradient(magnitude))
    grad_norm = grad / (np.max(grad) + 1e-15)

    # 找到需要加密的区域
    refine_mask = grad_norm > gradient_threshold
    x_refine: list[NDArray] = [x_base]

    if np.any(refine_mask):
        # 在梯度大的相邻点之间插入额外采样
        refine_indices = np.where(refine_mask)[0]
        for idx in refine_indices:
            if idx < base_resolution - 1:
                x_extra = np.linspace(
                    x_base[idx], x_base[idx + 1], refinement_factor, endpoint=False
                )
                x_refine.append(x_extra)

    return np.unique(np.concatenate(x_refine))
