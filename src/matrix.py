"""矩阵本征值和本征向量计算与可视化工具"""

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from sympy import Expr, Matrix, lambdify


def plot_2x2_matrix_eigensystem(
    m11: Expr,
    m12: Expr,
    m21: Expr,
    m22: Expr,
    param_symbol: Expr,
    param_range: tuple[float, float],
    param_resolution: int = 200,
    figsize: tuple[float, float] = (14, 10),
) -> Figure:
    """计算并绘制2阶复矩阵的本征值和本征向量随参数变化的图像。

    该函数接收一个2x2矩阵的四个元素（可以是符号表达式），计算其本征值和本征向量
    随指定参数变化的情况，并绘制以下内容：
    - 本征值的实部和虚部
    - 本征向量的分量（实部和虚部）

    Args:
        m11: 矩阵左上角元素（第1行第1列）。
        m12: 矩阵右上角元素（第1行第2列）。
        m21: 矩阵左下角元素（第2行第1列）。
        m22: 矩阵右下角元素（第2行第2列）。
        param_symbol: 参数符号（如 sympy.Symbol('x')）。
        param_range: 参数的取值范围 (最小值, 最大值)。
        param_resolution: 参数的分辨率（采样点数），默认为 200。
        figsize: 图像尺寸 (宽, 高)，默认为 (14, 10)。

    Returns:
        matplotlib Figure对象，包含本征值和本征向量的可视化。

    Examples:
        >>> from sympy import symbols, I
        >>> x = symbols('x', real=True)
        >>> # 定义一个简单的2x2矩阵
        >>> m11 = 1 + 0.1*x
        >>> m12 = 0.5
        >>> m21 = 0.5
        >>> m22 = 2 - 0.1*x
        >>> fig = plot_2x2_matrix_eigensystem(
        ...     m11, m12, m21, m22,
        ...     param_symbol=x,
        ...     param_range=(0, 10)
        ... )
        >>> plt.show()
    """
    # 构建2x2矩阵
    matrix = Matrix([[m11, m12], [m21, m22]])

    # 计算本征值和本征向量（符号形式）
    eigenvects = matrix.eigenvects()

    # 创建参数数组
    param_vals: NDArray[np.floating] = np.linspace(
        param_range[0], param_range[1], param_resolution
    )

    # 初始化存储数组
    eigenval1_vals: NDArray[np.complexfloating] = np.zeros(
        param_resolution, dtype=complex
    )
    eigenval2_vals: NDArray[np.complexfloating] = np.zeros(
        param_resolution, dtype=complex
    )
    eigenvec1_comp1: NDArray[np.complexfloating] = np.zeros(
        param_resolution, dtype=complex
    )
    eigenvec1_comp2: NDArray[np.complexfloating] = np.zeros(
        param_resolution, dtype=complex
    )
    eigenvec2_comp1: NDArray[np.complexfloating] = np.zeros(
        param_resolution, dtype=complex
    )
    eigenvec2_comp2: NDArray[np.complexfloating] = np.zeros(
        param_resolution, dtype=complex
    )

    # 将本征值和本征向量转换为numpy函数
    eigenval1_func: Callable = lambdify(param_symbol, eigenvects[0][0], modules="numpy")
    eigenval2_func: Callable = lambdify(param_symbol, eigenvects[1][0], modules="numpy")

    eigenvec1_func_comp1: Callable = lambdify(
        param_symbol, eigenvects[0][2][0][0], modules="numpy"
    )
    eigenvec1_func_comp2: Callable = lambdify(
        param_symbol, eigenvects[0][2][0][1], modules="numpy"
    )
    eigenvec2_func_comp1: Callable = lambdify(
        param_symbol, eigenvects[1][2][0][0], modules="numpy"
    )
    eigenvec2_func_comp2: Callable = lambdify(
        param_symbol, eigenvects[1][2][0][1], modules="numpy"
    )

    # 计算数值
    # 注意：如果表达式是常数，lambdify返回标量，需要广播到数组
    eigenval1_vals_raw = eigenval1_func(param_vals)
    eigenval2_vals_raw = eigenval2_func(param_vals)
    eigenvec1_comp1_raw = eigenvec1_func_comp1(param_vals)
    eigenvec1_comp2_raw = eigenvec1_func_comp2(param_vals)
    eigenvec2_comp1_raw = eigenvec2_func_comp1(param_vals)
    eigenvec2_comp2_raw = eigenvec2_func_comp2(param_vals)

    # 确保结果是数组（处理常数情况）
    eigenval1_vals = np.atleast_1d(eigenval1_vals_raw) * np.ones_like(param_vals)
    eigenval2_vals = np.atleast_1d(eigenval2_vals_raw) * np.ones_like(param_vals)
    eigenvec1_comp1 = np.atleast_1d(eigenvec1_comp1_raw) * np.ones_like(param_vals)
    eigenvec1_comp2 = np.atleast_1d(eigenvec1_comp2_raw) * np.ones_like(param_vals)
    eigenvec2_comp1 = np.atleast_1d(eigenvec2_comp1_raw) * np.ones_like(param_vals)
    eigenvec2_comp2 = np.atleast_1d(eigenvec2_comp2_raw) * np.ones_like(param_vals)

    # 创建图像
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # 绘制本征值1的实部和虚部
    axes[0, 0].plot(param_vals, np.real(eigenval1_vals), "b-", linewidth=2)
    axes[0, 0].set_xlabel(f"{param_symbol}", fontsize=12)
    axes[0, 0].set_ylabel("Re(λ₁)", fontsize=12)
    axes[0, 0].set_title("本征值1 - 实部", fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(param_vals, np.imag(eigenval1_vals), "r-", linewidth=2)
    axes[0, 1].set_xlabel(f"{param_symbol}", fontsize=12)
    axes[0, 1].set_ylabel("Im(λ₁)", fontsize=12)
    axes[0, 1].set_title("本征值1 - 虚部", fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)

    # 绘制本征值2的实部和虚部
    axes[1, 0].plot(param_vals, np.real(eigenval2_vals), "b-", linewidth=2)
    axes[1, 0].set_xlabel(f"{param_symbol}", fontsize=12)
    axes[1, 0].set_ylabel("Re(λ₂)", fontsize=12)
    axes[1, 0].set_title("本征值2 - 实部", fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(param_vals, np.imag(eigenval2_vals), "r-", linewidth=2)
    axes[1, 1].set_xlabel(f"{param_symbol}", fontsize=12)
    axes[1, 1].set_ylabel("Im(λ₂)", fontsize=12)
    axes[1, 1].set_title("本征值2 - 虚部", fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)

    # 绘制本征向量1的两个分量
    axes[0, 2].plot(param_vals, np.real(eigenvec1_comp1), "b-", linewidth=2, label="Re")
    axes[0, 2].plot(param_vals, np.imag(eigenvec1_comp1), "r--", linewidth=2, label="Im")
    axes[0, 2].set_xlabel(f"{param_symbol}", fontsize=12)
    axes[0, 2].set_ylabel("v₁[0]", fontsize=12)
    axes[0, 2].set_title("本征向量1 - 第1分量", fontsize=14)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 绘制本征向量2的两个分量
    axes[1, 2].plot(param_vals, np.real(eigenvec2_comp1), "b-", linewidth=2, label="Re")
    axes[1, 2].plot(param_vals, np.imag(eigenvec2_comp1), "r--", linewidth=2, label="Im")
    axes[1, 2].set_xlabel(f"{param_symbol}", fontsize=12)
    axes[1, 2].set_ylabel("v₂[0]", fontsize=12)
    axes[1, 2].set_title("本征向量2 - 第1分量", fontsize=14)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    return fig
