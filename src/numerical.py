"""数值计算常用的工具函数"""

from collections.abc import Callable

from sympy import Expr, Symbol, lambdify

from src.my_dtypes import ParamSpace3D


def get_numpy_func(expr: Expr, param_space: ParamSpace3D) -> Callable:
    """将 sympy 表达式转换为高效的numpy数值函数。

    Args:
        expr: sympy 表达式。
        param_space: 三维参数空间定义，包含参数符号、范围和分辨率。

    Returns:
        numpy数值函数。
    """
    # 替换固定参数
    expr_substituted: Expr = expr.subs(param_space.fixed_param)

    # 提取参数符号列表
    params: list[Symbol] = [param_space.x, param_space.y, param_space.z]

    # 使用 lambdify 将 sympy 表达式转换为高效的数值函数
    numpy_func = lambdify(params, expr_substituted, modules=["numpy"])

    return numpy_func
