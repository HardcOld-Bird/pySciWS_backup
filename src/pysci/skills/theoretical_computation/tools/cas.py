"""CAS 核心引擎：符号表达式构建/化简/替换/级数展开/LaTeX 输出/方程组求解。

封装 sympy 常用操作，提供更健壮的接口（自动处理多解、条件解、复数域等边界情况），
并面向物理声学场景提供哈密顿量/传递矩阵构建辅助。

用法::

    from pysci.skills.theoretical_computation.tools import cas

    H = cas.symbolic_matrix([[omega1, kappa], [kappa, omega2]])
    simplified = cas.simplify_expr(expr)
    latex_str = cas.expr_to_latex(simplified)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import sympy as sp
from sympy import Expr, Matrix, Symbol


# ---------------------------------------------------------------------------
# 矩阵构建
# ---------------------------------------------------------------------------
def symbolic_matrix(elements: Sequence[Sequence[Expr]], hermitian: bool = False) -> Matrix:
    """构建 sympy Matrix，可选强制厄米对称。

    Args:
        elements: 嵌套列表，matrix[i][j] 为第 i 行第 j 列元素。
        hermitian: 若为 True，自动将下三角设为上三角的共轭转置。

    Returns:
        sympy Matrix 对象。

    Examples:
        >>> import sympy as sp
        >>> w, k = sp.symbols("omega kappa")
        >>> H = symbolic_matrix([[w, k], [k, -w]])
        >>> H.shape
        (2, 2)
    """
    M = Matrix(elements)
    if hermitian:
        n = M.rows
        for i in range(n):
            for j in range(i):
                M[j, i] = sp.conjugate(M[i, j])
    return M


def transfer_matrix_chain(*matrices: Matrix) -> Matrix:
    """计算传递矩阵链的乘积（从左到右）。

    Args:
        *matrices: 按顺序排列的传递矩阵。

    Returns:
        总传递矩阵 M_total = M_1 * M_2 * ... * M_n。
    """
    if not matrices:
        raise ValueError("至少需要一个矩阵")
    result = matrices[0]
    for M in matrices[1:]:
        result = result * M
    return result


# ---------------------------------------------------------------------------
# 化简
# ---------------------------------------------------------------------------
def simplify_expr(expr: Expr, strategy: str = "auto") -> Expr:
    """智能化简 sympy 表达式。

    Args:
        expr: 待化简表达式。
        strategy: 化简策略：
            - ``"auto"``：依次尝试 trigsimp/radsimp/simplify，选最短结果。
            - ``"trig"``：仅三角化简。
            - ``"radical"``：仅根式化简。
            - ``"full"``：直接 sp.simplify。
            - ``"expand"``：展开。
            - ``"factor"``：因式分解。
            - ``"collect"``：按符号收集同类项。
            - ``"cancel"``：通分约分。

    Returns:
        化简后的表达式。
    """
    if strategy == "trig":
        return sp.trigsimp(expr)
    elif strategy == "radical":
        return sp.radsimp(expr)
    elif strategy == "full":
        return sp.simplify(expr)
    elif strategy == "expand":
        return sp.expand(expr)
    elif strategy == "factor":
        return sp.factor(expr)
    elif strategy == "collect":
        return sp.collect(expr, expr.free_symbols)
    elif strategy == "cancel":
        return sp.cancel(expr)
    else:
        # auto：尝试多种策略，选最短的
        candidates = [expr]
        for fn in (sp.trigsimp, sp.radsimp, sp.simplify, sp.expand, sp.cancel):
            try:
                result = fn(expr)
                candidates.append(result)
            except Exception:  # noqa: BLE001
                pass
        # 选择字符串表示最短的（通常是化简最好的）
        return min(candidates, key=lambda e: len(str(e)))


def simplify_matrix(M: Matrix, strategy: str = "auto") -> Matrix:
    """逐元素化简矩阵。

    Args:
        M: sympy Matrix。
        strategy: 化简策略（同 simplify_expr）。

    Returns:
        化简后的矩阵。
    """
    return M.applyfunc(lambda e: simplify_expr(e, strategy))


# ---------------------------------------------------------------------------
# 替换
# ---------------------------------------------------------------------------
def substitute_params(expr: Expr, param_dict: dict[Symbol, Any]) -> Expr:
    """批量参数替换。

    Args:
        expr: sympy 表达式。
        param_dict: 替换字典（符号 → 值/表达式）。

    Returns:
        替换后的表达式。
    """
    return expr.subs(param_dict)


def substitute_matrix(M: Matrix, param_dict: dict[Symbol, Any]) -> Matrix:
    """矩阵批量参数替换。

    Args:
        M: sympy Matrix。
        param_dict: 替换字典。

    Returns:
        替换后的矩阵。
    """
    return M.subs(param_dict)


# ---------------------------------------------------------------------------
# 级数展开
# ---------------------------------------------------------------------------
def series_expand(
    expr: Expr,
    var: Symbol,
    point: Any = 0,
    order: int = 6,
) -> Expr:
    """泰勒/洛朗级数展开。

    Args:
        expr: 待展开表达式。
        var: 展开变量。
        point: 展开点，默认 0。
        order: 展开阶数。

    Returns:
        级数展开结果（含 O 项）。
    """
    return sp.series(expr, var, point, order)


def taylor_coefficients(
    expr: Expr,
    var: Symbol,
    point: Any = 0,
    order: int = 6,
) -> list[Expr]:
    """提取泰勒展开系数列表。

    Args:
        expr: 待展开表达式。
        var: 展开变量。
        point: 展开点。
        order: 展开阶数。

    Returns:
        系数列表 [a_0, a_1, ..., a_{order-1}]，其中 expr ≈ sum(a_n * (var-point)^n)。
    """
    series = sp.series(expr, var, point, order).removeO()
    expanded = sp.expand(series)
    coeffs = []
    for n in range(order):
        c = expanded.coeff(var, n)
        if point != 0:
            # 需要手动处理非零展开点
            c = sp.diff(expr, var, n).subs(var, point) / sp.factorial(n)
        coeffs.append(sp.simplify(c))
    return coeffs


# ---------------------------------------------------------------------------
# LaTeX 输出
# ---------------------------------------------------------------------------
def expr_to_latex(expr: Expr, fold_short_frac: bool = True) -> str:
    """将 sympy 表达式转为 LaTeX 字符串。

    Args:
        expr: sympy 表达式。
        fold_short_frac: 是否折叠短分式。

    Returns:
        LaTeX 字符串（不含 $ 包裹）。
    """
    return sp.latex(expr, fold_short_frac=fold_short_frac)


def matrix_to_latex(M: Matrix, environment: str = "pmatrix") -> str:
    """将 sympy Matrix 转为 LaTeX 字符串。

    Args:
        M: sympy Matrix。
        environment: LaTeX 矩阵环境名。

    Returns:
        LaTeX 字符串。
    """
    return sp.latex(M, mat_env=environment)


# ---------------------------------------------------------------------------
# 方程组求解
# ---------------------------------------------------------------------------
def solve_system(
    equations: Sequence[sp.Eq | Expr],
    variables: Sequence[Symbol],
    dict_result: bool = True,
    simplify: bool = True,
) -> list[dict[Symbol, Expr]] | list[tuple[Expr, ...]]:
    """方程组求解封装。

    自动处理：
    - 将 Eq 转为 expr = lhs - rhs 的形式
    - 多解情况返回所有解
    - 可选化简

    Args:
        equations: 方程列表（sp.Eq 或表达式，后者视为 expr=0）。
        variables: 待求解变量。
        dict_result: True 返回字典列表，False 返回元组列表。
        simplify: 是否化简结果。

    Returns:
        解列表。
    """
    # 统一为 expr = 0 的形式
    exprs = []
    for eq in equations:
        if isinstance(eq, sp.Eq):
            exprs.append(eq.lhs - eq.rhs)
        else:
            exprs.append(eq)

    solutions = sp.solve(exprs, list(variables), dict=dict_result)

    if simplify and dict_result:
        solutions = [
            {k: sp.simplify(v) for k, v in sol.items()}
            for sol in solutions
        ]

    return solutions


def solve_condition(
    expr: Expr,
    var: Symbol,
    condition: str = "zero",
) -> list[Expr]:
    """求解单变量条件（零点、极点等）。

    Args:
        expr: 表达式。
        var: 求解变量。
        condition: 条件类型：
            - ``"zero"``：求 expr = 0 的解。
            - ``"real"``：求使 expr 为实数的条件。
            - ``"imaginary"``：求使 expr 为纯虚数的条件。

    Returns:
        解列表。
    """
    if condition == "zero":
        return sp.solve(expr, var)
    elif condition == "real":
        return sp.solve(sp.im(expr), var)
    elif condition == "imaginary":
        return sp.solve(sp.re(expr), var)
    else:
        raise ValueError(f"不支持的条件类型: {condition}")


# ---------------------------------------------------------------------------
# 实用辅助
# ---------------------------------------------------------------------------
def discriminant_2x2(H: Matrix) -> Expr:
    """计算 2x2 矩阵的判别式（本征值简并条件）。

    对于 H = [[a, b], [c, d]]，判别式为 (a-d)^2 + 4bc。
    判别式 = 0 时本征值简并（EP 条件）。

    Args:
        H: 2x2 sympy Matrix。

    Returns:
        判别式表达式。
    """
    if H.shape != (2, 2):
        raise ValueError(f"需要 2x2 矩阵，得到 {H.shape}")
    a, b, c, d = H[0, 0], H[0, 1], H[1, 0], H[1, 1]
    return sp.expand((a - d) ** 2 + 4 * b * c)


def eigenvalue_splitting(H: Matrix) -> Expr:
    """计算 2x2 矩阵的本征值劈裂量 Lambda。

    Lambda = sqrt((a-d)^2/4 + bc)，即判别式的平方根的一半。

    Args:
        H: 2x2 sympy Matrix。

    Returns:
        劈裂量表达式。
    """
    disc = discriminant_2x2(H)
    return sp.sqrt(disc) / 2


def pretty_print(expr: Expr | Matrix, label: str = "") -> None:
    """美观打印 sympy 表达式（Unicode 排版）。

    Args:
        expr: 表达式或矩阵。
        label: 可选标签前缀。
    """
    if label:
        print(f"{label}:")
    sp.pprint(expr)
    print()
