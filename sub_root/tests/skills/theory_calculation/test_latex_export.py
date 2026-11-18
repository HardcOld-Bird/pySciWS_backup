"""latex_export 固定样例：打印机输出归一化与确定性（精确字符串断言）。

期望值取自 pin 版 sympy 1.14.0 的 latex 打印机实测输出（uv.lock 锁定；
打印器设置被胶水显式固定，输出确定性是本模块的合同）。
"""

from __future__ import annotations

import pytest
import sympy as sp

from pysci.skills.theory_calculation import latex_export

x, y = sp.symbols("x y")


def test_expr_to_latex_fixed_samples():
    assert latex_export.expr_to_latex(x**2) == "x^{2}"
    assert latex_export.expr_to_latex((x + 1) / (y + 3)) == "\\frac{x + 1}{y + 3}"
    assert latex_export.expr_to_latex(sp.sqrt(x**2 + 1)) == "\\sqrt{x^{2} + 1}"
    assert (
        latex_export.expr_to_latex(sp.sin(x) + sp.exp(-x))
        == "\\sin{\\left(x \\right)} + e^{- x}"
    )


def test_expr_to_latex_fold_short_frac_contract():
    assert latex_export.expr_to_latex(x / y) == "x / y"  # 短分式折叠
    assert latex_export.expr_to_latex(x / y, fold_short_frac=False) == "\\frac{x}{y}"
    # 长分式不受折叠影响
    assert (
        latex_export.expr_to_latex((x + 1) / (y + 3))
        == "\\frac{x + 1}{y + 3}"
    )


def test_expr_to_latex_is_deterministic_and_normalized():
    expr = sp.sin(x) * sp.exp(-x) / (y + 3)
    first = latex_export.expr_to_latex(expr)
    assert first == latex_export.expr_to_latex(expr)  # 两次调用逐字相同
    assert first == first.strip()
    assert "  " not in first


def test_matrix_to_latex_pmatrix():
    m = sp.Matrix([[1, 2], [3, 4]])
    assert (
        latex_export.matrix_to_latex(m)
        == "\\begin{pmatrix}1 & 2\\\\3 & 4\\end{pmatrix}"
    )
    # 列向量与自定义环境
    assert (
        latex_export.matrix_to_latex(sp.Matrix([x, y]))
        == "\\begin{pmatrix}x\\\\y\\end{pmatrix}"
    )
    assert (
        latex_export.matrix_to_latex(m, env="bmatrix")
        == "\\begin{bmatrix}1 & 2\\\\3 & 4\\end{bmatrix}"
    )


def test_matrix_to_latex_rejects_non_matrix_and_bad_env():
    with pytest.raises(ValueError):
        latex_export.matrix_to_latex(x + 1)  # 标量不是矩阵
    with pytest.raises(ValueError):
        latex_export.matrix_to_latex(sp.Matrix([[1]]), env="quatrix")  # 环境不支持
