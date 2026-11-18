"""符号结果 → LaTeX 导出归一化（衔接⑤域排版链）。

ADR（自研理由，2026-11-17）：SymPy 自带 ``sympy.printing.latex`` 已覆盖全部打印需求，
本模块只做归一化薄胶水：①显式固定打印机设置（输出确定性，机械断言可钉死精确字符串）；
②空白折叠/去首尾（LaTeX 数学模式内无语义损失）；③矩阵→amsmath 矩阵环境（⑤域约定：
LaTeX 原生 amsmath、MD 写 ``$...$`` 走 pandoc→OMML）。不做格式策略路由（旧项目
strategy 路由器已按裁决砍除）。输出**不含** ``$`` 定界符，由⑤域模板决定定界形式。

机械断言：``tests/skills/theory_calculation/test_latex_export.py``。
"""

from __future__ import annotations

import re

import sympy as sp

#: 受支持的 amsmath 矩阵环境（matrix 为无定界裸环境）。
_ENVS = frozenset({"matrix", "pmatrix", "bmatrix", "vmatrix", "Vmatrix"})
_SPACE_RE = re.compile(r"[ \t]{2,}")


def _normalize(text: str) -> str:
    """归一化打印输出：折叠连续空白、去首尾空白（数学模式内无语义损失）。"""
    return _SPACE_RE.sub(" ", text).strip()


def expr_to_latex(expr: object, *, fold_short_frac: bool = True) -> str:
    """把 SymPy 表达式打印为归一化 LaTeX 串。

    Args:
        expr: 任意 sympy 打印机可接受的表达式。
        fold_short_frac: 短分式折叠为 ``x / y``（⑤域行内公式的偏好写法）；
            长分式（分子/分母含运算）不受影响，仍打印 ``\\frac{...}{...}``。

    Returns:
        归一化 LaTeX 串（不含 ``$`` 定界符）。
    """
    raw = sp.latex(expr, fold_short_frac=fold_short_frac, imaginary_unit="i")
    return _normalize(raw)


def matrix_to_latex(matrix: object, *, env: str = "pmatrix") -> str:
    """把 SymPy 矩阵打印为指定 amsmath 矩阵环境（剥掉 ``\\left[...\\right]`` 外壳）。

    打印结果中所有 ``matrix`` 环境（含分块矩阵的嵌套层）统一替换为 ``env``。

    Args:
        matrix: SymPy 矩阵（Matrix / ImmutableMatrix 等）。
        env: 目标 amsmath 矩阵环境名（见 :data:`_ENVS`）。

    Returns:
        形如 ``\\begin{pmatrix}1 & 2\\\\3 & 4\\end{pmatrix}`` 的环境串。

    Raises:
        ValueError: ``env`` 不受支持，或对象不是标准定界（``[...]``）的矩阵。
    """
    if env not in _ENVS:
        raise ValueError(f"不支持的矩阵环境：{env!r}（可选：{sorted(_ENVS)}）")
    raw = sp.latex(matrix)
    left, right = r"\left[", r"\right]"
    if not (raw.startswith(left) and raw.endswith(right)):
        raise ValueError(f"对象不是标准定界的矩阵：{raw!r}")
    body = raw[len(left) : -len(right)]
    body = body.replace(r"\begin{matrix}", rf"\begin{{{env}}}")
    body = body.replace(r"\end{matrix}", rf"\end{{{env}}}")
    return _normalize(body)
