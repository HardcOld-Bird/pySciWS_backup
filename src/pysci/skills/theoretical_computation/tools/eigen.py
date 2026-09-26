"""本征分析：符号/数值本征值-本征向量、EP 检测、简并追踪、复平面轨迹。

吸收并扩展旧 ``pysci.common.matrix.plot_2x2_matrix_eigensystem``，将计算逻辑与绘图逻辑分离，
提供面向非厄米物理（EP/BIC/CPA）的本征系统分析工具。

核心函数：
- ``eigensystem_symbolic(H)``：符号本征值/本征向量
- ``eigensystem_numeric(H_func, param_space)``：数值本征系统随参数演化
- ``detect_ep(eigenvalues, eigenvectors)``：例外点检测
- ``track_branches(eig_vals_grid)``：本征值分支追踪（避免交叉排序错误）

用法::

    from pysci.skills.theoretical_computation.tools import eigen

    eig = eigen.eigensystem_symbolic(H)
    print(eig.eigenvalues)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import sympy as sp
from numpy.typing import NDArray
from sympy import Expr, Matrix, Symbol

from . import cas


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------
@dataclass
class SymbolicEigensystem:
    """符号本征系统结果容器。

    Attributes:
        eigenvalues: 本征值列表（符号表达式）。
        eigenvectors: 本征向量列表（每个为 Matrix 列向量）。
        multiplicities: 各本征值的代数重数。
        characteristic_poly: 特征多项式。
    """

    eigenvalues: list[Expr]
    eigenvectors: list[Matrix]
    multiplicities: list[int]
    characteristic_poly: Expr | None = None


@dataclass
class NumericEigensystem:
    """数值本征系统结果容器。

    Attributes:
        eigenvalues: 本征值数组，形状 (n_params, n_eigenvalues)，复数。
        eigenvectors: 本征向量数组，形状 (n_params, dim, n_eigenvalues)，复数。
        param_values: 参数值数组。
    """

    eigenvalues: NDArray[np.complexfloating]
    eigenvectors: NDArray[np.complexfloating]
    param_values: NDArray[np.floating]


@dataclass
class EPDetection:
    """例外点检测结果。

    Attributes:
        found: 是否检测到 EP。
        param_location: EP 所在的参数值（若找到）。
        eigenvalue: EP 处的本征值。
        eigenvector: EP 处合并的本征向量。
        coalescence_measure: 本征向量合并度（0=正交，1=完全合并）。
    """

    found: bool
    param_location: Any = None
    eigenvalue: complex | None = None
    eigenvector: NDArray | None = None
    coalescence_measure: float | None = None


# ---------------------------------------------------------------------------
# 符号本征分析
# ---------------------------------------------------------------------------
def eigensystem_symbolic(H: Matrix) -> SymbolicEigensystem:
    """计算矩阵的符号本征值和本征向量。

    Args:
        H: sympy Matrix（哈密顿量）。

    Returns:
        SymbolicEigensystem 容器。
    """
    eigenvects = H.eigenvects()

    eigenvalues = []
    eigenvectors = []
    multiplicities = []

    for val, mult, vecs in eigenvects:
        eigenvalues.append(cas.simplify_expr(val))
        multiplicities.append(mult)
        for v in vecs:
            eigenvectors.append(v)

    # 特征多项式（可选，对大矩阵可能很慢）
    char_poly = None
    if H.shape[0] <= 4:
        try:
            lam = Symbol("lambda")
            char_poly = sp.expand((H - lam * Matrix.eye(H.rows)).det())
        except Exception:  # noqa: BLE001
            pass

    return SymbolicEigensystem(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        multiplicities=multiplicities,
        characteristic_poly=char_poly,
    )


def ep_condition_symbolic(H: Matrix) -> Expr:
    """计算 2x2 矩阵的 EP 条件表达式（判别式 = 0）。

    EP 条件：两个本征值和本征向量同时合并，即判别式 (a-d)^2 + 4bc = 0。

    Args:
        H: 2x2 sympy Matrix。

    Returns:
        EP 条件表达式（令其 = 0 即为 EP）。
    """
    return cas.discriminant_2x2(H)


# ---------------------------------------------------------------------------
# 数值本征分析
# ---------------------------------------------------------------------------
def eigensystem_numeric(
    H_func: Callable[..., NDArray],
    param_values: NDArray[np.floating],
    dim: int = 2,
) -> NumericEigensystem:
    """计算数值本征系统随单参数演化。

    对每个参数值构建矩阵并求本征值/本征向量。

    Args:
        H_func: 接受参数值返回 dim x dim 复矩阵的函数。
        param_values: 参数值数组。
        dim: 矩阵维度。

    Returns:
        NumericEigensystem 容器。
    """
    n_params = len(param_values)
    eigenvalues = np.zeros((n_params, dim), dtype=complex)
    eigenvectors = np.zeros((n_params, dim, dim), dtype=complex)

    for i, p in enumerate(param_values):
        H = np.array(H_func(p), dtype=complex)
        eigvals, eigvecs = np.linalg.eig(H)
        eigenvalues[i] = eigvals
        eigenvectors[i] = eigvecs

    return NumericEigensystem(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        param_values=param_values,
    )


def eigensystem_numeric_2d(
    H_func: Callable[..., NDArray],
    param_x: NDArray[np.floating],
    param_y: NDArray[np.floating],
    dim: int = 2,
) -> dict[str, NDArray]:
    """计算数值本征系统在二维参数空间中的分布。

    Args:
        H_func: 接受两个参数值返回 dim x dim 复矩阵的函数。
        param_x: 第一参数值数组。
        param_y: 第二参数值数组。
        dim: 矩阵维度。

    Returns:
        字典，含 eigenvalues (nx, ny, dim) 和 eigenvectors (nx, ny, dim, dim)。
    """
    nx, ny = len(param_x), len(param_y)
    eigenvalues = np.zeros((nx, ny, dim), dtype=complex)
    eigenvectors = np.zeros((nx, ny, dim, dim), dtype=complex)

    for i, px in enumerate(param_x):
        for j, py in enumerate(param_y):
            H = np.array(H_func(px, py), dtype=complex)
            eigvals, eigvecs = np.linalg.eig(H)
            eigenvalues[i, j] = eigvals
            eigenvectors[i, j] = eigvecs

    return {"eigenvalues": eigenvalues, "eigenvectors": eigenvectors}


# ---------------------------------------------------------------------------
# EP 检测
# ---------------------------------------------------------------------------
def detect_ep(
    eigensystem: NumericEigensystem,
    tol_eigenvalue: float = 1e-6,
    tol_eigenvector: float = 0.99,
) -> list[EPDetection]:
    """在数值本征系统序列中检测例外点。

    EP 判据：
    1. 两个本征值之间的距离 < tol_eigenvalue
    2. 对应本征向量的归一化内积模 > tol_eigenvector（接近 1 = 完全合并）

    Args:
        eigensystem: NumericEigensystem 结果。
        tol_eigenvalue: 本征值合并容差。
        tol_eigenvector: 本征向量合并阈值。

    Returns:
        检测到的 EP 列表。
    """
    detections: list[EPDetection] = []
    eigvals = eigensystem.eigenvalues
    eigvecs = eigensystem.eigenvectors
    params = eigensystem.param_values
    dim = eigvals.shape[1]

    for i in range(len(params)):
        for a in range(dim):
            for b in range(a + 1, dim):
                # 本征值距离
                val_dist = abs(eigvals[i, a] - eigvals[i, b])
                if val_dist > tol_eigenvalue:
                    continue

                # 本征向量合并度
                v_a = eigvecs[i, :, a]
                v_b = eigvecs[i, :, b]
                v_a_norm = v_a / (np.linalg.norm(v_a) + 1e-15)
                v_b_norm = v_b / (np.linalg.norm(v_b) + 1e-15)
                coalescence = abs(np.vdot(v_a_norm, v_b_norm))

                if coalescence > tol_eigenvector:
                    detections.append(EPDetection(
                        found=True,
                        param_location=params[i],
                        eigenvalue=complex(eigvals[i, a]),
                        eigenvector=v_a_norm,
                        coalescence_measure=float(coalescence),
                    ))

    return detections


def eigenvector_coalescence(
    eigvecs: NDArray[np.complexfloating],
    idx_a: int = 0,
    idx_b: int = 1,
) -> float:
    """计算两个本征向量的合并度（归一化内积的模）。

    Args:
        eigvecs: 本征向量矩阵 (dim, n_eigenvalues)，列为本征向量。
        idx_a: 第一个本征向量索引。
        idx_b: 第二个本征向量索引。

    Returns:
        合并度（0=正交，1=完全合并）。
    """
    v_a = eigvecs[:, idx_a]
    v_b = eigvecs[:, idx_b]
    v_a_norm = v_a / (np.linalg.norm(v_a) + 1e-15)
    v_b_norm = v_b / (np.linalg.norm(v_b) + 1e-15)
    return float(abs(np.vdot(v_a_norm, v_b_norm)))


# ---------------------------------------------------------------------------
# 本征值分支追踪
# ---------------------------------------------------------------------------
def track_branches(
    eig_vals_grid: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    """追踪本征值分支，避免交叉时的排序错误。

    使用最近邻匹配：对每个参数步，将本征值按与上一步的连续性重新排列。

    Args:
        eig_vals_grid: 形状 (n_params, n_eigenvalues) 的本征值数组。

    Returns:
        分支追踪后的本征值数组（同形状）。
    """
    n_params, n_eig = eig_vals_grid.shape
    tracked = np.zeros_like(eig_vals_grid)
    tracked[0] = eig_vals_grid[0]

    for i in range(1, n_params):
        # 计算当前步与上一步已追踪值的距离矩阵
        dist_matrix = np.abs(
            eig_vals_grid[i][:, np.newaxis] - tracked[i - 1][np.newaxis, :]
        )
        # 贪心最近邻匹配
        used = set()
        for j in range(n_eig):
            # 找到距离 tracked[i-1, j] 最近的当前本征值
            dists = dist_matrix[:, j].copy()
            for u in used:
                dists[u] = np.inf
            best = int(np.argmin(dists))
            tracked[i, j] = eig_vals_grid[i, best]
            used.add(best)

    return tracked


def sort_eigenvalues_real(
    eig_vals: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    """按实部排序本征值（用于能带图）。

    Args:
        eig_vals: 形状 (..., n_eigenvalues) 的本征值数组。

    Returns:
        排序后的本征值数组。
    """
    return np.sort(eig_vals.real, axis=-1) + 1j * np.take_along_axis(
        eig_vals.imag, np.argsort(eig_vals.real, axis=-1), axis=-1
    )
