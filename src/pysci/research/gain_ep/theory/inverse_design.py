"""CMT 管槽超表面逆向设计管线（目标 S 规格 + 差分进化 + 断点续算）。

目标函数（可组合、可加权，统一签名 ``term.evaluate(S, ports) -> float``）：
- :class:`MatrixMatch`：任意复目标矩阵匹配；可分别对复误差、幅值误差、相位误差
  加权（相位项用 ``1−cosΔφ``，无 2π 跳变，且 |T|=0 的元素自动跳过相位项）。
- :class:`ZeroElements`：指定矩阵元置零（单向隐身等）。
- :class:`JordanBlock`：约当块判据。2×2 时 ``|tr²−4det|`` 与 ``|λ₁−λ₂|²`` 恒等；
  一般维度取本征值最小两两间隙²（均按 S 的 RMS 幅度归一化）。本征值合并不等于
  约当块（简并但可对角化的 DP 也满足），因此另提供 :func:`jordan_structure`
  诊断：对本征值簇比较代数重数与几何重数（``P − rank(S−λI)``），给出是否
  defective。优化目标用平滑的本征值间隙作为文档「约当块形式」的可行代理，
  最终解可用该诊断复核。
- :class:`CombinedObjective`：按各 term 自身的 ``weight`` 求和。

几何约束（:class:`DesignSpace`）：管槽总宽 + 硬壁 ≥ 周期、槽不重叠、最小壁宽
``min_feature``、逐槽尺寸/介质上下界。解码时采用**修复策略**（按超额比例压缩槽宽、
逐槽夹紧位置），保证每个个体都可行；公开的 ``decode`` 仍走
``MetagratingParams`` 的完整校验，非法编码会抛 ``MetagratingParameterError``。

优化器：自研 DE/rand/1/bin（或 best/1），种群级评估用 ``ProcessPoolExecutor``
多进程并行（Windows spawn 安全）；每代落检查点（种群、适应度、收敛史、RNG 状态），
``resume=True`` 可精确定点续算。

用法::

    from pysci.research.gain_ep.theory.cmt_reflection_s_matrix import (
        compute_s_matrix, make_ep_params)
    from pysci.research.gain_ep.theory.inverse_design import (
        CombinedObjective, DesignSpace, MatrixMatch, inverse_design)

    truth = make_ep_params()
    space = DesignSpace(J=3, period=truth.period, theta_i=truth.theta_i)
    target = compute_s_matrix(truth)
    pop = space.perturbed_population(truth, scale=0.01, popsize=24, seed=0)
    res = inverse_design(
        space, CombinedObjective((MatrixMatch(target),)),
        popsize=24, max_generations=60, initial_population=pop, seed=0)
    print(res.objective_value, np.max(np.abs(res.S - target)))
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from pysci.paths import DATA_ROOT, assert_within_data, research_theory_dir
from pysci.research.gain_ep.theory.cmt_reflection_s_matrix import (
    MetagratingParameterError,
    MetagratingParams,
    Port,
    compute_s_matrix,
    solve_s_matrix,
)

__all__ = [
    "CombinedObjective",
    "DesignSpace",
    "Evaluator",
    "InverseResult",
    "JordanBlock",
    "JordanStructure",
    "MatrixMatch",
    "ZeroElements",
    "inverse_design",
    "jordan_structure",
]

LAM = 343.0 / 3430.0  # 名义波长 0.1 m
C0 = 343.0
D_DEFAULT = LAM / np.sqrt(2.0)


# ===========================================================================
# 目标函数
# ===========================================================================


class ObjectiveTerm(Protocol):
    """目标项协议：``evaluate(S, ports) -> float``。"""

    weight: float

    def evaluate(self, S: NDArray, ports: tuple[Port, ...]) -> float: ...


@dataclass(frozen=True)
class MatrixMatch:
    """复目标矩阵的加权最小二乘匹配。

    ``complex_weight`` 乘以 ``mean(w·|S−T|²)``；``amplitude_weight`` 乘以
    ``mean(w·(|S|−|T|)²)``；``phase_weight`` 乘以 ``mean(w·(1−cosΔφ))``
    （仅统计 |T|>0 的元素，Δφ = arg(S·conj(T))）。所有权重按元素权重之和归一化。

    Args:
        target: 目标 S 矩阵 (P, P)，复。
        weight: 整个 term 的权重。
        element_weights: 可选 (P, P) 非负逐元权重。
        complex_weight: 复误差权重。
        amplitude_weight: 幅值误差权重。
        phase_weight: 相位误差权重。
    """

    target: NDArray
    weight: float = 1.0
    element_weights: NDArray | None = None
    complex_weight: float = 1.0
    amplitude_weight: float = 0.0
    phase_weight: float = 0.0

    def evaluate(self, S: NDArray, ports: tuple[Port, ...]) -> float:
        S = np.asarray(S, dtype=complex)
        T = np.asarray(self.target, dtype=complex)
        if S.shape != T.shape:
            raise MetagratingParameterError(
                f"目标 S 形状 {T.shape} 与候选 S 形状 {S.shape} 不一致"
            )
        if self.element_weights is None:
            w = np.ones_like(T, dtype=float)
        else:
            w = np.asarray(self.element_weights, dtype=float)
            if w.shape != T.shape or np.any(w < 0):
                raise MetagratingParameterError("element_weights 须为同形状非负矩阵")
        w_sum = float(w.sum())
        if w_sum <= 0:
            raise MetagratingParameterError("element_weights 之和须为正")

        value = 0.0
        if self.complex_weight:
            value += self.complex_weight * float(np.sum(w * np.abs(S - T) ** 2)) / w_sum
        if self.amplitude_weight:
            value += (
                self.amplitude_weight
                * float(np.sum(w * (np.abs(S) - np.abs(T)) ** 2))
                / w_sum
            )
        if self.phase_weight:
            mask = np.abs(T) > 1e-12
            if np.any(mask):
                cos_dphi = np.real(S * np.conj(T)) / (np.abs(S) * np.abs(T) + 1e-300)
                cos_dphi = np.clip(cos_dphi, -1.0, 1.0)
                wm = w[mask]
                value += (
                    self.phase_weight
                    * float(np.sum(wm * (1.0 - cos_dphi[mask])))
                    / float(wm.sum())
                )
        return self.weight * value


@dataclass(frozen=True)
class ZeroElements:
    """指定矩阵元置零（单向隐身/定向抑制）。"""

    elements: tuple[tuple[int, int], ...]
    weight: float = 1.0

    def evaluate(self, S: NDArray, ports: tuple[Port, ...]) -> float:
        S = np.asarray(S, dtype=complex)
        values = []
        for i, j in self.elements:
            try:
                values.append(float(abs(S[i, j]) ** 2))
            except IndexError as exc:
                raise MetagratingParameterError(
                    f"ZeroElements 下标 ({i}, {j}) 超出 S 形状 {S.shape}"
                ) from exc
        return self.weight * float(np.mean(values)) if values else 0.0


@dataclass(frozen=True)
class JordanBlock:
    """约当块判据（本征值合并的平滑代理）。

    ``mode="trdet"`` 仅用于 2×2：``|tr²−4det|/(RMS(S)²)``，与 ``|λ₁−λ₂|²`` 恒等；
    ``mode="gap"`` 用于一般维度：最小两两本征值间隙²，按 RMS(S)² 归一化。
    本征值合并是约当块的必要条件而非充分条件；是否非可对角化用
    :func:`jordan_structure` 的几何重数诊断复核（文档见模块 docstring）。

    Args:
        weight: term 权重。
        mode: ``"trdet"`` 或 ``"gap"``。
    """

    weight: float = 1.0
    mode: str = "gap"

    def evaluate(self, S: NDArray, ports: tuple[Port, ...]) -> float:
        S = np.asarray(S, dtype=complex)
        scale = float(np.linalg.norm(S) / np.sqrt(S.shape[0]))
        scale = max(scale, 1e-12)
        if self.mode == "trdet" and S.shape == (2, 2):
            value = abs(np.trace(S) ** 2 - 4.0 * np.linalg.det(S))
        elif self.mode in ("trdet", "gap"):
            eig = np.linalg.eigvals(S)
            diff = np.abs(eig[:, None] - eig[None, :])
            np.fill_diagonal(diff, np.inf)
            value = float(np.min(diff) ** 2)
        else:
            raise MetagratingParameterError(f"未知 JordanBlock mode={self.mode!r}")
        return self.weight * value / scale**2


@dataclass(frozen=True)
class CombinedObjective:
    """按各 term 自身权重求和的组合目标。"""

    terms: tuple[ObjectiveTerm, ...]

    def evaluate(self, S: NDArray, ports: tuple[Port, ...]) -> float:
        return float(sum(term.evaluate(S, ports) for term in self.terms))

    def __call__(self, S: NDArray, ports: tuple[Port, ...]) -> float:
        return self.evaluate(S, ports)


@dataclass(frozen=True)
class JordanStructure:
    """S 矩阵的谱结构诊断（约当块判定依据）。

    Attributes:
        eigenvalues: 本征值（复数组）。
        min_gap: 最小两两本征值间距。
        algebraic: 各本征值簇的代数重数。
        geometric: 各本征值簇的几何重数 ``P − rank(S−λI)``。
        defective: 是否存在代数重数 > 几何重数的簇（非可对角化）。
    """

    eigenvalues: NDArray
    min_gap: float
    algebraic: tuple[int, ...]
    geometric: tuple[int, ...]
    defective: bool


def jordan_structure(S: NDArray, tol: float = 1e-6) -> JordanStructure:
    """诊断 S 的约当结构（本征值簇的代数/几何重数）。

    Args:
        S: 方阵。
        tol: 本征值聚类相对容差（相对 RMS 幅度）；矩阵秩的数值容差同尺度。

    Returns:
        :class:`JordanStructure`。
    """
    S = np.asarray(S, dtype=complex)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise MetagratingParameterError(f"S 须为方阵，实际形状 {S.shape}")
    eig = np.linalg.eigvals(S)
    scale = max(float(np.linalg.norm(S) / np.sqrt(S.shape[0])), 1e-12)
    clusters: list[list[complex]] = []
    for lam in eig:
        for cluster in clusters:
            if abs(lam - cluster[0]) <= tol * scale:
                cluster.append(lam)
                break
        else:
            clusters.append([lam])

    algebraic = []
    geometric = []
    for cluster in clusters:
        lam = complex(np.mean(cluster))
        algebraic.append(len(cluster))
        rank = np.linalg.matrix_rank(S - lam * np.eye(S.shape[0]), tol=tol * scale)
        geometric.append(S.shape[0] - int(rank))

    diff = np.abs(eig[:, None] - eig[None, :])
    np.fill_diagonal(diff, np.inf)
    min_gap = float(np.min(diff)) if diff.size else 0.0
    defective = any(a > g for a, g in zip(algebraic, geometric))
    return JordanStructure(
        eigenvalues=eig,
        min_gap=min_gap,
        algebraic=tuple(int(a) for a in algebraic),
        geometric=tuple(int(g) for g in geometric),
        defective=bool(defective),
    )


# ===========================================================================
# 设计空间与约束修复
# ===========================================================================


@dataclass(frozen=True)
class DesignSpace:
    """可优化的几何/介质参数空间（归一化基因型 [0,1]^dim）。

    基因型布局（dim = 5J−1）：``h[J] | w[J] | x[1:J] | cr[J] | ci[J]``。
    首槽起点固定为 ``x0``（周期平移规范），其余槽位置以绝对坐标编码。
    解码修复：
      1. 槽宽按超额比例压缩到 ``period − J·min_feature`` 预算内（保持 ≥ w 下界）；
      2. 位置逐槽夹紧到 ``[x_{s-1}+w_{s-1}+min_feature, period − (尾部槽宽+壁宽)]``，
         保证槽不重叠、尾部硬壁 ≥ min_feature、总宽 ≤ period。
    修复后的个体必然满足 ``MetagratingParams`` 的校验；公开 ``decode`` 仍走完整校验。

    Args:
        J: 每周期槽数。
        f0: 频率 (Hz)。
        c0: 背景声速 (m/s)。
        theta_i: 参考入射角（度）。
        period: 周期 D (m)。
        n_orders: Floquet 截断下限。
        k_modes: 槽内波导模截断数。
        h_bounds: 槽深上下界 (m)。
        w_bounds: 槽宽上下界 (m)。
        cr_bounds: 归一化复声速实部上下界。
        ci_bounds: 归一化复声速虚部上下界（负为增益）。
        min_feature: 硬壁最小宽度 (m)。
        x0: 首槽起点 (m)，默认 0（周期平移规范）。
    """

    J: int = 3
    f0: float = C0 / LAM
    c0: float = C0
    theta_i: float = 45.0
    period: float = D_DEFAULT
    n_orders: int = 10
    k_modes: int = 31
    h_bounds: tuple[float, float] = (0.05 * LAM, 0.85 * LAM)
    w_bounds: tuple[float, float] = (0.05 * D_DEFAULT, 0.35 * D_DEFAULT)
    cr_bounds: tuple[float, float] = (0.95, 1.05)
    ci_bounds: tuple[float, float] = (-0.3, 0.3)
    min_feature: float = 0.01 * D_DEFAULT
    x0: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.J, (int, np.integer)) or self.J < 1:
            raise MetagratingParameterError(f"J 必须为正整数，得到 {self.J!r}")
        for name in ("h_bounds", "w_bounds", "cr_bounds", "ci_bounds"):
            lo, hi = getattr(self, name)
            if not (np.isfinite(lo) and np.isfinite(hi)) or lo > hi:
                raise MetagratingParameterError(
                    f"{name} 须为有限且 lo≤hi，得到 {(lo, hi)}"
                )
        if self.period <= 0 or self.min_feature < 0:
            raise MetagratingParameterError("period 须为正、min_feature 须非负")
        if self.w_bounds[0] <= 0:
            raise MetagratingParameterError("槽宽下界须为正")
        if self.J * (self.w_bounds[0] + self.min_feature) > self.period:
            raise MetagratingParameterError(
                "设计空间不可行：J·(w_min+min_feature) 超出周期 D"
            )
        if not (0.0 <= self.x0 < self.period):
            raise MetagratingParameterError(f"x0 须在 [0, D) 内，得到 {self.x0}")

    @property
    def dim(self) -> int:
        """归一化基因型维度 dim = 5J−1。"""
        return 5 * self.J - 1

    def encode(self, params: MetagratingParams) -> NDArray:
        """把已知参数编码为归一化基因型（越界抛 ``MetagratingParameterError``）。"""
        if params.J != self.J:
            raise MetagratingParameterError(
                f"参数 J={params.J} 与设计空间 J={self.J} 不一致"
            )
        x = np.asarray(params.groove_positions, dtype=float)
        tol = 1e-9 * self.period
        if abs(x[0] - self.x0) > tol:
            raise MetagratingParameterError(
                f"参数首槽位置 {x[0]} 与设计空间规范 x0={self.x0} 不一致"
            )
        c = np.asarray(params.c_complex, dtype=complex)
        raw = np.concatenate(
            [
                np.asarray(params.h, dtype=float),
                np.asarray(params.w, dtype=float),
                x[1:],
                np.real(c) / params.c0,
                np.imag(c) / params.c0,
            ]
        )
        spans = self._spans()
        norm = (raw - spans[:, 0]) / (spans[:, 1] - spans[:, 0])
        if np.any(norm < -1e-9) or np.any(norm > 1.0 + 1e-9):
            raise MetagratingParameterError(
                "参数超出设计空间上下界：" + str(np.round(norm, 3))
            )
        return np.clip(norm, 0.0, 1.0)

    def _spans(self) -> NDArray:
        J = self.J
        lo = np.concatenate(
            [
                np.full(J, self.h_bounds[0]),
                np.full(J, self.w_bounds[0]),
                np.full(J - 1, self.x0),
                np.full(J, self.cr_bounds[0]),
                np.full(J, self.ci_bounds[0]),
            ]
        )
        hi = np.concatenate(
            [
                np.full(J, self.h_bounds[1]),
                np.full(J, self.w_bounds[1]),
                np.full(J - 1, self.x0 + self.period),
                np.full(J, self.cr_bounds[1]),
                np.full(J, self.ci_bounds[1]),
            ]
        )
        return np.stack([lo, hi], axis=1)

    def decode_arrays(self, genome: NDArray) -> dict[str, NDArray]:
        """基因型 → 修复后的物理数组（h/w/x/c_complex），不做完整参数校验。"""
        g = np.asarray(genome, dtype=float)
        if g.shape != (self.dim,):
            raise MetagratingParameterError(
                f"基因型形状应为 ({self.dim},)，实际 {g.shape}"
            )
        g = np.clip(g, 0.0, 1.0)
        J = self.J
        h = self.h_bounds[0] + g[0:J] * (self.h_bounds[1] - self.h_bounds[0])
        w = self.w_bounds[0] + g[J : 2 * J] * (self.w_bounds[1] - self.w_bounds[0])
        x_raw = self.x0 + g[2 * J : 3 * J - 1] * self.period
        cr = self.cr_bounds[0] + g[3 * J - 1 : 4 * J - 1] * (
            self.cr_bounds[1] - self.cr_bounds[0]
        )
        ci = self.ci_bounds[0] + g[4 * J - 1 :] * (
            self.ci_bounds[1] - self.ci_bounds[0]
        )

        # 槽宽修复：保持下界，按超额比例压缩到 (period − J·min_feature) 预算内。
        budget = self.period - J * self.min_feature
        excess = w - self.w_bounds[0]
        excess_sum = float(excess.sum())
        if excess_sum > budget - J * self.w_bounds[0]:
            scale = (budget - J * self.w_bounds[0]) / excess_sum
            w = self.w_bounds[0] + excess * scale

        # 位置修复：逐槽夹紧，保证不重叠、尾部硬壁与总宽约束。
        x = np.empty(J, dtype=float)
        x[0] = self.x0
        tail = np.cumsum(w[::-1])[::-1]  # tail[s] = Σ_{t≥s} w_t
        for s in range(1, J):
            lower = x[s - 1] + w[s - 1] + self.min_feature
            upper = self.x0 + self.period - (tail[s] + (J - s) * self.min_feature)
            if lower > upper + 1e-12:
                raise MetagratingParameterError(
                    f"槽位修复不可行：s={s}, lower={lower:.6g} > upper={upper:.6g}"
                )
            x[s] = min(max(float(x_raw[s - 1]), lower), upper)

        return {
            "h": h,
            "w": w,
            "x": x,
            "c_complex": self.c0 * (cr + 1j * ci),
            "cr": cr,
            "ci": ci,
        }

    def decode(self, genome: NDArray) -> MetagratingParams:
        """基因型 → ``MetagratingParams``（走完整校验，非法即抛异常）。"""
        arr = self.decode_arrays(genome)
        return MetagratingParams(
            f0=self.f0,
            c0=self.c0,
            theta_i=self.theta_i,
            period=self.period,
            J=self.J,
            h=arr["h"],
            w=arr["w"],
            x=arr["x"],
            c_complex=arr["c_complex"],
            n_orders=self.n_orders,
            k_modes=self.k_modes,
        )

    def perturbed_population(
        self,
        params: MetagratingParams,
        *,
        scale: float = 0.01,
        popsize: int = 24,
        seed: int = 0,
    ) -> NDArray:
        """在已知参数的归一化坐标附近生成扰动初始种群 (popsize, dim)。"""
        rng = np.random.default_rng(seed)
        center = self.encode(params)
        noise = rng.uniform(-scale, scale, size=(popsize, self.dim))
        return np.clip(center + noise, 0.0, 1.0)


# ===========================================================================
# 评估器（可 pickle，供多进程复用）
# ===========================================================================


class Evaluator:
    """个体评估器：复用 params 模板与 t2 快核缓存。

    设计空间固定的字段只构造一次模板 ``MetagratingParams``，评分时仅替换
    h/w/x/c_complex 数组并调用 ``compute_s_matrix``；修复后的数组保证合法，
    因此不重复执行参数校验（解码的 `decode` 仍提供完整校验语义）。

    Args:
        space: 设计空间。
        objective: 目标函数（可调用 ``(S, ports) -> float``）。
        ports: 参考端口元数据（与候选几何无关，端口数由 f0/θ/D 固定）。
    """

    def __init__(
        self,
        space: DesignSpace,
        objective: Callable[[NDArray, tuple[Port, ...]], float],
        ports: tuple[Port, ...],
    ) -> None:
        self.space = space
        self.objective = objective
        self.ports = ports
        self._template = space.decode(np.full(space.dim, 0.5))

    def score(self, genome: NDArray) -> float:
        """评估一个归一化基因型的标量目标值。"""
        arr = self.space.decode_arrays(genome)
        p = self._template
        p.h = arr["h"]
        p.w = arr["w"]
        p.x = arr["x"]
        p.c_complex = arr["c_complex"]
        S = compute_s_matrix(p)
        return float(self.objective(S, self.ports))


_WORKER: Evaluator | None = None


def _init_worker(evaluator: Evaluator) -> None:
    global _WORKER
    _WORKER = evaluator


def _worker_score(genome: NDArray) -> float:
    if _WORKER is None:  # pragma: no cover - 仅进程初始化失败时可见
        raise RuntimeError("worker 未初始化 Evaluator")
    return _WORKER.score(genome)


# ===========================================================================
# 差分进化
# ===========================================================================


def _de_trials(
    population: NDArray,
    rng: np.random.Generator,
    best: NDArray,
    F: float,
    CR: float,
    strategy: str,
) -> NDArray:
    """生成一整代试验向量（DE/rand/1/bin 或 DE/best/1/bin）。"""
    popsize, dim = population.shape
    trials = np.empty_like(population)
    for i in range(popsize):
        idx = rng.choice(popsize - 1, size=3, replace=False)
        idx[idx >= i] += 1  # 映射到不等于 i 的下标
        r1, r2, r3 = idx
        base = population[r1] if strategy == "rand1" else best
        mutant = base + F * (population[r2] - population[r3])
        mutant = np.clip(mutant, 0.0, 1.0)
        cross = rng.random(dim) < CR
        if not cross.any():
            cross[rng.integers(dim)] = True
        trials[i] = np.where(cross, mutant, population[i])
    return trials


@dataclass
class InverseResult:
    """逆向设计结果。

    Attributes:
        params: 最优个体对应的超表面参数。
        genome: 最优归一化基因型。
        objective_value: 最优目标值。
        S: 最优个体的 S 矩阵 (P, P)。
        ports: 端口元数据。
        history: 每代最优目标值（best-so-far）。
        generations: 实际完成代数。
        evaluations: 目标函数评估次数。
        output_dir: 结果导出目录（未导出为 None）。
        checkpoint_path: 检查点路径（未启用为 None）。
        evaluation_time: 目标函数评估阶段墙钟 (s)，用于计算并行评估吞吐。
    """

    params: MetagratingParams
    genome: NDArray
    objective_value: float
    S: NDArray
    ports: tuple[Port, ...]
    history: NDArray
    generations: int
    evaluations: int
    output_dir: Path | None = None
    checkpoint_path: Path | None = None
    evaluation_time: float = 0.0

    def summary(self) -> dict:
        """人类可读的结果摘要（JSON 友好）。"""
        p = self.params
        eval_rate = self.evaluations / max(self.evaluation_time, 1e-12)
        return {
            "objective_value": float(self.objective_value),
            "generations": int(self.generations),
            "evaluations": int(self.evaluations),
            "evaluation_time": float(self.evaluation_time),
            "evaluations_per_s": float(eval_rate),
            "f0": float(p.f0),
            "period": float(p.period),
            "theta_i": float(p.theta_i),
            "J": int(p.J),
            "h": [float(v) for v in p.h],
            "w": [float(v) for v in p.w],
            "x": [float(v) for v in p.groove_positions],
            "c_complex": [[float(v.real), float(v.imag)] for v in p.c_complex],
            "ports": [
                {
                    "order": port.order,
                    "kx": port.kx,
                    "angle_deg": port.angle_deg,
                }
                for port in self.ports
            ],
            "history": [float(v) for v in self.history],
        }


def _checkpoint_write(
    path: Path,
    *,
    population: NDArray,
    fitness: NDArray,
    history: NDArray,
    generation: int,
    evaluations: int,
    evaluation_time: float,
    rng_state: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        population=population,
        fitness=fitness,
        history=history,
        generation=np.array(generation),
        evaluations=np.array(evaluations),
        evaluation_time=np.array(evaluation_time),
        rng_state=np.array(json.dumps(rng_state)),
    )


def _checkpoint_read(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {
            "population": np.array(data["population"]),
            "fitness": np.array(data["fitness"]),
            "history": np.array(data["history"]),
            "generation": int(data["generation"]),
            "evaluations": int(data["evaluations"]),
            "evaluation_time": (
                float(data["evaluation_time"]) if "evaluation_time" in data else 0.0
            ),
            "rng_state": json.loads(str(data["rng_state"])),
        }


def _score_batch(
    genomes: NDArray,
    evaluator: Evaluator,
    pool: ProcessPoolExecutor | None,
    chunksize: int = 1,
) -> NDArray:
    if pool is None:
        return np.array([evaluator.score(g) for g in genomes], dtype=float)
    return np.fromiter(
        pool.map(_worker_score, genomes, chunksize=chunksize),
        dtype=float,
        count=len(genomes),
    )


def _export_result(
    result: InverseResult,
    output_dir: Path,
    *,
    assert_data: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if assert_data:
        assert_within_data(output_dir)
    p = result.params
    np.savez(
        output_dir / "result.npz",
        genome=result.genome,
        h=p.h,
        w=p.w,
        x=p.groove_positions,
        c_complex=p.c_complex,
        S=result.S,
        history=result.history,
        objective_value=np.array(result.objective_value),
        generations=np.array(result.generations),
        evaluations=np.array(result.evaluations),
        port_orders=np.array([port.order for port in result.ports]),
        port_kx=np.array([port.kx for port in result.ports]),
        port_angles_deg=np.array([port.angle_deg for port in result.ports]),
    )
    (output_dir / "summary.json").write_text(
        json.dumps(result.summary(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def inverse_design(
    space: DesignSpace,
    objective: Callable[[NDArray, tuple[Port, ...]], float],
    *,
    popsize: int = 32,
    max_generations: int = 100,
    F: float = 0.7,
    CR: float = 0.9,
    strategy: str = "rand1",
    seed: int = 0,
    workers: int = 1,
    patience: int | None = None,
    tol: float = 1e-8,
    initial_population: NDArray | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = 1,
    resume: bool = False,
    output_dir: str | Path | None = None,
    verbose: bool = False,
) -> InverseResult:
    """用差分进化做 CMT 逆向设计。

    目标函数签名 ``objective(S, ports) -> float``（可用 :class:`CombinedObjective`
    组合 :class:`MatrixMatch` / :class:`ZeroElements` / :class:`JordanBlock`）。
    每代只评估试验向量；种群级多进程并行（``workers``）；检查点保存种群、适应度、
    收敛史与 RNG 状态，``resume=True`` 从 ``checkpoint_path`` 精确定点续算。

    Args:
        space: 设计空间（几何/介质上下界与约束修复）。
        objective: 标量目标函数。
        popsize: 种群规模（≥4）。
        max_generations: 最大代数（含续算前的已完成代数）。
        F: 差分权重。
        CR: 交叉概率。
        strategy: ``"rand1"`` 或 ``"best1"``。
        seed: RNG 种子（续算时以检查点内 RNG 状态为准）。
        workers: 进程数；0 表示用满全部核，1 表示串行。
        patience: 连续多少代无改进则停止（None 不启用）。
        tol: 改进阈值（best-so-far 下降量 < tol 视为无改进）。
        initial_population: 可选 (popsize, dim) 归一化初始种群。
        checkpoint_path: 检查点 npz 路径（None 不保存）。
        checkpoint_every: 每多少代写一次检查点。
        resume: 从检查点续算（要求文件存在）。
        output_dir: 结果导出目录（None 不导出；默认路径由调用方给定）。
        verbose: 是否打印每代进度。

    Returns:
        :class:`InverseResult`。

    Raises:
        MetagratingParameterError: 参数/形状不合法。
        FileNotFoundError: ``resume=True`` 但检查点不存在。
    """
    if popsize < 4:
        raise MetagratingParameterError("popsize 须 ≥4（DE 需要 3 个互异个体）")
    if not (0.0 < F <= 2.0):
        raise MetagratingParameterError(f"F 须在 (0, 2]，得到 {F}")
    if not (0.0 <= CR <= 1.0):
        raise MetagratingParameterError(f"CR 须在 [0, 1]，得到 {CR}")
    if strategy not in ("rand1", "best1"):
        raise MetagratingParameterError(f"未知 strategy={strategy!r}")
    if checkpoint_every < 1:
        raise MetagratingParameterError("checkpoint_every 须 ≥1")

    ckpt = Path(checkpoint_path) if checkpoint_path is not None else None
    if resume and (ckpt is None or not ckpt.exists()):
        raise FileNotFoundError(f"检查点不存在：{ckpt}")

    rng = np.random.default_rng(seed)
    dim = space.dim
    history: list[float] = []
    if resume and ckpt is not None:
        state = _checkpoint_read(ckpt)
        if state["population"].shape != (popsize, dim):
            raise MetagratingParameterError(
                f"检查点种群形状 {state['population'].shape} 与当前 {(popsize, dim)} 不一致"
            )
        population = state["population"]
        fitness = state["fitness"]
        history = [float(v) for v in state["history"]]
        generation = state["generation"]
        evaluations = state["evaluations"]
        evaluation_time = state["evaluation_time"]
        rng.bit_generator.state = state["rng_state"]
    else:
        if initial_population is None:
            population = rng.random((popsize, dim))
        else:
            population = np.asarray(initial_population, dtype=float)
            if population.shape != (popsize, dim):
                raise MetagratingParameterError(
                    f"initial_population 形状应为 {(popsize, dim)}，实际 {population.shape}"
                )
            population = np.clip(population, 0.0, 1.0)
        generation = 0
        evaluations = 0
        evaluation_time = 0.0

    ports = solve_s_matrix(space.decode(np.full(dim, 0.5))).ports
    evaluator = Evaluator(space, objective, ports)

    pool: ProcessPoolExecutor | None = None
    if workers != 1:
        pool = ProcessPoolExecutor(
            max_workers=None if workers <= 0 else workers,
            initializer=_init_worker,
            initargs=(evaluator,),
        )
    workers_eff = workers if workers > 0 else (os.cpu_count() or 1)
    chunksize = max(1, popsize // (4 * max(workers_eff, 1)))
    t_start = time.perf_counter()
    try:
        if generation == 0 and not history:
            t_eval = time.perf_counter()
            fitness = _score_batch(population, evaluator, pool, chunksize)
            evaluation_time += time.perf_counter() - t_eval
            evaluations += popsize
            best_idx = int(np.argmin(fitness))
            history.append(float(fitness[best_idx]))
            generation = 1

        best_idx = int(np.argmin(fitness))
        best_value = float(fitness[best_idx])
        stale = 0
        while generation < max_generations:
            best_genome = population[best_idx].copy()
            trials = _de_trials(population, rng, best_genome, F, CR, strategy)
            t_eval = time.perf_counter()
            trial_fitness = _score_batch(trials, evaluator, pool, chunksize)
            evaluation_time += time.perf_counter() - t_eval
            evaluations += popsize
            improve = trial_fitness <= fitness
            population[improve] = trials[improve]
            fitness[improve] = trial_fitness[improve]

            new_idx = int(np.argmin(fitness))
            new_value = float(fitness[new_idx])
            if new_value < best_value - tol:
                best_value = new_value
                best_idx = new_idx
                stale = 0
            else:
                stale += 1
            history.append(best_value)
            generation += 1

            if verbose:
                elapsed = max(time.perf_counter() - t_start, 1e-12)
                print(
                    f"[gen {generation:4d}] best={best_value:.6e} "
                    f"eval={evaluations} ({evaluations / elapsed:.0f}/s)"
                )
            if ckpt is not None and generation % checkpoint_every == 0:
                _checkpoint_write(
                    ckpt,
                    population=population,
                    fitness=fitness,
                    history=np.asarray(history),
                    generation=generation,
                    evaluations=evaluations,
                    evaluation_time=evaluation_time,
                    rng_state=rng.bit_generator.state,
                )
            if patience is not None and stale >= patience:
                break
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    best_idx = int(np.argmin(fitness))
    best_genome = population[best_idx].copy()
    best_params = space.decode(best_genome)
    S = compute_s_matrix(best_params)
    result = InverseResult(
        params=best_params,
        genome=best_genome,
        objective_value=float(fitness[best_idx]),
        S=S,
        ports=ports,
        history=np.asarray(history),
        generations=generation,
        evaluations=evaluations,
        checkpoint_path=ckpt,
        evaluation_time=evaluation_time,
    )
    if output_dir is not None:
        out = Path(output_dir)
        result.output_dir = out
        in_data = out.resolve().is_relative_to(DATA_ROOT.resolve())
        _export_result(result, out, assert_data=in_data)
    return result


def default_output_dir(name: str) -> Path:
    """返回 ``data/research/<line>/theory/inverse_design/<name>`` 规范导出目录。"""
    return research_theory_dir("gain_ep", slug=f"inverse_design/{name}")
