"""逆设计薄胶水：目标/约束协议 + scipy DE 封装 + 每代 checkpoint（可恢复续跑）。

原地 ADR（用户裁决 2026-11-17）：旧研究线自研 DE/rand/1/bin **不迁移**，统一换
``scipy.optimize.differential_evolution``；本模块只保留三件薄胶水——目标/约束协议
（参数向量→标量）、每代 checkpoint（npz+json）、按种群 warm-restart 续跑。行为差异
（修复式解码→可行性规则、收敛判据、RNG 流重启续算）见
``.agents/skills/theory-calculation/references/recipe-inverse-design.md`` §4。
⭐几何修复式解码/进程池编排留研究线。路径走 :mod:`pysci.paths`（无 ``parents[N]``）：
checkpoint 是运行产物，写入限 ``data/`` 内（``assert_within_data`` 护栏）。

机械断言：``tests/skills/theory_calculation/test_inverse_design.py``。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import NonlinearConstraint, OptimizeResult, differential_evolution

from pysci.paths import assert_within_data

__all__ = [
    "Checkpoint",
    "CheckpointCallback",
    "Constraint",
    "Objective",
    "load_checkpoint",
    "run_inverse_design",
    "save_checkpoint",
]


class Objective(Protocol):
    """目标函数协议：参数向量 ``x``（一维）→ 标量目标（越小越好）。

    不可行点返回 ``+inf`` 罚分（勿抛异常）；``NaN`` 视为编程错误，主入口拦截。
    """

    def __call__(self, x: NDArray[np.float64]) -> float: ...


class Constraint(Protocol):
    """约束协议：参数向量 ``x`` → 标量残差，``≤0`` 可行（映射为 NonlinearConstraint）。"""

    def __call__(self, x: NDArray[np.float64]) -> float: ...


@dataclass(frozen=True)
class Checkpoint:
    """每代 checkpoint 状态（npz/json 落盘与恢复的往返单位）。

    Attributes:
        generation: 累计代数（续跑含历史）；best_x/best_fun: 当前最优。
        population/population_energies: 种群与适应度（物理坐标）。
        nfev: 累计评估次数；convergence: scipy 收敛度读数（tol/散布度口径）。
    """

    generation: int
    best_x: NDArray[np.float64]
    best_fun: float
    population: NDArray[np.float64]
    population_energies: NDArray[np.float64]
    nfev: int
    convergence: float


def _as_npz_path(path: Path | str) -> Path:
    """校验 checkpoint 路径为 ``.npz`` 后缀（json 摘要按同 stem 派生）。"""
    p = Path(path)
    if p.suffix != ".npz":
        raise ValueError(f"checkpoint 路径须以 .npz 结尾，实际 {p}")
    return p


def save_checkpoint(path: Path | str, ckpt: Checkpoint) -> Path:
    """checkpoint 落盘：``<path>.npz``（数组真源）+ ``<path>.json``（人读摘要）。

    写入须在 ``data/`` 内（运行产物约定）。Returns: ``.npz`` 路径。
    Raises: ValueError（路径出 data/ 或非 .npz 后缀）。
    """
    out = assert_within_data(_as_npz_path(path), what="逆设计 checkpoint")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        generation=np.array(ckpt.generation),
        best_x=ckpt.best_x,
        best_fun=np.array(ckpt.best_fun),
        population=ckpt.population,
        population_energies=ckpt.population_energies,
        nfev=np.array(ckpt.nfev),
        convergence=np.array(ckpt.convergence),
    )
    summary = {
        "generation": ckpt.generation,
        "best_fun": ckpt.best_fun,
        "best_x": [float(v) for v in ckpt.best_x],
        "nfev": ckpt.nfev,
        "convergence": ckpt.convergence,
        "population_shape": list(ckpt.population.shape),
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    out.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out


def load_checkpoint(path: Path | str) -> Checkpoint:
    """从 ``<path>.npz`` 恢复状态（:func:`save_checkpoint` 的逆）。

    Raises: FileNotFoundError（npz 不存在）；ValueError（字段缺失/形状不一致）。
    """
    src = _as_npz_path(path)
    if not src.is_file():
        raise FileNotFoundError(f"checkpoint 不存在：{src}")
    with np.load(src) as data:
        required = {
            "generation", "best_x", "best_fun", "population",
            "population_energies", "nfev", "convergence",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"checkpoint 字段缺失：{sorted(missing)}")
        ckpt = Checkpoint(
            generation=int(data["generation"]),
            best_x=np.asarray(data["best_x"], dtype=float),
            best_fun=float(data["best_fun"]),
            population=np.asarray(data["population"], dtype=float),
            population_energies=np.asarray(data["population_energies"], dtype=float),
            nfev=int(data["nfev"]),
            convergence=float(data["convergence"]),
        )
    if ckpt.population.ndim != 2 or ckpt.population.shape[0] != ckpt.population_energies.shape[0]:
        raise ValueError("population 与 population_energies 形状不一致")
    return ckpt


class CheckpointCallback:
    """scipy DE 每代 checkpoint 回调（返回 ``False`` 继续优化）。

    **参数名必须恰为 ``intermediate_result``**：scipy 按参数名路由，只有该名能拿到
    含 ``population`` 的 ``OptimizeResult``（否则退化为 (xk, convergence) 旧签名）。
    ``generation_offset``：续跑的历史代数偏移（累计代数 = res.nit + offset）。
    """

    def __init__(self, path: Path | str, *, generation_offset: int = 0) -> None:
        self.path = _as_npz_path(path)
        self.generation_offset = int(generation_offset)

    def __call__(self, intermediate_result: Any) -> bool:
        """把当前代状态写入 checkpoint（npz+json）。"""
        res = intermediate_result
        save_checkpoint(self.path, Checkpoint(
            generation=int(res.nit) + self.generation_offset,
            best_x=np.asarray(res.x, dtype=float).copy(),
            best_fun=float(res.fun),
            population=np.asarray(res.population, dtype=float).copy(),
            population_energies=np.asarray(res.population_energies, dtype=float).copy(),
            nfev=int(res.nfev),
            convergence=float(res.convergence),
        ))
        return False


def _validate_bounds(bounds: Sequence[tuple[float, float]]) -> tuple[NDArray, NDArray]:
    """bounds → (lo, hi)；拦截形状/非有限/lo≥hi（scipy 不校验 lo>hi）。"""
    arr = np.asarray(bounds, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 1:
        raise ValueError(f"bounds 须为 (维数, 2) 的 (lo, hi) 序列，实际形状 {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("bounds 须为有限值")
    lo, hi = arr[:, 0], arr[:, 1]
    if np.any(lo >= hi):
        raise ValueError("bounds 每维须 lo < hi（scipy 不校验，此处拦截）")
    return lo, hi


def run_inverse_design(
    objective: Objective,
    bounds: Sequence[tuple[float, float]],
    *,
    constraints: Sequence[Constraint] = (),
    seed: int = 0,
    popsize: int = 15,
    maxiter: int = 100,
    tol: float = 1e-6,
    polish: bool = False,
    checkpoint_path: Path | str | None = None,
    resume_from: Path | str | None = None,
) -> OptimizeResult:
    """scipy DE 逆设计主入口：输入校验 → 每代 checkpoint → 可恢复续跑。

    Args:
        objective: 目标协议（越小越好；不可行点返回 ``+inf``）。
        bounds: 逐维 ``(lo, hi)``，须有限且 ``lo < hi``。
        constraints: 约束协议序列（``≤0`` 可行）。
        seed: 随机种子（固定种子全流程确定性）。
        popsize: 种群倍率（总规模 = popsize × 维数）。
        maxiter: 本**次**运行的最大代数（续跑不含历史代数）。
        tol: 种群收敛容差（scipy 判据，行为差异见 recipe §4）。
        polish: 是否末尾 L-BFGS-B（约束下 trust-constr）精修。
        checkpoint_path: 每代 checkpoint 的 ``.npz`` 路径（None 不落盘）。
        resume_from: 从该 checkpoint 种群 warm-restart 续跑（种群精确接续、
            RNG 流重启；非自研 DE 的 RNG 精确续算，见 recipe §4）。

    Returns:
        scipy ``OptimizeResult``（``x``/``fun``/``nit``/``population`` 等）。

    Raises:
        ValueError: bounds/参数非法、目标返回 NaN（中点哨兵求值拦截；运行期
            NaN 仍被 ``wrapped`` 拦截但会被 scipy 包为 RuntimeError）、
            续跑种群维度不符。
        FileNotFoundError: ``resume_from`` 不存在。
    """
    lo, hi = _validate_bounds(bounds)
    if not callable(objective):
        raise TypeError("objective 必须可调用（参数向量→标量）")
    if not isinstance(popsize, (int, np.integer)) or popsize < 1:
        raise ValueError(f"popsize 须为正整数，得到 {popsize!r}")
    if not isinstance(maxiter, (int, np.integer)) or maxiter < 0:
        raise ValueError(f"maxiter 须为非负整数，得到 {maxiter!r}")

    init: NDArray[np.float64] | str = "latinhypercube"
    offset = 0
    if resume_from is not None:
        ckpt = load_checkpoint(resume_from)
        if ckpt.population.shape[1] != lo.shape[0]:
            raise ValueError(
                f"续跑种群维度 {ckpt.population.shape[1]} 与 bounds 维数 {lo.shape[0]} 不符"
            )
        init = ckpt.population
        offset = ckpt.generation

    if np.isnan(float(objective((lo + hi) / 2.0))):  # 哨兵求值：非法目标当场拦截
        raise ValueError("objective 返回 NaN（编程错误；不可行点用 +inf 罚分）")

    callback = None
    if checkpoint_path is not None:
        callback = CheckpointCallback(checkpoint_path, generation_offset=offset)

    def wrapped(x: NDArray[np.float64]) -> float:
        value = float(objective(x))
        if np.isnan(value):
            raise ValueError("objective 返回 NaN（编程错误；不可行点用 +inf 罚分）")
        return value

    for c in constraints:
        if not callable(c):
            raise TypeError("constraints 的元素必须可调用（参数向量→标量残差，≤0 可行）")
    scipy_constraints = []
    if constraints:
        def residual(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.array([float(c(x)) for c in constraints], dtype=float)
        scipy_constraints = [NonlinearConstraint(residual, -np.inf, 0.0)]

    return differential_evolution(
        wrapped,
        list(zip(lo, hi, strict=True)),
        constraints=scipy_constraints,
        seed=seed,
        popsize=popsize,
        maxiter=maxiter,
        tol=tol,
        polish=polish,
        init=init,
        callback=callback,
    )
