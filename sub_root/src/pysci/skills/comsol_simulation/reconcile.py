"""理论↔仿真 S 矩阵对账契约：同入同出校验 + 幅度对账 + 冻结判据（纯函数）。

工作流位（SKILL.md 验证步）：理论核与 COMSOL 模板各自求出 S 矩阵后，先过
:func:`check_same_in_same_out` 同入同出契约，再进 :func:`compare_s_matrix`
幅度对账。本模块全部纯函数——输入普通数据结构，输出 :class:`CheckResult` 或
数据类，不触碰 mph/COMSOL；把两侧求解编排进来的调用在研究树/调用方。

同入同出契约：同一参数集（``params_digest`` 相同）分别进理论核与 COMSOL 模板，
两侧必须产出**同语义** S 矩阵——同形状、同频率、同端口序列。契约不成立时
:func:`compare_s_matrix` 直接拒绝（ValueError），不进入对账。

ADR（2026-11-17，落位）：本模块是 CMT 管线吸收计划（``others/plans/cmt-absorption.md``）
的②侧产物。**独立成模块而非并入 validate.py**——validate.py 是"单解物理合理性检查"
（能量守恒/收敛阶/标量序列比对），对账需要携带元数据的 S 矩阵载体
（:class:`SMatrixCase`）与冻结判据常量，职责不同；返回风格复用
:class:`validate.CheckResult` 保持一致。

ADR（2026-11-17，来源与零复制）：对账契约语义提取自旧 pySciWS gain_ep 管线
``cmt_comsol_validation``（只读解剖见 ``community_research/2026-11-17/
legacy-02-cmt-gain-ep.md``）；⭐**零复制**——旧研究线代码一行未搬，全部按新 API 重写。

ADR（冻结判据）：主导元绝对幅度差 ``|Δ| ≤ 0.02``（:data:`DOMINANT_ABS_TOL`）来自
旧管线实证验收口径（"主导元绝对 |Δ|≤0.02（冻结）"；旧研究线已知系统偏差锚点 0.0388
为其 D1 未闭合现状，非本判据值）。**冻结纪律**：此值为验收判据常量，写死后不得
随模型/调参改动；如需变更必须用户裁决并改写本 ADR。

ADR（幅度口径，冻结）：COMSOL 与理论核的相位参考相差一个固定相位，对账**只比较
|S| 幅度**（旧管线口径"勿改"）；相位对账不在本模块范围。
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from pysci.skills.comsol_simulation.validate import CheckResult

#: 相对指标（max_rel_diff）只在 ``|S_theory| > REL_FLOOR`` 的元素上计；
#: 避免近零元的相对误差爆炸（旧管线语义：阈 1e-3）。
REL_FLOOR = 1e-3

#: 主导元判定阈：``|S_theory| > DOMINANT_FLOOR`` 视为主导元（旧管线语义：阈 0.2）。
DOMINANT_FLOOR = 0.2

#: 冻结判据阈：主导元最大绝对幅度差 ``|Δ| ≤ DOMINANT_ABS_TOL`` 才 PASS。
#: 数值来源与冻结纪律见模块 ADR。
DOMINANT_ABS_TOL = 0.02


@dataclass(frozen=True)
class SMatrixCase:
    """一次求解产出的同语义 S 矩阵（理论核或 COMSOL 模板同出形态）。

    Attributes:
        s: n×n 复散射矩阵；``s[p][q]`` = q 端口入射、p 端口出射的散射系数。
        frequency_hz: 求解频率（Hz，有限正数）。
        port_labels: 端口语义标签，长度 n，顺序即矩阵端口顺序。
        params_digest: 入参参数集摘要（:func:`digest_of`）；两侧相同即"同入"。
        source: 产出方标注（如 ``"theory"`` / ``"comsol"``），仅诊断用。

    Raises:
        ValueError: 矩阵非方阵/含非有限元/频率非法/端口标签数不符/摘要为空。
    """

    s: tuple[tuple[complex, ...], ...]
    frequency_hz: float
    port_labels: tuple[str, ...]
    params_digest: str
    source: str = ""

    def __post_init__(self) -> None:
        rows = tuple(tuple(complex(v) for v in row) for row in self.s)
        if not rows or any(len(row) != len(rows) for row in rows):
            raise ValueError(f"S 矩阵须为非空方阵：{[len(r) for r in rows]}")
        for row in rows:
            for v in row:
                if not (math.isfinite(v.real) and math.isfinite(v.imag)):
                    raise ValueError(f"S 矩阵含非有限元：{v!r}")
        if not (math.isfinite(self.frequency_hz) and self.frequency_hz > 0):
            raise ValueError(f"frequency_hz 须为有限正数：{self.frequency_hz!r}")
        labels = tuple(self.port_labels)
        if len(labels) != len(rows):
            raise ValueError(
                f"端口标签数 {len(labels)} 与矩阵阶数 {len(rows)} 不符"
            )
        if not self.params_digest:
            raise ValueError("params_digest 不得为空（同入校验依据）")
        object.__setattr__(self, "s", rows)
        object.__setattr__(self, "port_labels", labels)


@dataclass(frozen=True)
class AmplitudeMetrics:
    """幅度对账全指标（幅度口径，见模块 ADR）。

    Attributes:
        max_abs_diff: 全元素最大绝对幅度差 ``max ||S_th|-|S_sim||``。
        max_rel_diff: 相对幅度差最大值，仅计 ``|S_th| > REL_FLOOR`` 元素
            （无合格元素时为 0.0）。
        rms_diff: 全元素绝对幅度差的均方根。
        dominant_max_abs_diff: 主导元（``|S_th| > DOMINANT_FLOOR``）最大绝对
            幅度差——冻结判据读数；无主导元时为 0.0 且 ``n_dominant == 0``。
        n_elements: 元素总数。
        n_dominant: 主导元个数。
    """

    max_abs_diff: float
    max_rel_diff: float
    rms_diff: float
    dominant_max_abs_diff: float
    n_elements: int
    n_dominant: int


def digest_of(canonical_params: str) -> str:
    """参数集稳定摘要：规范化参数串的 sha256 前 16 个十六进制字符。

    Args:
        canonical_params: 参数集的规范化字符串表示（调用方负责规范化，
            理论核与 COMSOL 模板两侧必须喂同一串）。

    Returns:
        16 字符十六进制摘要，用于 :attr:`SMatrixCase.params_digest`。
    """
    return hashlib.sha256(canonical_params.encode("utf-8")).hexdigest()[:16]


def check_same_in_same_out(
    theory: SMatrixCase,
    sim: SMatrixCase,
    *,
    name: str = "same_in_same_out",
) -> CheckResult:
    """同入同出契约校验：同参数集进 → 同语义 S 矩阵出。

    校验四条：①同入（params_digest 相等）；②形状（阶数相等）；③频率
    （frequency_hz 严格相等——契约层不做容差，容差属物理判据）；④端口元数据
    （port_labels 序列相等，顺序敏感）。全部成立才 PASS；违规条目逐条列出。

    Args:
        theory: 理论核产出的 S 矩阵。
        sim: COMSOL 模板产出的 S 矩阵。
        name: 检查名。

    Returns:
        :class:`CheckResult`；``value`` 恒为 None（本检查无量纲）。
    """
    violated: list[str] = []
    if theory.params_digest != sim.params_digest:
        violated.append(
            f"同入违反：参数摘要 {theory.params_digest} != {sim.params_digest}"
        )
    n_th, n_sim = len(theory.s), len(sim.s)
    if n_th != n_sim:
        violated.append(f"形状违反：{n_th}x{n_th} vs {n_sim}x{n_sim}")
    if theory.frequency_hz != sim.frequency_hz:
        violated.append(
            f"频率违反：{theory.frequency_hz} Hz != {sim.frequency_hz} Hz"
        )
    if theory.port_labels != sim.port_labels:
        violated.append(
            f"端口元数据违反：{theory.port_labels} != {sim.port_labels}"
        )
    if violated:
        return CheckResult(
            name=name,
            ok=False,
            value=None,
            message=f"FAIL {name}：" + "；".join(violated),
        )
    return CheckResult(
        name=name,
        ok=True,
        value=None,
        message=(
            f"PASS {name}：同入同出契约成立（摘要 {theory.params_digest}，"
            f"{n_th}x{n_th}，{theory.frequency_hz} Hz，端口 {theory.port_labels}）"
        ),
    )


def amplitude_metrics(
    theory_s: Sequence[Sequence[complex]],
    sim_s: Sequence[Sequence[complex]],
) -> AmplitudeMetrics:
    """幅度对账全指标（幅度口径：只比较 |S|，相位不入账，见模块 ADR）。

    Args:
        theory_s: 理论核 n×n 复 S 矩阵。
        sim_s: 仿真 n×n 复 S 矩阵（与 theory_s 同形状）。

    Returns:
        :class:`AmplitudeMetrics`。掩膜语义（旧管线）：``max_rel_diff`` 仅计
        ``|theory| > REL_FLOOR`` 元素（无合格元素时 0.0）；主导元为
        ``|theory| > DOMINANT_FLOOR``。

    Raises:
        ValueError: 形状不一致或为空。
    """
    th = [[complex(v) for v in row] for row in theory_s]
    sim = [[complex(v) for v in row] for row in sim_s]
    if not th or not sim:
        raise ValueError("S 矩阵不得为空")
    if len(th) != len(sim) or any(
        len(r1) != len(r2) for r1, r2 in zip(th, sim, strict=False)
    ):
        raise ValueError("S 形状不一致，拒绝对账")
    diffs: list[float] = []
    rels: list[float] = []
    dom_diffs: list[float] = []
    for row_th, row_sim in zip(th, sim, strict=True):
        for v_th, v_sim in zip(row_th, row_sim, strict=True):
            a, b = abs(v_th), abs(v_sim)
            d = abs(a - b)
            diffs.append(d)
            if a > REL_FLOOR:
                rels.append(d / a)
            if a > DOMINANT_FLOOR:
                dom_diffs.append(d)
    n = len(diffs)
    return AmplitudeMetrics(
        max_abs_diff=max(diffs),
        max_rel_diff=max(rels) if rels else 0.0,
        rms_diff=math.sqrt(sum(d * d for d in diffs) / n),
        dominant_max_abs_diff=max(dom_diffs) if dom_diffs else 0.0,
        n_elements=n,
        n_dominant=len(dom_diffs),
    )


def compare_s_matrix(
    theory: SMatrixCase,
    sim: SMatrixCase,
    *,
    name: str = "compare_s_matrix",
) -> CheckResult:
    """幅度对账：三指标 + 主导元绝对 |Δ|≤0.02 冻结判据。

    先过 :func:`check_same_in_same_out` 契约，不成立直接拒绝（ValueError，拦截）；
    然后 :func:`amplitude_metrics` 计算 max_abs / max_rel / rms 三指标，冻结判据
    ``主导元最大绝对幅度差 ≤ DOMINANT_ABS_TOL`` 出 PASS/FAIL。

    Args:
        theory: 理论核产出的 S 矩阵。
        sim: COMSOL 模板产出的 S 矩阵。
        name: 检查名。

    Returns:
        :class:`CheckResult`；``value`` = 主导元最大绝对幅度差（冻结判据读数），
        ``message`` 内含三指标全读数，可直接贴交付说明。

    Raises:
        ValueError: 同入同出契约不成立（拦截），或矩阵无主导元
            （``|S_th| > DOMINANT_FLOOR`` 一个不剩，冻结判据无定义）。
    """
    contract = check_same_in_same_out(theory, sim)
    if not contract.ok:
        raise ValueError(f"同入同出契约不成立，拒绝对账：{contract.message}")
    m = amplitude_metrics(theory.s, sim.s)
    if m.n_dominant == 0:
        raise ValueError(
            f"无主导元（|S_theory| > {DOMINANT_FLOOR} 一个不剩），冻结判据无定义"
        )
    ok = m.dominant_max_abs_diff <= DOMINANT_ABS_TOL
    return CheckResult(
        name=name,
        ok=ok,
        value=m.dominant_max_abs_diff,
        message=(
            f"{'PASS' if ok else 'FAIL'} {name}：主导元最大幅度差 "
            f"{m.dominant_max_abs_diff:.3g} "
            f"{'<=' if ok else '>'} {DOMINANT_ABS_TOL}（冻结判据）；"
            f"max_abs={m.max_abs_diff:.3g}, max_rel={m.max_rel_diff:.3g}, "
            f"rms={m.rms_diff:.3g}；{m.n_elements} 元素含 {m.n_dominant} 主导元，"
            f"频率 {theory.frequency_hz} Hz，端口 {theory.port_labels}"
        ),
    )


def synthetic_case_pair(
    seed: int,
    *,
    perturbation: float = 0.0,
    frequency_hz: float = 10e9,
    params_digest_value: str = "synthetic",
) -> tuple[SMatrixCase, SMatrixCase]:
    """合成互易 2×2 散射矩阵对（理论基准 + 加扰"仿真"副本）——冒烟/联调专用。

    基准矩阵无耗互易：``S = [[r, t], [t, r]]``，``|r|²+|t|² = 1``，r/t 的相位由
    种子随机（同种子逐比特可复现；``|r|,|t| ∈ (0.38, 0.93)`` 全部为主导元）。
    扰动模型 = 幅度整体缩放 ``(1+p)`` + 小相位抖动（相位不入幅度对账），故主导元
    幅度差解析可预：``Δ ≈ p·|S_ij|``——冒烟判定 PASS/FAIL 可手算核对。

    Args:
        seed: 随机种子（固定即确定性）。
        perturbation: 扰动幅度 p（≥0）；p=0 时两侧逐元素相等。
        frequency_hz: 两侧共同求解频率。
        params_digest_value: 两侧共同参数摘要（同入）。

    Returns:
        ``(theory_case, sim_case)``。

    Raises:
        ValueError: perturbation 为负或非有限数。
    """
    if not (math.isfinite(perturbation) and perturbation >= 0):
        raise ValueError(f"perturbation 须为非负有限数：{perturbation!r}")
    rng = random.Random(seed)
    theta = rng.uniform(math.pi / 8, math.pi / 4)
    phase_r = rng.uniform(0.0, 2.0 * math.pi)
    phase_t = rng.uniform(0.0, 2.0 * math.pi)

    def _unit(phi: float) -> complex:
        return complex(math.cos(phi), math.sin(phi))

    r = math.cos(theta) * _unit(phase_r)
    t = math.sin(theta) * _unit(phase_t)
    base = ((r, t), (t, r))  # 对称 = 互易

    def _perturb(v: complex) -> complex:
        jitter = _unit(perturbation * rng.uniform(-1.0, 1.0))
        return v * (1.0 + perturbation) * jitter

    sim_rows = tuple(tuple(_perturb(v) for v in row) for row in base)
    common = {
        "frequency_hz": frequency_hz,
        "port_labels": ("P1", "P2"),
        "params_digest": params_digest_value,
    }
    theory_case = SMatrixCase(
        s=tuple(tuple(row) for row in base), source="theory", **common
    )
    sim_case = SMatrixCase(s=sim_rows, source="synthetic-sim", **common)
    return theory_case, sim_case
