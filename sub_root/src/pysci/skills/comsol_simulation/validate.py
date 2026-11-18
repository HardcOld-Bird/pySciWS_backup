"""仿真验证函数族：能量守恒 / 收敛阶 / 参考比对（纯函数，NumPy 不依赖）。

工作流位（SKILL.md 验证步）：求解→后处理之后，对**导出的数值**做物理合理性检查。
本模块全部是纯函数——输入普通 float/序列，输出 :class:`CheckResult`，不触碰 mph/COMSOL，
因此机械断言恒跑；把 COMSOL 导出数据喂进来的编排在研究树/调用方。

ADR（2026-11-17）：旧 postprocess.py 的渲染系列移交③域（pyvista）；只保留验证数值核心
（参考比对/守恒/收敛阶）作②域薄胶水——这是研究流程特有口径，社区无对应物。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CheckResult:
    """单条验证结论（判定 + 读数 + 人话）。

    Attributes:
        name: 检查名（如 ``"energy_conservation"``）。
        ok: 是否通过（硬判据；软告警场景由调用方另行标注）。
        value: 主读数（如守恒残差比、收敛阶）；无量纲检查为 None。
        message: 人话结论（含阈值与实测值），直接可贴交付说明。
    """

    name: str
    ok: bool
    value: float | None
    message: str


def relative_error(actual: float, reference: float) -> float:
    """相对误差 ``|a-b|/|b|``（reference=0 时退化为绝对误差）。

    Args:
        actual: 实测值。
        reference: 参考值（解析解/黄金基准/上一级网格）。

    Returns:
        非负相对误差。

    Raises:
        ValueError: 任一输入非有限数。
    """
    for v in (actual, reference):
        if not math.isfinite(v):
            raise ValueError(f"relative_error 需要有限数：{actual!r}, {reference!r}")
    if reference == 0:
        return abs(actual)
    return abs(actual - reference) / abs(reference)


def compare_to_reference(
    values: list[float],
    reference: list[float],
    *,
    rtol: float = 1e-2,
    atol: float = 0.0,
    name: str = "compare_to_reference",
) -> CheckResult:
    """逐点比对实测序列与参考序列（同入同出比对，④域 S 矩阵验证同形态）。

    判据：每点 ``|a-b| <= atol + rtol*|b|`` 全部成立才 PASS。

    Args:
        values: 实测序列。
        reference: 参考序列（与 values 等长）。
        rtol: 相对容差。
        atol: 绝对容差。
        name: 检查名。

    Returns:
        :class:`CheckResult`；``value`` = 最大相对误差。

    Raises:
        ValueError: 序列为空或长度不等。
    """
    if not values or len(values) != len(reference):
        raise ValueError(
            f"比对序列须非空且等长：{len(values)} vs {len(reference)}"
        )
    worst = 0.0
    for i, (a, b) in enumerate(zip(values, reference, strict=True)):
        err = relative_error(a, b)
        if err > worst:
            worst = err
        if abs(a - b) > atol + rtol * abs(b):
            return CheckResult(
                name=name,
                ok=False,
                value=worst,
                message=(
                    f"FAIL {name}：第 {i} 点越界 |{a}-{b}| > "
                    f"{atol}+{rtol}*|{b}|（最大相对误差 {worst:.3g}）"
                ),
            )
    return CheckResult(
        name=name,
        ok=True,
        value=worst,
        message=f"PASS {name}：全部 {len(values)} 点在容差内（最大相对误差 {worst:.3g}）",
    )


def energy_residual(
    incoming: float, outgoing: float, absorbed: float = 0.0
) -> float:
    """能量守恒残差比 ``|Ein - Eout - Eabsorbed| / |Ein|``。

    Args:
        incoming: 入射/输入功率（或能量，单位一致即可）。
        outgoing: 出射/输出功率。
        absorbed: 吸收功率（无吸收过程传 0）。

    Returns:
        残差比（非负）。

    Raises:
        ValueError: 入射为 0 或任一输入非有限数。
    """
    for v in (incoming, outgoing, absorbed):
        if not math.isfinite(v):
            raise ValueError(f"energy_residual 需要有限数：{incoming}, {outgoing}, {absorbed}")
    if incoming == 0:
        raise ValueError("incoming 为 0，残差比无定义")
    return abs(incoming - outgoing - absorbed) / abs(incoming)


def check_energy_conservation(
    incoming: float,
    outgoing: float,
    absorbed: float = 0.0,
    *,
    tol: float = 1e-2,
    name: str = "energy_conservation",
) -> CheckResult:
    """能量守恒硬门：残差比 ≤ tol 才 PASS。

    Args:
        incoming / outgoing / absorbed: 见 :func:`energy_residual`。
        tol: 残差比阈值（默认 1%）。
        name: 检查名。

    Returns:
        :class:`CheckResult`；``value`` = 残差比。
    """
    r = energy_residual(incoming, outgoing, absorbed)
    ok = r <= tol
    return CheckResult(
        name=name,
        ok=ok,
        value=r,
        message=(
            f"{'PASS' if ok else 'FAIL'} {name}：残差比 {r:.3g} "
            f"{'<=' if ok else '>'} {tol}（Ein={incoming}, Eout={outgoing}, Eabs={absorbed}）"
        ),
    )


def convergence_orders(errors: list[float], hs: list[float]) -> list[float]:
    """由逐级网格误差与网格尺寸估收敛阶：``p_i = ln(e_i/e_{i+1}) / ln(h_i/h_{i+1})``。

    Args:
        errors: 由粗到细的误差序列（≥2 项，全部 > 0）。
        hs: 对应网格尺寸（由粗到细严格递减，全部 > 0）。

    Returns:
        ``len(errors)-1`` 个收敛阶读数。

    Raises:
        ValueError: 长度不匹配、含非正数或 hs 非递减。
    """
    if len(errors) != len(hs) or len(errors) < 2:
        raise ValueError("errors 与 hs 须等长且至少 2 级")
    for e in errors:
        if not math.isfinite(e) or e <= 0:
            raise ValueError(f"误差须为正有限数：{e!r}")
    for h in hs:
        if not math.isfinite(h) or h <= 0:
            raise ValueError(f"网格尺寸须为正有限数：{h!r}")
    for i in range(len(hs) - 1):
        if hs[i] <= hs[i + 1]:
            raise ValueError(f"hs 须严格递减：hs[{i}]={hs[i]} >= hs[{i + 1}]={hs[i + 1]}")
    return [
        math.log(errors[i] / errors[i + 1]) / math.log(hs[i] / hs[i + 1])
        for i in range(len(errors) - 1)
    ]


def check_convergence(
    errors: list[float],
    hs: list[float],
    *,
    expected_order: float = 2.0,
    tol: float = 0.3,
    name: str = "convergence",
) -> CheckResult:
    """收敛性硬门：**最细两级**的实测阶与期望阶偏差 ≤ tol 才 PASS。

    Args:
        errors / hs: 见 :func:`convergence_orders`。
        expected_order: 期望收敛阶（线性元 1、二次元 2 等）。
        tol: 阶偏差容差。
        name: 检查名。

    Returns:
        :class:`CheckResult`；``value`` = 最细两级实测阶。
    """
    orders = convergence_orders(errors, hs)
    p = orders[-1]
    ok = abs(p - expected_order) <= tol
    return CheckResult(
        name=name,
        ok=ok,
        value=p,
        message=(
            f"{'PASS' if ok else 'FAIL'} {name}：最细两级实测阶 {p:.3g} "
            f"{'≈' if ok else '≉'} 期望 {expected_order}（容差 ±{tol}）"
        ),
    )
