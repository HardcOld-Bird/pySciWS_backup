"""扫描窗宽的解析归一（backlog 20261028-190254-sim）。

ParCurve2D 扫描线导出（``parmin1=0/parmax1=1/res1=N``）的**采样端点随会话浮动**：可能
缺末点，实测 ``x[-1]-x[0]`` 在 ``a·D·(N-1)/N`` 与 ``a·D`` 之间跳（B1/F0/S1=0.1408668 vs
S2=0.1414214，0.39% 抖动）。以端点差作提取归一会把该抖动带进 S/谱/功率读数，跨配置复跑
窗宽不再逐位一致。

窗宽真值是**解析值 ``a*D``**（数据集 ``exprx = s*a*D``，``s∈[0,1]``）：

- :func:`scan_window` — 解析窗宽 ``a*D``（经模型求值，``a``/``D`` 可为表达式）。
- :func:`assert_sampled_window` — 采样跨度须与解析窗宽在一采样步内一致（错扫描配置→早失败）。

归一一律用解析窗宽；采样跨度只做一致性断言，**绝不**再当归一分母。
"""

from __future__ import annotations

from typing import Any


def scan_window(model: Any) -> float:
    """解析窗宽 ``a*D``（米）——ParCurve2D 数据集 ``exprx=s*a*D`` 的端点差真值。

    Args:
        model: mph.Model（或可 ``evaluate`` 的等价物）。``a``/``D`` 以模型求值取数
            （二者常是表达式，如 ``D = lambda/sqrt(2)``，不解析参数字符串）。

    Returns:
        窗宽（米，float）。
    """
    import numpy as np

    return float(np.ravel(model.evaluate("a*D", unit="m"))[0])


def assert_sampled_window(x: Any, wlen: float) -> None:
    """采样跨度须与解析窗宽 ``a*D`` 在**一个采样步**内一致。

    缺末点（``a·D·(N-1)/N``）本就在一采样步内，放行；偏差更大说明 scan 数据集/参数
    不匹配（如 res 变了、exprx 不是 ``s*a*D``、或读错模型），**早失败**而非静默失真。

    Args:
        x: 采样坐标（一维序列，单调）。
        wlen: :func:`scan_window` 的解析窗宽。

    Raises:
        ValueError: 点数 < 2。
        RuntimeError: 跨度与解析窗宽差超过 1.5 个采样步。
    """
    n = len(x)
    if n < 2:
        raise ValueError(f"采样点数不足：{n}（须 ≥2）")
    span = float(x[-1] - x[0])
    step = span / (n - 1)
    if abs(span - wlen) > 1.5 * step:
        raise RuntimeError(
            f"扫描窗宽与解析 a*D 不符：采样 span={span:.9g} / 解析 {wlen:.9g}"
            f"（差 {abs(span - wlen):.3g} > 1.5 采样步 {step:.3g}）"
            "——核对 scan 数据集的 exprx=s*a*D 与模型参数 a/D"
        )
