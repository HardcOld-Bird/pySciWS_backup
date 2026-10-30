"""``scan_window`` 行为钉子：解析窗宽 ``a*D`` 归一 + 采样跨度一致性断言。

backlog 20261028-190254-sim：ParCurve2D 导出端点随会话浮动（实测 255/256 情形
``x[-1]-x[0]=a·D·255/256`` vs ``a·D``，0.39% 抖动），端点差作归一分母会把抖动带进
S 提取。钉三件事：解析窗宽经模型求值取数、缺末点放行（一采样步内）、错扫描配置早失败。
"""

from __future__ import annotations

import numpy as np
import pytest

from pysci.skills.comsol_simulation.tools.scan_window import (
    assert_sampled_window,
    scan_window,
)


class _FakeModel:
    """冒充 mph.Model：只实现 ``evaluate``（窗宽表达式恒 ``a*D``）。"""

    def __init__(self, wlen: float):
        self.wlen = wlen
        self.calls: list[str] = []

    def evaluate(self, expr, unit=None):
        self.calls.append(str(expr))
        return np.array([self.wlen])


def test_scan_window_reads_analytic_a_times_d():
    m = _FakeModel(0.1414213562373095)
    assert scan_window(m) == pytest.approx(0.1414213562373095)
    assert m.calls == ["a*D"], (
        "窗宽真值只认解析表达式 a*D（a/D 可为表达式，不解析字符串）"
    )


def test_assert_accepts_missing_last_sample():
    """缺末点（``a·D·(N-1)/N``）在一采样步内 → 放行——正是实测的 255/256 情形。"""
    d = 0.07071067811865475
    full = np.linspace(0.0, 2 * d, 257)
    assert_sampled_window(full, 2 * d)
    assert_sampled_window(full[:-1], 2 * d)  # 255/256 跨度亦放行


def test_assert_rejects_wrong_scan_config():
    """跨度差超过 1.5 采样步（如 exprx 非 s*a*D / res 变了 / 读错模型）→ 早失败。"""
    d = 0.07071067811865475
    x = np.linspace(0.0, 1.5 * d, 129)
    with pytest.raises(RuntimeError, match="解析 a\\*D 不符"):
        assert_sampled_window(x, 2 * d)


def test_assert_rejects_too_few_points():
    with pytest.raises(ValueError):
        assert_sampled_window(np.array([0.1]), 0.2)
