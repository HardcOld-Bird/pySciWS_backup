"""divergence_safety 机械断言：旧 7 项发散安全测试期望（对照期望值重写，未迁移旧代码）。

期望来源：旧 pySciWS ``tests/skills/theoretical_computation/test_divergent_primitives.py``
（①无奇点=均匀网格 ②奇点邻域加密且不触奇点 ③区间外奇点忽略 ④退化区间报错
⑤窗外/NaN/Inf 置 NaN ⑥复数取实部 ⑦发散绘图返回 Figure）。
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from pysci.skills.theory_calculation import divergence_safety  # noqa: E402


# ── ① 无奇点：退化为均匀 linspace ─────────────────────────────────
def test_adaptive_sample_no_singularities_is_uniform():
    x = divergence_safety.adaptive_sample_1d((0.0, 1.0), base_resolution=50)
    assert np.allclose(x, np.linspace(0.0, 1.0, 50))


# ── ② 奇点邻域加密、不触奇点、严格升序去重 ────────────────────────
def test_adaptive_sample_refines_near_singularity_without_touching():
    x = divergence_safety.adaptive_sample_1d(
        (0.0, 1.0), singularities=[0.5], base_resolution=50, approach_points=8
    )
    assert 0.5 not in x  # 不在奇点本身采样
    assert np.all(np.diff(x) > 0)  # 严格升序（去重后）
    near = x[(x > 0.45) & (x < 0.55)]
    base = np.linspace(0.0, 1.0, 50)
    base_near = base[(base > 0.45) & (base < 0.55)]
    assert len(near) > len(base_near)  # 奇点邻域确实加密


# ── ③ 区间外奇点忽略 ─────────────────────────────────────────────
def test_adaptive_sample_ignores_out_of_range_singularities():
    x = divergence_safety.adaptive_sample_1d(
        (0.0, 1.0), singularities=[-5.0, 5.0], base_resolution=30
    )
    assert np.allclose(x, np.linspace(0.0, 1.0, 30))


# ── ④ 退化区间报错 ───────────────────────────────────────────────
def test_adaptive_sample_degenerate_range_raises():
    with pytest.raises(ValueError):
        divergence_safety.adaptive_sample_1d((1.0, 1.0))


# ── ⑤ 窗外与 NaN/Inf 置 NaN ──────────────────────────────────────
def test_mask_divergent_nan_out_of_window():
    y = np.array([0.0, 5.0, 100.0, -100.0, np.nan, np.inf])
    m = divergence_safety.mask_divergent(y, (-10.0, 10.0))
    assert m[0] == 0.0 and m[1] == 5.0  # 窗内保留
    assert np.isnan(m[2]) and np.isnan(m[3])  # 超窗置 nan
    assert np.isnan(m[4]) and np.isnan(m[5])  # nan/inf 置 nan


# ── ⑥ 复数取实部后再判窗 ─────────────────────────────────────────
def test_mask_divergent_complex_takes_real():
    y = np.array([1.0 + 2.0j, 100.0 + 0.0j])
    m = divergence_safety.mask_divergent(y, (-10.0, 10.0))
    assert m[0] == 1.0 and np.isnan(m[1])


# ── ⑦ 发散绘图返回 Figure ────────────────────────────────────────
def test_quick_plot_divergent_returns_figure():
    x = np.linspace(0.0, 1.0, 60)
    y = 1.0 / (x - 0.5)  # 奇点发散
    fig = divergence_safety.quick_plot_divergent(
        x, y, ylim=(-20.0, 20.0), title="divergent"
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


# ── 补充边界（超出 7 项的最小护栏）────────────────────────────────
def test_mask_divergent_rejects_reversed_window():
    with pytest.raises(ValueError):
        divergence_safety.mask_divergent(np.array([0.0]), (10.0, -10.0))


def test_quick_plot_divergent_auto_window_masks_nonfinite():
    x = np.linspace(0.0, 1.0, 20)
    y = np.where(x == x[3], np.inf, x)  # 仅非有限值被遮断
    fig = divergence_safety.quick_plot_divergent(x, y)
    line = fig.axes[0].get_lines()[0]
    masked = line.get_ydata()
    assert np.isnan(masked[3]) and masked[0] == x[0]
    plt.close(fig)
