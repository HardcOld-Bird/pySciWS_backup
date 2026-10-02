"""theoretical_computation 发散安全绘图 / 奇点感知采样原语单测（P1-3）。

覆盖：numerical.adaptive_sample（奇点邻域加密但不过采样、不触及奇点）与
visualize.mask_divergent / quick_plot_divergent（超窗 nan 遮断断线，优于 clip 饱和）。
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from pysci.skills.theoretical_computation.tools import (  # noqa: E402
    numerical,
    visualize,
)


# ---------------------------------------------------------------------------
# adaptive_sample（奇点感知）
# ---------------------------------------------------------------------------
def test_adaptive_sample_no_singularities_is_uniform():
    x = numerical.adaptive_sample((0.0, 1.0), base_resolution=50)
    assert np.allclose(x, np.linspace(0.0, 1.0, 50))


def test_adaptive_sample_refines_near_singularity_without_touching():
    x = numerical.adaptive_sample(
        (0.0, 1.0), singularities=[0.5], base_resolution=50, approach_points=8
    )
    assert 0.5 not in x  # 不在奇点本身采样
    assert np.all(np.diff(x) > 0)  # 升序去重
    near = x[(x > 0.45) & (x < 0.55)]
    base = np.linspace(0.0, 1.0, 50)
    base_near = base[(base > 0.45) & (base < 0.55)]
    assert len(near) > len(base_near)  # 奇点邻域确实加密


def test_adaptive_sample_ignores_out_of_range_singularities():
    x = numerical.adaptive_sample(
        (0.0, 1.0), singularities=[-5.0, 5.0], base_resolution=30
    )
    assert np.allclose(x, np.linspace(0.0, 1.0, 30))


def test_adaptive_sample_degenerate_range_raises():
    with pytest.raises(ValueError):
        numerical.adaptive_sample((1.0, 1.0))


# ---------------------------------------------------------------------------
# mask_divergent / quick_plot_divergent（nan 遮断）
# ---------------------------------------------------------------------------
def test_mask_divergent_nan_out_of_window():
    y = np.array([0.0, 5.0, 100.0, -100.0, np.nan, np.inf])
    m = visualize.mask_divergent(y, (-10.0, 10.0))
    assert m[0] == 0.0 and m[1] == 5.0  # 窗内保留
    assert np.isnan(m[2]) and np.isnan(m[3])  # 超窗置 nan
    assert np.isnan(m[4]) and np.isnan(m[5])  # nan/inf 置 nan


def test_mask_divergent_complex_takes_real():
    y = np.array([1.0 + 2.0j, 100.0 + 0.0j])
    m = visualize.mask_divergent(y, (-10.0, 10.0))
    assert m[0] == 1.0 and np.isnan(m[1])


def test_quick_plot_divergent_returns_figure():
    x = np.linspace(0.0, 1.0, 60)
    y = 1.0 / (x - 0.5)  # 奇点发散
    fig = visualize.quick_plot_divergent(x, y, ylim=(-20.0, 20.0), title="divergent")
    assert isinstance(fig, Figure)
    plt.close(fig)
