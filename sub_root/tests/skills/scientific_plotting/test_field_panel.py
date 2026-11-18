"""truthful colorbar 场面板机械断言：端标取整 v2 阶梯（固定样例）+ CSV 读取。"""

from __future__ import annotations

from importlib.util import find_spec

import pytest

from pysci.skills.scientific_plotting import field_panel

_HAS_MPL = find_spec("matplotlib") is not None


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        (65.48, 70),  # 6.548e1 → 7.0e1
        (1.4655, 1.5),
        (1.8277, 2.0),
        (2.11, 2.5),
        (1.1968, 1.2),  # d1=1 另含 d2∈{2,4,6,8}
        (0.6436, 0.65),
        (0.98, 1.0),
        (9.7, 10),
        (0.151, 0.16),
    ],
)
def test_ceil_label_v2_ladder(x, expected):
    assert field_panel.ceil_label(x) == expected


@pytest.mark.parametrize("x", [0.1, 1.0, 1.2, 1.5, 2.5, 6.5, 10.0, 0.15])
def test_ceil_label_legal_values_unchanged(x):
    assert field_panel.ceil_label(x) == x


def test_ceil_label_rejects_bad_input():
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="正有限数"):
            field_panel.ceil_label(bad)


def test_read_csv_xyv(tmp_path):
    src = tmp_path / "field.csv"
    src.write_text("x,y,v\n0,0,1.5\n1 0 2.5\n1,1,3.5\n# comment\n", encoding="utf-8")
    xs, ys, vals = field_panel.read_csv_xyv(src)
    assert (xs, ys, vals) == ([0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [1.5, 2.5, 3.5])


def test_read_csv_xyv_rejects_empty(tmp_path):
    src = tmp_path / "empty.csv"
    src.write_text("x,y,v\n", encoding="utf-8")
    with pytest.raises(ValueError, match="无有效"):
        field_panel.read_csv_xyv(src)


@pytest.mark.hardware
@pytest.mark.skipif(not _HAS_MPL, reason="matplotlib 未安装")
def test_plot_field_shares_norm_with_colorbar(tmp_path):
    """truthful 断言：colorbar 的 norm/cmap 与面板着色同源，vmax 端标取整。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [0.0, 1.0, 0.0, 1.0]
    ys = [0.0, 0.0, 1.0, 1.0]
    vals = [0.0, 0.5, 1.0, 65.48]
    fig, ax = plt.subplots()
    m = field_panel.plot_field(ax, xs, ys, vals, cmap="viridis")
    cbar = field_panel.add_truthful_colorbar(fig, m, ax=ax)
    assert cbar.norm is m.norm
    assert m.norm.vmax == 70  # ceil_label(65.48) = 70，绝不裁峰
    plt.close(fig)
