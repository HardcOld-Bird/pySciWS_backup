"""面板标号机械断言：标号文本生成（纯函数）+ 轴上放置（hardware 渲染）。"""

from __future__ import annotations

from importlib.util import find_spec

import pytest

from pysci.skills.scientific_plotting import labels

_HAS_MPL = find_spec("matplotlib") is not None


def test_panel_label_text_paren_default():
    assert labels.panel_label_text(0) == "(a)"
    assert labels.panel_label_text(2) == "(c)"
    assert labels.panel_label_text(25) == "(z)"


def test_panel_label_text_styles():
    assert labels.panel_label_text(1, style="plain") == "b"
    assert labels.panel_label_text(1, style="paren_after") == "b)"


def test_panel_label_text_rejects_bad_index():
    with pytest.raises(ValueError, match="越界"):
        labels.panel_label_text(26)
    with pytest.raises(ValueError, match="越界"):
        labels.panel_label_text(-1)


def test_panel_label_text_rejects_bad_style():
    with pytest.raises(ValueError, match="风格"):
        labels.panel_label_text(0, style="bracket")


@pytest.mark.hardware
@pytest.mark.skipif(not _HAS_MPL, reason="matplotlib 未安装")
def test_label_panel_places_bold_text_outside_axes():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    artist = labels.label_panel(ax, 0)
    assert artist.get_text() == "(a)"
    assert artist.get_fontweight() == "bold"
    assert artist.get_position() == pytest.approx((-0.02, 1.02))
    assert artist.get_transform() == ax.transAxes
    plt.close(fig)
