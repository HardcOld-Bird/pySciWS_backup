"""导出封装机械断言：savefig 参数拼装（纯函数）+ 尺寸契约施加 + 真实渲染（hardware）。"""

from __future__ import annotations

from importlib.util import find_spec

import pytest

from pysci.skills.scientific_plotting import export

_HAS_MPL = find_spec("matplotlib") is not None


def test_build_savefig_kwargs_png():
    kw = export.build_savefig_kwargs("png", dpi=600, transparent=True)
    assert kw == {"format": "png", "dpi": 600, "transparent": True}


def test_build_savefig_kwargs_pdf_deterministic():
    kw = export.build_savefig_kwargs("pdf")
    assert kw["format"] == "pdf"
    assert kw["metadata"]["CreationDate"] is None  # 去时间戳 → 字节级确定性


def test_build_savefig_kwargs_svg_keeps_text():
    kw = export.build_savefig_kwargs("svg")
    assert kw["format"] == "svg"
    assert kw["metadata"]["Date"] is None


def test_build_savefig_kwargs_eps():
    assert export.build_savefig_kwargs("eps") == {
        "format": "eps",
        "metadata": {"Creator": "pysci"},
    }


def test_build_savefig_kwargs_case_insensitive():
    assert export.build_savefig_kwargs("PDF")["format"] == "pdf"


def test_build_savefig_kwargs_unknown_format():
    with pytest.raises(ValueError, match="不支持的导出格式"):
        export.build_savefig_kwargs("gif")


def test_export_formats_is_publication_set():
    assert export.EXPORT_FORMATS == ("pdf", "eps", "svg", "png")


class _FakeFig:
    """鸭子类型 Figure 假件（尺寸契约施加不依赖 matplotlib）。"""

    def __init__(self, w_in: float = 6.4, h_in: float = 4.8):
        self._w, self._h = w_in, h_in

    def get_size_inches(self):
        return (self._w, self._h)

    def set_size_inches(self, w: float, h: float):
        self._w, self._h = w, h


def test_fit_figure_size_by_contract():
    fig = _FakeFig()
    w_mm, h_mm = export.fit_figure_size(fig, journal="aps", span="double")
    assert w_mm == pytest.approx(170.0)
    assert h_mm == pytest.approx(4.8 * 25.4)  # 高度未给 → 保持


def test_fit_figure_size_explicit_overrides():
    fig = _FakeFig()
    w_mm, _ = export.fit_figure_size(fig, width_mm=85.0, height_mm=50.0)
    assert w_mm == pytest.approx(85.0)


def test_fit_figure_size_requires_width_info():
    with pytest.raises(ValueError, match="width_mm"):
        export.fit_figure_size(_FakeFig())


@pytest.mark.hardware
@pytest.mark.skipif(not _HAS_MPL, reason="matplotlib 未安装")
def test_export_figure_writes_all_formats_and_preview(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pysci.skills.scientific_plotting import styles as st

    with st.style_context("house", "journal-aps"):
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [0, 1, 0])
        out = export.export_figure(
            fig,
            tmp_path / "fig1",
            journal="aps",
            span="double",
            guard_data=False,  # 测试产物落 tmp（项目常态落 data/，护栏默认开）
        )
    plt.close(fig)
    names = {p.name for p in out}
    assert names == {
        "fig1.pdf",
        "fig1.eps",
        "fig1.svg",
        "fig1.png",
        "fig1_preview.png",
    }
    for p in out:
        assert p.stat().st_size > 0
    # 字体契约：PDF FontType 42、SVG 保留 <text>
    assert b"/FontFile2" in (tmp_path / "fig1.pdf").read_bytes()
    assert b"<text" in (tmp_path / "fig1.svg").read_bytes()
