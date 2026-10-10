"""audit 交付件宽度假绿防护 + STYLE.yaml 逐键继承的回归测试。

背景（backlog 20261009-audit-width-masked）：audit 的宽度判定此前只比较「内存重绘图」
与其自身目标宽度——二者同源于 eff_width，恒等，永远 PASS。研究线根 STYLE.yaml 的
``width: double`` 会让单栏图（85mm 交付件）按 170mm 目标假绿。此处断言：

- ``_measure_deliverable_width_mm`` 能从 EPS/PDF 量取实际宽度、对缺失/非矢量返回 None；
- ``audit_figure(deliverable_width_mm=...)`` 与目标偏差过大 → WARN ``width-deliverable``；
- ``load_style_config`` **逐键合并**（根级基线 + 图目录覆盖），不再是「取第一个文件」；
- 端到端 ``audit_figure_dir``：单栏交付件 + 根级 double 且无图目录 STYLE.yaml → WARN
  （复现假绿）；图目录声明 width:single（逐键继承）→ 判定正确、无 WARN。
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from pysci.skills.scientific_plotting.tools import audit, runner  # noqa: E402
from pysci.skills.scientific_plotting.tools.export import save_figure  # noqa: E402
from pysci.skills.scientific_plotting.tools.style import style_context  # noqa: E402

_PIPELINE_SRC = """\
import matplotlib.pyplot as plt


def build_figure(style=None, **kwargs):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return fig
"""


@pytest.fixture
def fig_roots(tmp_path, monkeypatch):
    """重定向 src 代码侧根到 tmp，数据侧含 ``research/<n>_<name>/`` 段以供路径反推。"""
    data_figures = tmp_path / "data" / "research" / "1_gain_ep" / "article" / "figures"
    src_figures = (
        tmp_path / "src" / "pysci" / "research" / "gain_ep" / "article" / "figures"
    )
    data_figures.mkdir(parents=True)
    src_figures.mkdir(parents=True)
    monkeypatch.setattr(runner, "figures_code_root", lambda research: src_figures)
    return {"data": data_figures, "src": src_figures, "tmp": tmp_path}


def _make_pipeline(roots, slug):
    """建 figdir（含 out/）+ src 侧真实管线模块，返回 figdir。"""
    fd = roots["data"] / slug
    (fd / "out").mkdir(parents=True, exist_ok=True)
    (roots["src"] / f"{slug}.py").write_text(_PIPELINE_SRC, encoding="utf-8")
    return fd


def _render_deliverable(figdir, *, width):
    """在给定 width 下渲染并导出 EPS 交付件到 out/<figdir.name>.eps（模拟一次 build）。"""
    stem = figdir.name
    with style_context("aps", width=width):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        res = save_figure(fig, figdir / "out", stem, formats=("eps",), close=True)
    return res.deliverables["eps"]


# ---------------------------------------------------------------------------
# _measure_deliverable_width_mm / _find_deliverable
# ---------------------------------------------------------------------------
def test_measure_eps_deliverable_width(tmp_path):
    with style_context("aps", width="single"):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        res = save_figure(fig, tmp_path, "m", formats=("eps",), close=True)
    eps = res.deliverables["eps"]
    w = audit._measure_deliverable_width_mm(eps)
    assert w is not None
    # 紧裁后窄于 85mm 设计宽度，但数量级正确（远非 170mm 双栏）
    assert 50.0 < w < 85.0


def test_measure_missing_or_non_vector_returns_none(tmp_path):
    assert audit._measure_deliverable_width_mm(tmp_path / "nope.eps") is None
    png = tmp_path / "x.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    assert audit._measure_deliverable_width_mm(png) is None


def test_measure_pdf_mediabox(tmp_path):
    p = tmp_path / "t.pdf"
    p.write_text("%PDF-1.4\n/MediaBox [0 0 481.89 300.00]\n", encoding="latin-1")
    w = audit._measure_deliverable_width_mm(p)
    assert w is not None
    assert abs(w - 481.89 * 25.4 / 72) < 0.01  # ≈170mm（双栏）


def test_find_deliverable_prefers_eps(tmp_path):
    (tmp_path / "s.eps").write_text("%%BoundingBox: 0 0 100 100\n", encoding="utf-8")
    (tmp_path / "s.pdf").write_text("/MediaBox [0 0 100 100]\n", encoding="utf-8")
    assert audit._find_deliverable(tmp_path, "s").name == "s.eps"
    (tmp_path / "s.eps").unlink()
    assert audit._find_deliverable(tmp_path, "s").name == "s.pdf"
    assert audit._find_deliverable(tmp_path, "missing") is None


# ---------------------------------------------------------------------------
# audit_figure 交付件宽度比对
# ---------------------------------------------------------------------------
def _fig():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    return fig


def test_audit_figure_warns_on_deliverable_width_mismatch():
    fig = _fig()
    try:
        rep = audit.audit_figure(
            fig,
            target_width_mm=170.0,
            deliverable_width_mm=77.0,
            check_colorblind=False,
        )
    finally:
        plt.close(fig)
    codes = [i.code for i in rep.issues]
    assert "width-deliverable" in codes
    assert rep.metrics["deliverable_width_mm"] == 77.0
    # 交付件偏差是 WARN（非致命），不使 ok 变 False
    assert all(
        i.level == audit.WARN for i in rep.issues if i.code == "width-deliverable"
    )


def test_audit_figure_no_warn_when_deliverable_matches():
    fig = _fig()
    try:
        rep = audit.audit_figure(
            fig,
            target_width_mm=170.0,
            deliverable_width_mm=143.0,
            check_colorblind=False,
        )
    finally:
        plt.close(fig)
    assert "width-deliverable" not in [i.code for i in rep.issues]


def test_audit_figure_deliverable_none_skips_check():
    fig = _fig()
    try:
        rep = audit.audit_figure(
            fig,
            target_width_mm=170.0,
            deliverable_width_mm=None,
            check_colorblind=False,
        )
    finally:
        plt.close(fig)
    assert "width-deliverable" not in [i.code for i in rep.issues]
    assert "deliverable_width_mm" not in rep.metrics


# ---------------------------------------------------------------------------
# load_style_config 逐键继承（Fix B）
# ---------------------------------------------------------------------------
def test_load_style_config_merges_key_by_key(tmp_path):
    root = tmp_path / "figures"
    fd = root / "figA"
    fd.mkdir(parents=True)
    (root / "STYLE.yaml").write_text(
        "style: aps\nwidth: double\npalette: okabe-ito\n", encoding="utf-8"
    )
    (fd / "STYLE.yaml").write_text("width: single\n", encoding="utf-8")
    cfg = runner.load_style_config(fd)
    # 图目录只声明 width，其余继承根级
    assert cfg == {"style": "aps", "width": "single", "palette": "okabe-ito"}


def test_load_style_config_no_figdir_file_uses_root(tmp_path):
    root = tmp_path / "figures"
    fd = root / "figB"
    fd.mkdir(parents=True)
    (root / "STYLE.yaml").write_text("style: nature\nwidth: double\n", encoding="utf-8")
    cfg = runner.load_style_config(fd)
    assert cfg == {"style": "nature", "width": "double"}


def test_load_style_config_only_figdir_file(tmp_path):
    root = tmp_path / "figures"
    fd = root / "figC"
    fd.mkdir(parents=True)
    (fd / "STYLE.yaml").write_text("width: single\n", encoding="utf-8")
    assert runner.load_style_config(fd) == {"width": "single"}


def test_load_style_config_none(tmp_path):
    fd = tmp_path / "figures" / "figD"
    fd.mkdir(parents=True)
    assert runner.load_style_config(fd) == {}


# ---------------------------------------------------------------------------
# 端到端：audit_figure_dir 复现并修复假绿
# ---------------------------------------------------------------------------
def test_e2e_single_deliverable_masked_by_root_double_warns(fig_roots):
    """根级 width:double + 单栏交付件 + 无图目录 STYLE.yaml → 判定目标 170、交付件 ~77 → WARN。"""
    fd = _make_pipeline(fig_roots, "fig_masked")
    (fig_roots["data"] / "STYLE.yaml").write_text(
        "style: aps\nwidth: double\n", encoding="utf-8"
    )
    _render_deliverable(fd, width="single")  # 实际交付件是单栏 85mm(紧裁~77)

    rep = audit.audit_figure_dir(fd)
    assert rep.metrics["target_width_mm"] == 170.0
    assert rep.metrics["deliverable_width_mm"] < 100.0
    assert "width-deliverable" in [i.code for i in rep.issues]


def test_e2e_figdir_width_single_inherits_and_passes(fig_roots):
    """图目录声明 width:single（逐键继承根级 style）→ 目标 85、交付件 ~77 → 无 WARN。"""
    fd = _make_pipeline(fig_roots, "fig_fixed")
    (fig_roots["data"] / "STYLE.yaml").write_text(
        "style: aps\nwidth: double\n", encoding="utf-8"
    )
    (fd / "STYLE.yaml").write_text("width: single\n", encoding="utf-8")
    _render_deliverable(fd, width="single")

    rep = audit.audit_figure_dir(fd)
    assert rep.metrics["target_width_mm"] == 85.0
    assert "width-deliverable" not in [i.code for i in rep.issues]


def test_e2e_matching_double_deliverable_passes(fig_roots):
    """根级 double + 双栏交付件 → 目标 170、交付件 ~143 → 无 WARN（正常紧裁不误报）。"""
    fd = _make_pipeline(fig_roots, "fig_double_ok")
    (fig_roots["data"] / "STYLE.yaml").write_text(
        "style: aps\nwidth: double\n", encoding="utf-8"
    )
    _render_deliverable(fd, width="double")

    rep = audit.audit_figure_dir(fd)
    assert rep.metrics["target_width_mm"] == 170.0
    assert "width-deliverable" not in [i.code for i in rep.issues]


def test_e2e_no_deliverable_skips_check(fig_roots):
    """从未 build（out/ 无交付件）→ 跳过交付件检查，不误报。"""
    fd = _make_pipeline(fig_roots, "fig_unbuilt")
    rep = audit.audit_figure_dir(fd)
    assert "width-deliverable" not in [i.code for i in rep.issues]
    assert "deliverable_width_mm" not in rep.metrics
