"""轻量 audit 机械断言：五项判定（固定样例）+ 报告语义 + 文件后缀分发 + Figure 自检。"""

from __future__ import annotations

from importlib.util import find_spec

from pysci.skills.scientific_plotting import audit
from pysci.skills.scientific_plotting.styles import HOUSE_CYCLE

_HAS_MPL = find_spec("matplotlib") is not None


# ── width ──────────────────────────────────────────────────────────────────


def test_check_width_within_tolerance_passes():
    f = audit.check_width(170.8, 170.0)
    assert f.status == "pass"


def test_check_width_beyond_tolerance_fails():
    f = audit.check_width(165.0, 170.0)
    assert f.status == "fail"
    assert "tight bbox" in f.detail


def test_check_width_missing_evidence_is_na():
    assert audit.check_width(None, 170.0).status == "na"
    assert audit.check_width(170.0, None).status == "na"


# ── fontsize ───────────────────────────────────────────────────────────────


def test_check_min_font_pass_and_fail():
    assert audit.check_min_font([6.0, 8.0]).status == "pass"
    f = audit.check_min_font([8.0, 4.5])
    assert f.status == "fail"
    assert "[4.5]" in f.detail


def test_check_min_font_empty_is_na():
    assert audit.check_min_font([]).status == "na"


# ── font embedding ─────────────────────────────────────────────────────────


def test_check_pdf_fonttype():
    assert audit.check_pdf_fonttype(b"x /FontFile2 y").status == "pass"
    f = audit.check_pdf_fonttype(b"x /Font /Foo y")
    assert f.status == "fail"
    assert "fonttype=42" in f.detail
    assert audit.check_pdf_fonttype(b"% no fonts here").status == "na"


def test_check_eps_fonttype():
    assert audit.check_eps_fonttype(b"x /sfnts y").status == "pass"
    f3 = audit.check_eps_fonttype(b"x /CharStrings y")
    assert f3.status == "fail"
    assert "Type3" in f3.detail
    f_res = audit.check_eps_fonttype(b"%%BeginResource: font Foo")
    assert f_res.status == "fail"
    assert audit.check_eps_fonttype(b"%!PS-Adobe").status == "na"


def test_check_svg_text():
    assert audit.check_svg_text(b"<svg><text>a</text></svg>").status == "pass"
    f = audit.check_svg_text(b"<svg><path d='...'/></svg>")
    assert f.status == "fail"
    assert "svg.fonttype=none" in f.detail
    assert audit.check_svg_text(b"<svg/>").status == "na"


# ── panel labels ───────────────────────────────────────────────────────────


def test_check_panel_label_count_match_and_mismatch():
    ok = audit.check_panel_label_count(2, ["(a)", "(b)", "title"])
    assert ok.status == "pass"
    bad = audit.check_panel_label_count(3, ["(a)", "(b)"])
    assert bad.status == "fail"
    assert "面板 3 个" in bad.detail


def test_check_panel_label_count_no_panels_is_na():
    assert audit.check_panel_label_count(0, []).status == "na"


# ── color cycle ────────────────────────────────────────────────────────────


def test_check_color_cycle_exact_and_mismatch():
    assert audit.check_color_cycle(HOUSE_CYCLE).status == "pass"
    # hex 归一化（大小写/# 前缀）后仍一致
    loose = [h.lstrip("#") for h in HOUSE_CYCLE]
    assert audit.check_color_cycle(loose).status == "pass"
    f = audit.check_color_cycle(tuple(reversed(HOUSE_CYCLE)))
    assert f.status == "fail"
    assert audit.check_color_cycle([]).status == "na"


# ── report / dispatch ──────────────────────────────────────────────────────


def test_audit_report_passed_semantics():
    good = audit.AuditReport(
        (
            audit.Finding("width", "pass", "ok"),
            audit.Finding("fontsize", "na", "依据不足"),
        )
    )
    assert good.passed
    assert "[PASS]" in good.format()
    bad = audit.AuditReport((audit.Finding("width", "fail", "偏"),))
    assert not bad.passed
    assert "[FAIL]" in bad.format()


def test_audit_bytes_dispatch(tmp_path):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF /FontFile2")
    assert audit.audit_bytes(pdf)[0].status == "pass"
    png = tmp_path / "a.png"
    png.write_bytes(b"\x89PNG")
    assert audit.audit_bytes(png)[0].status == "na"
    odd = tmp_path / "a.weird"
    odd.write_bytes(b"?")
    assert audit.audit_bytes(odd)[0].status == "na"


def _fake_fig():
    class Text:  # 类名参与 audit 的鸭子类型判定（Text/Annotation）
        def __init__(self, text, size):
            self._t, self._s = text, size

        def get_text(self):
            return self._t

        def get_fontsize(self):
            return self._s

    class _Ax:
        def __init__(self, visible, label, data=True):
            self._v, self._l, self._d = visible, label, data

        def get_visible(self):
            return self._v

        def get_label(self):
            return self._l

        def has_data(self):
            return self._d

    class _Fig:
        def get_size_inches(self):
            return (170.0 / 25.4, 100.0 / 25.4)

        def findobj(self):
            return [
                Text("(a)", 7.0),
                Text("(b)", 7.0),
                Text("title", 8.0),
                Text("", 7.0),  # 空文本不计
            ]

        # 2 个可见数据面板轴 + 1 个 colorbar 轴（排除）+ 1 个不可见轴（排除）
        # + 1 个可见无数据轴（ultraplot 图例/caption 自动轴，排除）
        axes = [
            _Ax(True, "ax1"),
            _Ax(True, "ax2"),
            _Ax(True, "<colorbar>"),
            _Ax(False, "ax3"),
            _Ax(True, "subplot_3", data=False),
        ]

    return _Fig()


def test_audit_figure_on_duck_figure():
    report = audit.audit_figure(
        _fake_fig(),
        expected_width_mm=170.0,
        cycle_colors=HOUSE_CYCLE,
    )
    assert report.passed, report.format()
    checks = {f.check: f.status for f in report.findings}
    assert checks == {
        "width": "pass",
        "fontsize": "pass",
        "panel-labels": "pass",
        "color-cycle": "pass",
    }


def test_audit_figure_no_evidence_is_na():
    report = audit.audit_figure(_fake_fig())
    checks = {f.check: f.status for f in report.findings}
    assert checks["width"] == "na"
    assert checks["color-cycle"] == "na"
