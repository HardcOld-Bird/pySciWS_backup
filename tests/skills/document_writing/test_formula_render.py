"""公式渲染管线回归（backlog 20261103-134217-writing）。

钉三件事：①模板禁 standalone（TeXLive 2026 standalone 使 ``equation*``/``\\[`` 不进
数学模式）；②编译**零 error** 验收——报错即抛、PDF/PNG 一律不落盘（误差恢复产物不可
进 deck）；③同一正确源串重渲染必然得到**正确分数结构**（分子在上、分母在下且水平
对齐）。需要 TeX 的用例按 ``settings.tex_ready`` 探针跳过（无 TeX 环境不假红）。
"""

from __future__ import annotations

import pytest

from pysci.skills.document_writing.tools import formula_render as fr
from pysci.skills.document_writing.tools.config import settings

#: 分数源串（正确写法；分子 2π、分母 λ）。
FRAC = r"\begin{equation*} k = \frac{2\pi}{\lambda} \end{equation*}"

#: 最小分数源串（结构判据用 ASCII 字形，抽文本层稳定）。
FRAC_AB = r"\begin{equation*} \frac{a}{b} \end{equation*}"


def test_template_article_amsmath_not_standalone() -> None:
    tex = fr.formula_tex(FRAC)
    assert "\\documentclass" in tex and "article" in tex
    assert "\\usepackage{amsmath}" in tex
    assert "standalone" not in tex, (
        "TeXLive 2026 standalone 使 equation*/\\[ 不进数学模式——禁用，走 article+amsmath"
    )


def test_zero_error_gate_keeps_bad_output_off_disk(tmp_path) -> None:
    """编译报错即抛 FormulaCompileError；PDF/PNG 一律不落盘。"""
    if not settings.tex_ready:
        pytest.skip("TeX 工具链不可用")
    bad = r"\begin{equation*} \frac{1}{ \end{equation*}"  # 未闭合 → 编译错
    pdf = tmp_path / "bad.pdf"
    with pytest.raises(fr.FormulaCompileError):
        fr.build_formula_pdf(bad, pdf)
    assert not pdf.exists(), "误差恢复产物不可落盘"

    png = tmp_path / "bad.png"
    with pytest.raises(fr.FormulaRenderError):
        fr.render_formula_png(bad, png)
    assert not png.exists(), "误差恢复产物不可进 deck"


def test_fraction_structure_correct_and_repeatable(tmp_path) -> None:
    """同一正确源串两次渲染均得正确分数结构：分子 bbox 在分母上方、水平对齐。"""
    if not settings.tex_ready:
        pytest.skip("TeX 工具链不可用")
    import pymupdf

    def frac_boxes(pdf):
        doc = pymupdf.open(str(pdf))
        try:
            words = [w for w in doc[0].get_text("words") if w[4].strip()]
        finally:
            doc.close()
        by = {w[4]: w for w in words}
        assert "a" in by and "b" in by, f"分数字形缺失：{sorted(by)}"
        return by["a"], by["b"]

    for i in (1, 2):
        a, b = frac_boxes(fr.build_formula_pdf(FRAC_AB, tmp_path / f"frac{i}.pdf"))
        assert a[1] < b[1], (
            f"第 {i} 渲染：分子须在分母上方（a@{a[1]:.1f} vs b@{b[1]:.1f}）"
        )
        assert a[0] < b[2] and b[0] < a[2], f"第 {i} 渲染：分子分母须水平对齐"


def test_render_formula_png_ink_crop_rgba(tmp_path) -> None:
    """成品：ink 裁剪紧致、RGBA 透明底、有墨迹。"""
    if not settings.tex_ready:
        pytest.skip("TeX 工具链不可用")
    import numpy as np
    from PIL import Image

    out = fr.render_formula_png(FRAC, tmp_path / "k.png")
    im = Image.open(out)
    assert im.mode == "RGBA"
    w, h = im.size
    assert 0 < w < 2000 and 0 < h < 600, f"ink 裁剪后应紧致（不含整页白边）：{w}x{h}"
    px = im.load()
    assert px[0, 0][3] == 0 and px[w - 1, h - 1][3] == 0, "四角应为透明（ink 裁剪后）"
    assert (np.asarray(im)[..., 3] > 0).sum() > 100, "墨迹过少——疑似空产物"


# ---------------------------------------------------------------------------
# 渲染图↔源串机械结构自检（backlog 20261103-131952-reviewer）：空分子/错位分组/
# 截断在产出时即拦，不再依赖终检像素取证
# ---------------------------------------------------------------------------
INCIDENT = r"\[ \frac{1}{n_{0}}k_{x} \]"  # 事故源串（p14_f4_t.png 的正确源）


def _synth_pdf(path, *, bar: bool = True, above=None, below=None, side=None):
    """合成单页 PDF：分数杠 + 上/下/侧字形——不依赖 TeX 的判据夹具。"""
    import pymupdf

    doc = pymupdf.open()
    pg = doc.new_page(width=220, height=200)
    if bar:
        pg.draw_line((60, 100), (140, 100), width=1)
    if above:
        pg.insert_text((80, 90), above, fontsize=12)
    if below:
        pg.insert_text((80, 115), below, fontsize=12)
    if side:
        pg.insert_text((170, 50), side, fontsize=12)
    doc.save(str(path))
    doc.close()
    return path


def test_required_chars_strips_commands_keeps_content() -> None:
    assert fr._required_chars(r"\frac{1}{n_{0}}k_{x} \alpha x") == {
        "1": 1,
        "n": 1,
        "0": 1,
        "k": 1,
        "x": 2,
    }


def test_structure_check_passes_correct_layout(tmp_path) -> None:
    """正例（合成）：1 在杠上、n0 在杠下、kx 在侧——结构相符不报。"""
    pdf = _synth_pdf(tmp_path / "ok.pdf", above="1", below="n0", side="kx")
    fr.verify_formula_structure(INCIDENT, pdf)


def test_structure_check_real_render_passes(tmp_path) -> None:
    """正例（真渲染）：事故源串经管线产出必然过自检（build 全链集成）。"""
    if not settings.tex_ready:
        pytest.skip("TeX 工具链不可用")
    fr.build_formula_pdf(INCIDENT, tmp_path / "real.pdf")


def test_structure_check_flags_empty_numerator(tmp_path) -> None:
    """空分子：杠上方窗内无字形（'1' 渲染到了别处）→ 分数层 FAIL。"""
    pdf = _synth_pdf(tmp_path / "empty.pdf", below="n0", side="1")
    with pytest.raises(fr.FormulaStructureError, match="分子"):
        fr.verify_formula_structure(r"\[ \frac{1}{n_{0}} \]", pdf)


def test_structure_check_flags_truncation(tmp_path) -> None:
    """截断丢字：源串 k/x 在产物中缺失 → 字形层 FAIL。"""
    pdf = _synth_pdf(tmp_path / "trunc.pdf", above="1", below="n0")
    with pytest.raises(fr.FormulaStructureError, match="字形层"):
        fr.verify_formula_structure(INCIDENT, pdf)


def test_structure_check_flags_swallowed_follow_tokens(tmp_path) -> None:
    """错位分组（事故第二症状）：kx 被吞进分母区 → 单杠严格绑定 FAIL。"""
    pdf = _synth_pdf(tmp_path / "swallow.pdf", above="1", below="n0kx")
    with pytest.raises(fr.FormulaStructureError, match="分母"):
        fr.verify_formula_structure(INCIDENT, pdf)


def test_structure_check_flags_missing_fraction_bar(tmp_path) -> None:
    """错位分组丢杠：\\frac 在产物中没有分数杠 → 分数层 FAIL。"""
    pdf = _synth_pdf(tmp_path / "nobar.pdf", bar=False, above="1", below="2")
    with pytest.raises(fr.FormulaStructureError, match="分数杠"):
        fr.verify_formula_structure(r"\[ \frac{1}{2} \]", pdf)


def test_structure_check_overline_caveat_no_false_positive(tmp_path) -> None:
    """横线命令（\\overline/\\sqrt 族）使杠不可辨 → 分数层降级跳过，不误报。"""
    pdf = _synth_pdf(tmp_path / "over.pdf", below="y12")
    fr.verify_formula_structure(r"\[ \overline{y} + \frac{1}{2} \]", pdf)


def test_strict_binding_skipped_for_multi_chunk() -> None:
    """多式同页时严格绑定跳过（跨式串扰不可控），宽松层仍兜底。"""
    assert fr._frac_arg_chars(r"\[ x \] \[ \frac{1}{2} \]") is None
