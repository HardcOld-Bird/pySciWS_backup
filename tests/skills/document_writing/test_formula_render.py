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
