"""bilingual reorder-suggest 回归（平移旧 pySciWS test_reorder_suggest.py 140 行期望
+ 聚类/映射/切块固定样例扩充）。

核心行为（B 路径，只出建议不改源）：PDF 文本块坐标 → 排版带键序 → 同键归组把图注残段
缝回图注；孤立面板标签归位/剔出；映射不上只单列「未映射」不硬排。全部用合成 md + 合成
PDF 块（离线确定）；真 PDF 提取与驱动落盘用 pymupdf 现造小 PDF（无外部依赖恒跑）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pysci.skills.literature_research.bilingual import reorder as rs

pymupdf = pytest.importorskip("pymupdf", reason="需要 pymupdf（随 pymupdf4llm 提供）")


def _pdf(page: int, band: int, col: float, y0: float, text: str, x0: float = 40.0):
    return rs.PdfBlock(
        page=page,
        x0=x0,
        y0=y0,
        x1=x0 + 250,
        y1=y0 + 30,
        text=text,
        band=band,
        col=col,
    )


def _pipeline(md_text: str, pdf_blocks: list[rs.PdfBlock]):
    mbs = rs.parse_md_blocks(md_text)
    mapped, unmapped = rs.map_blocks(mbs, pdf_blocks)
    keys = rs.suggest_order(mbs, mapped)
    order = sorted(keys, key=lambda i: (keys[i], i))
    stitches = rs._stitch_pairs(order, mbs, keys, mapped)
    report = rs.build_report(order, mbs, keys, mapped, unmapped, stitches)
    suggested = rs.build_suggested_md(order, mbs, stitches)
    return mbs, order, stitches, report, suggested


# ---------------------------------------------------------------------------
# ① 图注残段缝合（跨栏同带：左栏图注头 + 右栏续段）
# ---------------------------------------------------------------------------
def test_caption_fragment_stitched_across_columns() -> None:
    md = (
        "Fig. 1 | Concept of a crystal.\n\n"
        "a Schematic picture of a crystal whose strength is modulated between\n\n"
        "![](images/x.jpg)\n\n"
        "-100Hz to 100 Hz with period Tm.\n"
    )
    pdf = [
        _pdf(
            1,
            0,
            0.0,
            100,
            "Fig. 1 | Concept of a crystal. a Schematic picture "
            "of a crystal whose strength is modulated between",
        ),
        _pdf(1, 0, 1.0, 100, "−100Hz to 100 Hz with period Tm."),
    ]
    _mbs, _order, stitches, report, suggested = _pipeline(md, pdf)

    assert len(stitches) == 2, "两残段都应缝回图注（含跨栏右段）"
    assert report.count("[图注残段缝合]") == 2
    assert "modulated between -100Hz to 100 Hz" in suggested.replace("−", "-")
    assert "modulated between\n\n-100Hz" not in suggested, "残段不得独立成块"


# ---------------------------------------------------------------------------
# ② 孤立面板标签：归位/剔出建议必出，正文不吞
# ---------------------------------------------------------------------------
def test_isolated_panel_label_suggested_not_swallowed() -> None:
    md = "Fig. 3 | Emission spectra.\n\n![](images/a.jpg)\n\nd\n\n![](images/b.jpg)\n"
    pdf = [
        _pdf(2, 0, 0.5, 50, "d"),
        _pdf(2, 1, 0.0, 200, "Fig. 3 | Emission spectra."),
    ]
    _mbs, _order, _st, report, suggested = _pipeline(md, pdf)

    assert "[标签归位或剔出可译单元]" in report
    assert suggested.split("\n\n").count("d") == 1, "标签保持独立块（不缝进图注文字）"
    assert "spectra. d" not in suggested


# ---------------------------------------------------------------------------
# ③ 序推断：md 序与 PDF 排版带序不一致 → 建议序跟 PDF
# ---------------------------------------------------------------------------
def test_order_inference_follows_pdf_bands() -> None:
    md = (
        "The beta dispersion measurement result shows resonance near 4.7 kHz today.\n\n"
        "Above, we have constructed a time-varying metamaterial with gain and loss.\n"
    )
    pdf = [
        _pdf(
            1,
            0,
            0.0,
            100,
            "Above, we have constructed a time-varying metamaterial "
            "with gain and loss.",
        ),
        _pdf(
            1,
            1,
            0.0,
            300,
            "The beta dispersion measurement result shows resonance "
            "near 4.7 kHz today.",
        ),
    ]
    _mbs, order, _st, report, suggested = _pipeline(md, pdf)

    assert suggested.startswith("Above, we have constructed"), (
        "PDF 带序（y 小者先）胜出"
    )
    assert "行（text）" in report and "[移动]" in report
    assert order[0] == 1


# ---------------------------------------------------------------------------
# ④ 未映射提示：单列人工注意，不硬排
# ---------------------------------------------------------------------------
def test_unmapped_block_listed_not_forced() -> None:
    md = (
        "Floquet band structure opens a momentum gap at the Brillouin edge zone.\n\n"
        "zqx wjk vvt unmapmable gibberish token soup nothing alike here.\n"
    )
    pdf = [
        _pdf(
            1,
            0,
            0.0,
            100,
            "Floquet band structure opens a momentum gap at the Brillouin edge zone.",
        ),
    ]
    _mbs, _order, _st, report, _sug = _pipeline(md, pdf)

    assert "## 未映射（人工注意，不硬排）" in report
    tail = report.split("## 未映射", 1)[1]
    assert "zqx wjk" in tail
    assert "[移动] 3 行" not in report, "未映射块不得作为移动建议硬排"


# ---------------------------------------------------------------------------
# ⑤ 切块固定样例：八类块识别 + 行号（报告依据）
# ---------------------------------------------------------------------------
def test_parse_md_blocks_kinds_and_line_numbers() -> None:
    md = (
        "# 标题\n"
        "\n"
        "![](images/a.jpg)\n"
        "\n"
        "$$\n"
        "E = mc^2\n"
        "$$\n"
        "\n"
        "<table>\n"
        "<tr><td>1</td></tr>\n"
        "</table>\n"
        "\n"
        "Fig. 2 | Caption head here.\n"
        "\n"
        "c Experimental sub-caption continuation.\n"
        "\n"
        "a\n"
        "\n"
        "Plain body sentence with words.\n"
    )
    blocks = rs.parse_md_blocks(md)
    kinds = [b.kind for b in blocks]
    assert kinds == [
        "heading",
        "image",
        "formula",
        "table",
        "caption",
        "fragment",
        "label",
        "text",
    ]
    assert [b.index for b in blocks] == list(range(8))
    assert [b.line for b in blocks] == [1, 3, 5, 9, 13, 15, 17, 19]
    assert blocks[2].text == "$$\nE = mc^2\n$$", "公式块整块逐字"
    assert blocks[3].text == "<table>\n<tr><td>1</td></tr>\n</table>", "表格整块逐字"


def test_parse_md_blocks_reference_number_not_fragment() -> None:
    """「10. Author…」式参考文献条目不得判成图注残段（负向先行防误判）。"""
    blocks = rs.parse_md_blocks("10. Author Q, Name W. Title of the reference.\n")
    assert [b.kind for b in blocks] == ["text"]


# ---------------------------------------------------------------------------
# ⑥ 排版带聚类固定样例（_assign_bands：同页 y 交叠并簇 + 带序自上而下）
# ---------------------------------------------------------------------------
def test_assign_bands_merges_overlap_and_orders_top_down() -> None:
    a = rs.PdfBlock(page=1, x0=40, y0=100, x1=290, y1=130, text="band one left")
    b = rs.PdfBlock(page=1, x0=310, y0=105, x1=560, y1=128, text="band one right")
    c = rs.PdfBlock(page=1, x0=40, y0=300, x1=290, y1=330, text="band two")
    rs._assign_bands([c, b, a])  # 乱序输入亦须归位
    assert a.band == b.band == 0, "y 交叠 >30% 并簇同带"
    assert c.band == 1, "带序按区间上沿自上而下"
    assert a.key < c.key, "阅读序键（页,带,栏,y,x）决定建议序"


def test_assign_bands_wide_short_text_band_marked_full_width() -> None:
    """图表带（短文本密布、横向铺开）→ col=0.5（通栏·图表），带内按 (y,x)。"""
    blocks = [
        rs.PdfBlock(page=1, x0=40, y0=100, x1=90, y1=115, text="a"),
        rs.PdfBlock(page=1, x0=300, y0=100, x1=350, y1=115, text="b"),
        rs.PdfBlock(page=1, x0=520, y0=100, x1=570, y1=115, text="c"),
    ]
    rs._assign_bands(blocks)
    assert all(pb.col == 0.5 for pb in blocks), "铺开短文本带按通栏·图表处理"
    assert all(pb.band == 0 for pb in blocks)


# ---------------------------------------------------------------------------
# ⑦ 相似度/映射固定样例（含公式 \tag{N} 尾锚与 PDF 伪字形）
# ---------------------------------------------------------------------------
def test_similarity_fixed_samples() -> None:
    same = "The beta dispersion measurement shows resonance near 4.7 kHz."
    assert rs.similarity(same, same) >= 0.9, "全等必高分"
    assert rs.similarity(same, same.upper()) >= 0.9, "大小写/标点归一消音"
    low = rs.similarity(same, "zqx wjk vvt gibberish token soup")
    assert low < rs.MAP_THRESHOLD, f"无关文本须低于阈值：{low}"
    assert rs.similarity("", same) == 0.0, "空侧 0 分"


def test_formula_tag_tail_anchor_maps_pseudo_glyphs() -> None:
    """公式块走 \\tag{N} ↔ PDF (N)/ðNÞ 尾锚（数学字形相似度不可靠）。"""
    md = "$$\na + b = c\n\\tag{3}\n$$\n"
    mbs = rs.parse_md_blocks(md)
    assert [b.kind for b in mbs] == ["formula"]
    pdf = [
        _pdf(1, 0, 0.0, 50, "a + b = c ð3Þ"),
        _pdf(1, 1, 0.0, 300, "totally different words here"),
    ]
    mapped, unmapped = rs.map_blocks(mbs, pdf)
    assert not unmapped
    assert mapped[0] is pdf[0], "编号尾锚命中伪字形 PDF 块"
    # 括号正体亦命中
    mapped2, _ = rs.map_blocks(mbs, [_pdf(1, 0, 0.0, 50, "x y z (3)")])
    assert mapped2[0].text.endswith("(3)")


def test_image_anchored_to_nearest_caption_key() -> None:
    """图片无文本 → 挂最近图注头的键；与图注同键归组。"""
    md = "Fig. 4 | A nice figure.\n\n![](images/a.jpg)\n"
    pdf = [_pdf(1, 0, 0.0, 50, "Fig. 4 | A nice figure.")]
    mbs = rs.parse_md_blocks(md)
    mapped, _ = rs.map_blocks(mbs, pdf)
    keys = rs.suggest_order(mbs, mapped)
    assert keys[1] == keys[0], "图片与最近图注同键（同位搬运）"


# ---------------------------------------------------------------------------
# ⑧ 建议稿/报告拼装固定样例（块原文逐字）
# ---------------------------------------------------------------------------
def test_build_suggested_md_verbatim_blocks() -> None:
    mbs = rs.parse_md_blocks("# T\n\nFirst para text.\n\nSecond para text.\n")
    order = [2, 1, 0]
    out = rs.build_suggested_md(order, mbs, stitches=set())
    assert out == "Second para text.\n\nFirst para text.\n\n# T\n"
    assert "Second para text.\n\nFirst para text." in out, "空行分块与源同构"


def test_build_report_basis_phrases() -> None:
    """报告依据短语：PDF 页/带/栏/纵坐标；左/右栏与通栏·图表口径。"""
    left = _pdf(3, 2, 0.0, 210, "x")
    right = _pdf(3, 2, 1.0, 210, "y")
    full = _pdf(3, 2, 0.5, 210, "z")
    assert left.basis == "PDF p3 带2 左栏 y=210"
    assert right.basis == "PDF p3 带2 右栏 y=210"
    assert full.basis == "PDF p3 带2 通栏·图表 y=210"


# ---------------------------------------------------------------------------
# ⑨ 可编程驱动：真 PDF 提取 + 建议落盘（pymupdf 现造，无外部依赖）
# ---------------------------------------------------------------------------
def test_pdf_text_blocks_extracts_and_clusters(tmp_path: Path) -> None:
    """PyMuPDF 提取：文本块归一空白、按页/栏赋值、排版带已聚（跨页带序归零）。"""
    pdf = tmp_path / "src.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=595.276, height=841.89)
    page.insert_text((40, 100), "Left column body sentence for band zero.")
    page.insert_text((320, 250), "Right column body sentence lower band.")
    page2 = doc.new_page(width=595.276, height=841.89)
    page2.insert_text((40, 100), "Second page opening sentence here.")
    doc.save(str(pdf))
    doc.close()

    blocks = rs.pdf_text_blocks(pdf)
    assert len(blocks) == 3
    assert all(pb.band >= 0 for pb in blocks)
    assert [pb.page for pb in blocks] == [1, 1, 2]
    assert blocks[0].col == 0.0 and blocks[1].col == 1.0, f"左右栏判据：{blocks}"
    assert blocks[2].col == 0.0 and blocks[2].band == 0, "带序按页归零"
    texts = [pb.text for pb in blocks]
    assert "Left column body sentence for band zero." in texts
    assert all("  " not in t for t in texts), "块内空白已归一"


def test_suggest_reorder_writes_artifacts_and_keeps_source(tmp_path: Path) -> None:
    """驱动全链：reorder-suggested.md + reorder-report.md 落盘；源 md 不改（只出建议）。"""
    md = tmp_path / "src.md"
    md_text = (
        "Fig. 1 | Concept of a crystal.\n\n"
        "Plain body sentence with several words in it.\n\n"
        "![](images/x.jpg)\n"
    )
    md.write_text(md_text, encoding="utf-8")
    pdf = tmp_path / "src.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=595.276, height=841.89)
    page.insert_text((40, 100), "Fig. 1 | Concept of a crystal.")
    page.insert_text((40, 300), "Plain body sentence with several words in it.")
    doc.save(str(pdf))
    doc.close()

    result = rs.suggest_reorder(md, pdf, tmp_path / "out")
    assert result.suggested_path.is_file() and result.report_path.is_file()
    assert result.n_blocks == 3
    assert result.n_unmapped == 0, "固定样例两文本块均可映射"
    suggested = result.suggested_path.read_text(encoding="utf-8")
    report = result.report_path.read_text(encoding="utf-8")
    assert "Concept of a crystal." in suggested
    assert "images/x.jpg" in suggested
    assert "reorder-report" in report.splitlines()[0]
    assert md.read_text(encoding="utf-8") == md_text, "源 md 逐字不动（只出建议）"


def test_suggest_reorder_missing_inputs_raise(tmp_path: Path) -> None:
    """缺文件语义：FileNotFoundError（不静默造空建议）。"""
    with pytest.raises(FileNotFoundError):
        rs.suggest_reorder(tmp_path / "no.md", tmp_path / "no.pdf", tmp_path / "out")
