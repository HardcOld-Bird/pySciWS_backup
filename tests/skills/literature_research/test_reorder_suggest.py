"""reorder-suggest 回归（bilingual-orderfix p2-1）：残段缝合 / 标签处置 / 序推断 / 未映射。

核心行为（B 路径，只出建议不改源）：PDF 文本块坐标 → 排版带键序 → 同键归组把图注残段
缝回图注；孤立面板标签归位/剔出；映射不上只单列「未映射」不硬排。全部用合成 md + 合成
PDF 块（离线确定），真机冒烟另走 temporal_ep 样本。
"""

from __future__ import annotations

from pysci.skills.literature_research.tools import reorder_suggest as rs


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
