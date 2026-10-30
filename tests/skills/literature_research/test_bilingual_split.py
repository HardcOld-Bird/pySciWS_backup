"""bilingual 句切分与结构解析的回归（任务书 20261023-001431 §三「切分」面）。

锁三件事：学术缩写不误切（Fig./Eq./et al./e.g./vs.）、``?``/``!`` 正确切、省略号
不作句末；外加 minerU 风格 md 的结构解析（标题/段落/图注/表注/公式/表格/参考文献）
与切片自检。纯函数面，不碰磁盘。
"""

from __future__ import annotations

import re

import pytest

from pysci.skills.literature_research.tools import bilingual as bl

#: 合成 fixture：迷你 minerU 风格 md（标题/软换行/缩写/图+图注/公式/表格/表注/参考文献）。
MINI_MD = """# Introduction

This is the first sentence. It has a soft
wrap inside it! Is it true?

See Fig. 1 and Eq. (2). The cost is e.g. five dollars vs. ten. Smith et al. showed this.

![x](images/fig1.jpg)

Fig. 1. The experimental setup (a). It works well.

$$
E = mc^2
$$

<table>
<tr><td>1</td></tr>
</table>

Table 1. The measured data.

## References

[1] A. Author et al., Title, J. 2020.
"""


def test_abbreviations_are_not_split() -> None:
    """Fig./Eq./e.g./vs./et al. 的句中点不得成为句末。"""
    para = (
        "See Fig. 1 and Eq. (2). The cost is e.g. five dollars vs. ten. "
        "Smith et al. showed this."
    )
    assert bl.split_sentences(para) == [
        "See Fig. 1 and Eq. (2).",
        "The cost is e.g. five dollars vs. ten.",
        "Smith et al. showed this.",
    ]
    assert bl.split_sentences("Compare Fig. 1. Then stop.") == [
        "Compare Fig. 1.",
        "Then stop.",
    ]


def test_question_and_exclaim_split() -> None:
    """``?``/``!`` 是句末符，正确切分。"""
    assert bl.split_sentences("Is it true? Yes! It is.") == [
        "Is it true?",
        "Yes!",
        "It is.",
    ]


def test_ellipsis_is_not_sentence_end() -> None:
    """省略号（``...``/``…``）不作句末——学术正文中几乎总是句中省略。"""
    assert bl.split_sentences(
        "We measured it... then stopped. Next came the rest."
    ) == [
        "We measured it... then stopped.",
        "Next came the rest.",
    ]
    assert bl.split_sentences("The value… remains unknown.") == [
        "The value… remains unknown.",
    ]
    assert bl.split_sentences("Trailing dots are not the end...") == [
        "Trailing dots are not the end...",
    ]


def test_soft_wrap_merged_before_splitting() -> None:
    """段内软换行先合并再切句：跨行句是完整一句。"""
    assert bl.split_sentences("It has a soft\nwrap inside it! Is it true?") == [
        "It has a soft wrap inside it!",
        "Is it true?",
    ]


# ---------------------------------------------------------------------------
# 面板号回并逐字守恒（backlog 20261029-121943-lit）
# ---------------------------------------------------------------------------
PANEL_MD = """# Results

As shown in Fig. 3(a). What we see here is clear. Figs. 3(b)–(d). More details follow.
"""


def test_panel_number_without_space_stays_verbatim() -> None:
    """数字紧接单字母面板号（``Fig. 3(a).``）不得被回并插空格——分隔按原文间隙取。"""
    assert bl.split_sentences("as shown in Fig. 3(a). What we see here") == [
        "as shown in Fig. 3(a).",
        "What we see here",
    ]
    assert bl.split_sentences("Figs. 3(b)–(d). More details follow.") == [
        "Figs. 3(b)–(d).",
        "More details follow.",
    ]


def test_panel_paragraph_passes_slice_verification() -> None:
    """含无空格面板号的段落整链过切片自检，且各单元 en 空白归一后与源文逐字一致。"""
    doc = bl.parse_markdown(PANEL_MD)
    bl.verify_slices(PANEL_MD, doc)  # 不抛即通过（prepare 的硬闸）
    norm = lambda s: re.sub(r"\s+", " ", s).strip()  # noqa: E731
    whole = norm(PANEL_MD)
    for u in doc.units:
        assert norm(u.en) in whole, f"{u.id} 与源文不再逐字一致：{u.en!r}"


def test_parse_ids_types_and_order() -> None:
    """ID = 类型字母 + 4 位序号，全文档唯一、按出现顺序；类型判定正确。"""
    doc = bl.parse_markdown(MINI_MD)
    ids = [u.id for u in doc.units]
    assert ids == [
        "H0001",
        "S0001",
        "S0002",
        "S0003",
        "S0004",
        "S0005",
        "S0006",
        "C0001",
        "C0002",
        "C0003",
        "T0001",
        "T0002",
        "H0002",
    ]
    assert [u.type for u in doc.units] == [
        "H",
        "S",
        "S",
        "S",
        "S",
        "S",
        "S",
        "C",
        "C",
        "C",
        "T",
        "T",
        "H",
    ]
    orders = [u.order for u in doc.units]
    assert orders == sorted(orders)
    assert doc.units[0].en == "Introduction"
    assert doc.units[1].en == "This is the first sentence."


def test_parse_pass_blocks_verbatim() -> None:
    """透传块（公式/表格/图片行/参考文献）原样保留，且计入 order 序列。"""
    doc = bl.parse_markdown(MINI_MD)
    kinds = [b.kind for b in doc.blocks]
    assert kinds == ["image", "formula", "table", "reference"]
    by_kind = {b.kind: b for b in doc.blocks}
    assert by_kind["image"].content == "![x](images/fig1.jpg)"
    assert by_kind["formula"].content == "$$\nE = mc^2\n$$"
    assert by_kind["table"].content == "<table>\n<tr><td>1</td></tr>\n</table>"
    assert by_kind["reference"].content == "[1] A. Author et al., Title, J. 2020."
    assert doc.images == ["images/fig1.jpg"]
    all_orders = sorted([*doc.units, *doc.blocks], key=lambda x: x.order)
    assert [x.order for x in all_orders] == list(range(len(all_orders)))


def test_image_and_caption_share_para() -> None:
    """图注与其图片块同 para（渲染契约：图片原位，紧跟其图注句对，不隔 ---）。"""
    doc = bl.parse_markdown(MINI_MD)
    image = next(b for b in doc.blocks if b.kind == "image")
    captions = [u for u in doc.units if u.type == "C"]
    assert {u.para for u in captions} == {image.para}


def test_references_section_is_pass_through() -> None:
    """参考文献节标题后的段落按条目透传，不再切句成待译单元。"""
    doc = bl.parse_markdown(MINI_MD)
    ref_units = [u for u in doc.units if "Author" in u.en]
    assert ref_units == []
    refs = [b for b in doc.blocks if b.kind == "reference"]
    assert len(refs) == 1 and "et al." in refs[0].content


def test_table_caption_typed_t() -> None:
    """表注段落按前缀惯例判 T（Fig. 判 C，无前缀一律 S）。"""
    doc = bl.parse_markdown(MINI_MD)
    assert [u.en for u in doc.units if u.type == "T"] == [
        "Table 1.",
        "The measured data.",
    ]
    plain = bl.parse_markdown("Random text. Second bit.")
    assert [u.type for u in plain.units] == ["S", "S"]


def test_verify_slices_passes_on_clean_parse() -> None:
    """切片自检：解析结果按 order 能在源文档中按序找回。"""
    doc = bl.parse_markdown(MINI_MD)
    bl.verify_slices(MINI_MD, doc)  # 不抛即通过


def test_verify_slices_detects_tampered_slice() -> None:
    """切片被改（错位/篡改）时自检必须报 ValueError，prepare 据此拒绝。"""
    doc = bl.parse_markdown(MINI_MD)
    doc.units[1].en = "This sentence was tampered."
    with pytest.raises(ValueError, match="切片自检失败"):
        bl.verify_slices(MINI_MD, doc)
