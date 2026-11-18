"""句切分与结构解析回归（旧 test_bilingual_split.py 期望值平移；语义准绳不改）。

锁四件事：学术缩写/省略号不误切、``?``/``!`` 正确切、游标逐字（面板号不回并漂移）、
markdown-it-py 解析的单元/透传块/para 契约与切片自检。固定样例机械断言；真 wtpsplit
接缝就绪时同口径复验（缺件跳过）。
"""

from __future__ import annotations

import re

import pytest

from pysci.skills.literature_research.bilingual import units as bl

from .conftest import REAL_DEFAULT_SEGMENTER, WTPSPLIT_READY

#: 合成 fixture：迷你 MinerU 风格 md（标题/软换行/缩写/图+图注/公式/表格/表注/参考文献）。
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


def split(paragraph: str) -> list[str]:
    """参照切分器口径的 split_sentences（autouse 垫片已把 default_segmenter 换掉）。"""
    return bl.split_sentences(paragraph)


# ---------------------------------------------------------------------------
# 句切分契约
# ---------------------------------------------------------------------------
def test_abbreviations_are_not_split() -> None:
    """Fig./Eq./e.g./vs./et al. 的句中点不得成为句末。"""
    para = (
        "See Fig. 1 and Eq. (2). The cost is e.g. five dollars vs. ten. "
        "Smith et al. showed this."
    )
    assert split(para) == [
        "See Fig. 1 and Eq. (2).",
        "The cost is e.g. five dollars vs. ten.",
        "Smith et al. showed this.",
    ]
    assert split("Compare Fig. 1. Then stop.") == ["Compare Fig. 1.", "Then stop."]


def test_question_and_exclaim_split() -> None:
    """``?``/``!`` 是句末符，正确切分。"""
    assert split("Is it true? Yes! It is.") == ["Is it true?", "Yes!", "It is."]


def test_ellipsis_is_not_sentence_end() -> None:
    """省略号（``...``/``…``）不作句末——学术正文中几乎总是句中省略。"""
    assert split("We measured it... then stopped. Next came the rest.") == [
        "We measured it... then stopped.",
        "Next came the rest.",
    ]
    assert split("The value… remains unknown.") == ["The value… remains unknown."]
    assert split("Trailing dots are not the end...") == [
        "Trailing dots are not the end..."
    ]


def test_soft_wrap_merged_before_splitting() -> None:
    """段内软换行先合并再切句：跨行句是完整一句。"""
    assert split("It has a soft\nwrap inside it! Is it true?") == [
        "It has a soft wrap inside it!",
        "Is it true?",
    ]


def test_panel_number_without_space_stays_verbatim() -> None:
    """数字紧接单字母面板号（``Fig. 3(a).``）不得被回并插空格——分隔按原文间隙取。"""
    assert split("as shown in Fig. 3(a). What we see here") == [
        "as shown in Fig. 3(a).",
        "What we see here",
    ]
    assert split("Figs. 3(b)–(d). More details follow.") == [
        "Figs. 3(b)–(d).",
        "More details follow.",
    ]


def test_reassemble_merges_fragments_without_sentence_end() -> None:
    """游标拼回纯逻辑：碎片不以 . ? ! 结尾即回并，分隔按原文间隙（不硬插空格）。"""
    merged = "Fig. 3(a). What we see"
    out = bl.reassemble(merged, ["Fig. 3(a)", ". What we see"])
    assert out == ["Fig. 3(a). What we see"]
    # 定位失败时退化为空格分隔（不炸、不丢原文：间隙照抄，仅插入碎片）。
    assert bl.reassemble("a b c", ["a", "zzz", "c"]) == ["a zzz b c"]


def test_reassemble_sentence_end_rule_only_dot_q_bang() -> None:
    """句末合并只认强标点（句末符契约与切分器无关）。"""
    out = bl.reassemble("One. Two. Three.", ["One.", " Two.", " Three."])
    assert out == ["One.", "Two.", "Three."]
    out = bl.reassemble("One two three", ["One", " two", " three"])
    assert out == ["One two three"]


# ---------------------------------------------------------------------------
# 句末硬契约补齐（切分器跨句界合块时的规整；CJK 强标点与括号内不切）
# ---------------------------------------------------------------------------
def test_force_split_repairs_merged_sentences() -> None:
    """切分器把 ``Yes! It is.`` 合块/``Table 1. …`` 漏切时，硬契约层强制按 . ? ! 补切。"""

    def chunky(_protected: str) -> list[str]:
        return ["Is it true? Yes! It is. Table 1. The measured data."]

    assert bl.split_sentences(
        "Is it true? Yes! It is. Table 1. The measured data.", segmenter=chunky
    ) == ["Is it true?", "Yes!", "It is.", "Table 1.", "The measured data."]


def test_force_split_cjk_strong_punctuation() -> None:
    """中文强标点（。！？）无空格直接断（K2 实测 sat-3l 中文边界误切/漏切防护）。"""

    def chunky(_protected: str) -> list[str]:
        return ["机器学习是分支。它使计算机学习！真的吗？真的。"]

    assert bl.split_sentences(
        "机器学习是分支。它使计算机学习！真的吗？真的。", segmenter=chunky
    ) == ["机器学习是分支。", "它使计算机学习！", "真的吗？", "真的。"]


def test_force_split_keeps_brackets_intact() -> None:
    """括号内不切（``（deep learning）`` 类内含点号/强标点不误断）。"""

    def chunky(_protected: str) -> list[str]:
        return ["We use the method (also known as e.g. this. Not a stop) here. Done."]

    assert bl.split_sentences(
        "We use the method (also known as e.g. this. Not a stop) here. Done.",
        segmenter=chunky,
    ) == [
        "We use the method (also known as e.g. this. Not a stop) here.",
        "Done.",
    ]


def test_ellipsis_cjk_not_sentence_end_in_force_split() -> None:
    """省略号（…）在硬契约层也不作句末（排除集双保险）。"""

    def chunky(_protected: str) -> list[str]:
        return ["它停了下来……然后继续。下一句。"]

    assert bl.split_sentences("它停了下来……然后继续。下一句。", segmenter=chunky) == [
        "它停了下来……然后继续。",
        "下一句。",
    ]


# ---------------------------------------------------------------------------
# 结构解析契约（markdown-it-py token.map 逐字切片）
# ---------------------------------------------------------------------------
def test_parse_ids_types_and_order() -> None:
    """ID = 类型字母 + 4 位序号，全文档唯一、按出现顺序；类型判定正确。"""
    doc = bl.parse_markdown(MINI_MD)
    ids = [u.id for u in doc.units]
    assert ids == [
        "H0001",
        "S0001", "S0002", "S0003", "S0004", "S0005", "S0006",
        "C0001", "C0002", "C0003",
        "T0001", "T0002",
        "H0002",
    ]
    assert [u.type for u in doc.units] == [
        "H", "S", "S", "S", "S", "S", "S", "C", "C", "C", "T", "T", "H",
    ]
    orders = [u.order for u in doc.units]
    assert orders == sorted(orders)
    assert doc.units[0].en == "Introduction"
    assert doc.units[1].en == "This is the first sentence."
    assert doc.units[0].level == 1


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


def test_formula_block_with_blank_line_stays_one_block() -> None:
    """公式块含空行仍整块透传（CommonMark 断块由行级续块 shim 补齐，不产碎句）。"""
    md = "# T\n\n$$\nE = mc^2\n\nF = ma\n$$\n"
    doc = bl.parse_markdown(md)
    formulas = [b for b in doc.blocks if b.kind == "formula"]
    assert len(formulas) == 1
    assert formulas[0].content == "$$\nE = mc^2\n\nF = ma\n$$"
    assert [u.en for u in doc.units] == ["T"]


def test_image_and_caption_share_para() -> None:
    """图注与其图片块同 para（渲染契约：图片原位，紧跟其图注句对，不隔 ---）。"""
    doc = bl.parse_markdown(MINI_MD)
    image = next(b for b in doc.blocks if b.kind == "image")
    captions = [u for u in doc.units if u.type == "C"]
    assert {u.para for u in captions} == {image.para}


def test_image_line_with_caption_same_paragraph() -> None:
    """MinerU 真实产物形态：图行与图注行同段落（图行后硬换行紧跟图注）。

    2026-11-17 全链冒烟实测回归：图行须剥离为透传块并登记引用名，图注判 C、与图片
    同 para；不得把 ``![](…)`` 并进正文句（否则图不回补、图注原子失效）。
    """
    md = (
        "# Model\n\n"
        "Body text here.\n\n"
        "![](images/fig1.jpg)  \n"
        "Figure 1: The circuit model. The basic elements are shown.\n"
    )
    doc = bl.parse_markdown(md)
    assert doc.images == ["images/fig1.jpg"]
    images_ = [b for b in doc.blocks if b.kind == "image"]
    assert len(images_) == 1 and images_[0].content == "![](images/fig1.jpg)"
    captions = [u for u in doc.units if u.type == "C"]
    assert [u.en for u in captions] == [
        "Figure 1: The circuit model.",
        "The basic elements are shown.",
    ]
    assert {u.para for u in captions} == {images_[0].para}
    assert not any("![](" in u.en for u in doc.units), "图行不得混入待译单元"
    bl.verify_slices(md, doc)  # 剥离后切片自检仍须无损


def test_spaced_abbreviations_not_split() -> None:
    """空格变体缩写（MinerU/OCR 常见「i. e.」「e. g.」）的句中点不得成为句末。"""
    para = (
        "The elements are: the source, i. e. the photocurrent, "
        "and e. g. the diode. Done."
    )
    assert split(para) == [
        "The elements are: the source, i. e. the photocurrent, and e. g. the diode.",
        "Done.",
    ]


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


def test_panel_paragraph_passes_slice_verification() -> None:
    """含无空格面板号的段落整链过切片自检，且各单元 en 空白归一后与源文逐字一致。"""
    md = (
        "# Results\n\n"
        "As shown in Fig. 3(a). What we see here is clear. "
        "Figs. 3(b)–(d). More details follow.\n"
    )
    doc = bl.parse_markdown(md)
    bl.verify_slices(md, doc)  # 不抛即通过（prepare 的硬闸）
    norm = lambda s: re.sub(r"\s+", " ", s).strip()  # noqa: E731
    whole = norm(md)
    for u in doc.units:
        assert norm(u.en) in whole, f"{u.id} 与源文不再逐字一致：{u.en!r}"


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


# ---------------------------------------------------------------------------
# 摘要识别（role=abstract，2026-10-23 裁决）
# ---------------------------------------------------------------------------
ABS_MD = """Article

# A Study of Example Systems

https://doi.org/10.1000/xyz1

Received: 1 January 2026

Alice Smith $^{1}$ , Bob Jones $^{2,3}$

This is the abstract first sentence. It continues in the same abstract paragraph.

Body paragraph one lives here. It has two sentences.

## Results

Result sentence one here.
"""


def test_mark_abstract_meta_band_heuristic() -> None:
    """元数据/作者行后首个段落全组置 role=abstract；无元数据带不标注。"""
    doc = bl.parse_markdown(ABS_MD)
    n = bl.mark_abstract(doc)
    assert n == 2
    assert {u.en for u in doc.units if u.role == "abstract"} == {
        "This is the abstract first sentence.",
        "It continues in the same abstract paragraph.",
    }
    plain = bl.parse_markdown("Body only. Two sentences.")
    assert bl.mark_abstract(plain) == 0


# ---------------------------------------------------------------------------
# 真 wtpsplit 接缝（缺件跳过；装好/vendor 模型后自动激活）
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not WTPSPLIT_READY, reason="wtpsplit/SaT 模型待依赖代理安装与 vendor")
def test_wtpsplit_default_seam_matches_contract() -> None:
    """真 wtpsplit 接缝与参照切分器同口径产出（缩写/面板号/句末契约不依赖切分器）。"""
    para = "See Fig. 1 and Eq. (2). The cost is e.g. five vs. ten. Smith et al. showed this."
    assert bl.split_sentences(para, segmenter=REAL_DEFAULT_SEGMENTER) == split(para)
