"""``_tex_to_markdown`` 的噪声剥离（注释行 / REVTeX 前置宏 / 图片浮动体）。

这一层的产物会写进 ``cache/extracted/**/*.md``，而 :mod:`rag` 正是扫这个目录建语料的
（``rag index`` 只认 ``cache/extracted/**/*.md``）。所以「LaTeX 排版指令混进正文」不只是
难看：它会把 RAG 分块的开头让给 ``\\affiliation{...}`` 这类零信息行，把真正的标题与摘要
挤出第一个分块。

实测样本是 arXiv 1803.04110（REVTeX 4.2）：修之前产物的前 500 字符全是 ``%`` 注释与
``\\title`` / ``\\affiliation`` 原始宏，正文要到第 28 行才开始；修之后第 0 行就是标题。
"""

from __future__ import annotations

from pysci.skills.literature_research.tools import pdf_extract

_strip = pdf_extract._strip_tex_comments
_to_md = pdf_extract._tex_to_markdown


# ---------------------------------------------------------------------------
#  注释行
# ---------------------------------------------------------------------------
def test_comment_lines_are_gone_even_without_a_document_environment() -> None:
    """没有 ``\\begin{document}`` 时注释照样被剥掉。

    这是本项的**根因**：多文件 arXiv 工程常把 ``\\begin{document}`` 留在被 ``\\input``
    的子文件里，于是「去掉 preamble」整步失效，被作者注掉的 ``%\\title{...}`` 就以正文
    身份留在产物里。剥注释必须发生在切 preamble **之前**，才不依赖那一步是否生效。
    """
    tex = "%\\title{old title}\n%\\author{Someone}\nreal body line\n"

    out = _to_md(tex)

    assert "%" not in out
    assert "old title" not in out
    assert "real body line" in out


def test_an_escaped_percent_is_a_literal_character_not_a_comment() -> None:
    """``50\\% efficiency`` 里的 ``%`` 是字面字符，删掉它就丢了正文。

    判据是它前面连续反斜杠的奇偶：奇数个 ⇒ 被转义；偶数个 ⇒ 注释起始。
    """
    assert pdf_extract._strip_line_comment(r"50\% efficiency") == r"50\% efficiency"
    assert pdf_extract._strip_line_comment(r"a \% b % c") == r"a \% b"
    # 两个反斜杠是 LaTeX 的「换行」，其后的 % 仍是注释起始
    assert pdf_extract._strip_line_comment("x \\\\ % c") == "x \\\\"
    assert pdf_extract._strip_line_comment("no percent here") == "no percent here"


def test_verbatim_content_keeps_its_percent_signs() -> None:
    """代码清单里的 ``%`` 是内容，不是注释——整段必须原样放行。"""
    tex = "\\begin{verbatim}\n100% literal % still literal\n\\end{verbatim}\n"

    out = _strip(tex)

    assert "100% literal % still literal" in out


def test_a_balanced_comment_environment_is_dropped_whole() -> None:
    """``comment`` 包的内容 LaTeX 自己就整段丢弃，故与注释同类。"""
    tex = "keep\n\\begin{comment}\nwhole section gone\n\\end{comment}\ntail\n"

    out = _strip(tex)

    assert "whole section gone" not in out
    assert "keep" in out and "tail" in out


def test_an_unbalanced_comment_environment_does_not_eat_the_rest() -> None:
    """不配平时宁可漏删，也不能把此后整个文档吃掉。

    作者手改留下的半个 ``\\begin{comment}`` 是真实存在的形态；若照删，产物会变成空
    文档——那比漏几行注释严重得多，而且是**静默**的（抽取照样「成功」）。
    """
    tex = "a\n\\begin{comment}\nb\n"

    assert _strip(tex) == tex


# ---------------------------------------------------------------------------
#  REVTeX 前置宏
# ---------------------------------------------------------------------------
_REVTEX_HEAD = """\\begin{document}
\\title{Topological Edge State and Exceptional Point}
\\author{Weiwei Zhu}
\\affiliation{Tongji University}
\\author{Yun Jing}
\\email{yjing2@ncsu.edu}
\\affiliation{North Carolina State University}
\\begin{abstract}
We report an observation.
\\end{abstract}
\\maketitle
\\section{Introduction}
Body text here.
\\end{document}
"""


def test_title_becomes_a_real_markdown_heading() -> None:
    """标题必须是产物的第一行。

    不是为了好看：RAG 按块检索，标题落在第一个分块里才可能被「这篇讲什么」这类
    查询命中；被二十行 ``\\affiliation`` 顶下去等于标题在语料里不存在。
    """
    out = _to_md(_REVTEX_HEAD)

    assert out.split("\n")[0] == "# Topological Edge State and Exceptional Point"


def test_authors_are_merged_into_one_line_and_affiliations_dropped() -> None:
    out = _to_md(_REVTEX_HEAD)

    assert "**Authors:** Weiwei Zhu, Yun Jing" in out
    assert "affiliation" not in out.lower()
    assert "ncsu.edu" not in out, "邮箱对检索零贡献，且属个人信息"
    assert "maketitle" not in out


def test_the_abstract_and_body_survive_the_front_matter_pass() -> None:
    """删前置宏不得误伤正文——这是本项最容易做错的方向。"""
    out = _to_md(_REVTEX_HEAD)

    assert "We report an observation." in out
    assert "Body text here." in out
    assert "## Introduction" in out


def test_a_title_with_nested_braces_is_captured_whole() -> None:
    """``.+?`` 会在内层花括号处提前收尾，故显式支持一层嵌套。"""
    out = _to_md("\\begin{document}\n\\title{The {XYZ} effect}\n\\end{document}\n")

    assert "# The {XYZ} effect" in out


def test_front_matter_handling_tolerates_a_document_without_any() -> None:
    """没有 ``\\title`` / ``\\author`` 的源码不该被塞进空标题或空作者行。"""
    out = _to_md("\\begin{document}\njust body\n\\end{document}\n")

    assert out == "just body"


# ---------------------------------------------------------------------------
#  图片浮动体
# ---------------------------------------------------------------------------
def test_includegraphics_and_the_figure_wrapper_go_but_the_caption_stays() -> None:
    """图本身在 markdown 里无从渲染，但 caption 是有信息量的正文。"""
    tex = (
        "\\begin{document}\n"
        "\\begin{figure}\n"
        "\\centering\n"
        "\\includegraphics[width=0.8\\linewidth]{Fig-1}\n"
        "\\caption{The schematic of a unit cell.}\n"
        "\\end{figure}\n"
        "\\end{document}\n"
    )

    out = _to_md(tex)

    assert "includegraphics" not in out
    assert "begin{figure}" not in out and "end{figure}" not in out
    assert "centering" not in out
    assert "The schematic of a unit cell." in out


def test_a_table_environment_is_left_alone() -> None:
    """表格沿用既定的「保留原样」决策：删包裹只会把 ``&`` 行拆散，比留着更糟。"""
    tex = (
        "\\begin{document}\n"
        "\\begin{table}\n"
        "\\begin{tabular}{lc}\n"
        "A & 1 \\\\\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
        "\\end{document}\n"
    )

    out = _to_md(tex)

    assert "\\begin{table}" in out
    assert "A & 1" in out


# ---------------------------------------------------------------------------
#  端到端判据
# ---------------------------------------------------------------------------
def test_no_line_of_the_output_starts_with_a_percent_sign() -> None:
    """这条断言直接对应实测缺陷，也最不容易随实现细节漂移。

    样本刻意同时包含：``\\begin{document}`` 之前的注释（preamble 切割能兜住）、之后的
    注释（兜不住，只能靠剥注释）、以及一个必须保留的字面 ``\\%``。
    """
    tex = (
        "%!TEX program = pdflatex\n"
        "\\documentclass{revtex4-2}\n"
        "%\\title{a commented out title}\n"
        "\\begin{document}\n"
        "\\title{Real Title}\n"
        "% a commented line inside the body\n"
        "Text with 50\\% yield % trailing note\n"
        "\\end{document}\n"
    )

    out = _to_md(tex)

    assert not [ln for ln in out.split("\n") if ln.lstrip().startswith("%")]
    assert "50\\% yield" in out
    assert "trailing note" not in out
    assert "a commented out title" not in out
    assert out.startswith("# Real Title")
