"""tex_clean 固定样例断言：注释/preamble/thebibliography 剔除与轻量 MD 映射。"""

from __future__ import annotations

from pysci.skills.literature_research import tex_clean

SRC = (
    "\\documentclass{article}\n"
    "% preamble comment\n"
    "\\title{A Paper}\n"
    "\\begin{document}\n"
    "% body comment\n"
    "We keep 100\\% of $x^2$ and \\cite{doe2020,roe2021}.\n"
    "\\section{Intro}\n"
    "Hello \\textbf{bold} \\emph{it} \\eqref{eq:1}.\n"
    "\\begin{thebibliography}{9}\n"
    "\\bibitem{doe2020} Doe.\n"
    "\\end{thebibliography}\n"
    "\\end{document}\n"
)


def test_strip_comments_keeps_escaped_percent():
    out = tex_clean.strip_comments("100\\% done % trailing\nnext")
    assert out == "100\\% done \nnext"


def test_strip_preamble_and_end_document():
    out = tex_clean.strip_preamble(SRC)
    assert "documentclass" not in out
    assert "title{A Paper}" not in out
    assert "\\end{document}" not in out
    assert "\\section{Intro}" in out


def test_strip_preamble_without_anchor_is_identity():
    assert tex_clean.strip_preamble("body only") == "body only"


def test_strip_thebibliography():
    out = tex_clean.strip_thebibliography(SRC)
    assert "thebibliography" not in out
    assert "bibitem" not in out


def test_tex_body_to_markdown_mapping():
    out = tex_clean.tex_body_to_markdown("x \\section{Intro} \\subsection{Sub} \\textbf{b} \\textit{i} \\ref{eq:1}")
    assert "## Intro" in out
    assert "### Sub" in out
    assert "**b**" in out
    assert "*i*" in out
    assert "[ref:eq:1]" in out


def test_clean_arxiv_tex_full_chain():
    out = tex_clean.clean_arxiv_tex(SRC)
    assert "documentclass" not in out  # preamble 剔除
    assert "% body comment" not in out  # 注释剔除
    assert "100\\%" in out  # 转义百分号保留
    assert "$x^2$" in out  # 数学原样
    assert "\\cite{doe2020,roe2021}" in out  # cite 原样（citecheck 依赖）
    assert "## Intro" in out  # 轻量 MD 映射
    assert "bibitem" not in out  # 书目剔除
