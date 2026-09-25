"""latex_build 测试：重点覆盖 .log 解析（纯函数，无需 TeX）。"""

from __future__ import annotations

import pytest

from pysci.skills.document_writing.tools import latex_build
from pysci.skills.document_writing.tools.config import settings

SAMPLE_LOG = r"""
This is XeTeX, Version 3.14159265
(./main.tex
LaTeX2e <2025-06-01>
./main.tex:42: Undefined control sequence.
l.42 \badcommand
./sections/intro.tex:17: Missing $ inserted.
<inserted text>
! LaTeX Error: Environment foobar undefined.
l.57 \begin{foobar}
LaTeX Warning: Reference `fig:spectrum' on page 1 undefined on input line 30.
Package hyperref Warning: Token not allowed in a PDF string.
Overfull \hbox (15.0pt too wide) in paragraph at lines 20--25
Underfull \vbox (badness 10000) detected at line 88
)
Output written on build/main.pdf (3 pages).
"""


def test_parse_log_errors():
    errors, warnings = latex_build.parse_log(SAMPLE_LOG)
    msgs = " ".join(e.message for e in errors)
    # file-line-error 风格
    assert any(e.line == 42 and "Undefined control sequence" in e.message for e in errors)
    assert any(e.file.endswith("intro.tex") and e.line == 17 for e in errors)
    # 经典 ! 风格 + 向下找 l.N 行号
    assert any("Environment foobar undefined" in e.message and e.line == 57 for e in errors)
    assert len(errors) >= 3
    assert msgs  # 非空


def test_parse_log_warnings_and_boxes():
    errors, warnings = latex_build.parse_log(SAMPLE_LOG)
    kinds = [w.kind for w in warnings]
    assert "box" in kinds  # Overfull/Underfull 被识别
    # undefined reference 警告带行号
    assert any(w.line == 30 and "fig:spectrum" in w.message for w in warnings)
    # overfull 盒记录了行范围起点
    assert any(w.kind == "box" and w.line == 20 for w in warnings)


def test_parse_log_empty():
    errors, warnings = latex_build.parse_log("")
    assert errors == [] and warnings == []


def test_issue_fmt():
    it = latex_build.TexIssue("error", "Boom", file="main.tex", line=9)
    assert it.fmt() == "[error] main.tex:9: Boom"


def test_engine_flag_mapping():
    assert latex_build._ENGINE_FLAG["xelatex"] == "-xelatex"
    assert latex_build._ENGINE_FLAG["pdflatex"] == "-pdf"


@pytest.mark.skipif(settings.tex_ready, reason="仅在未安装 TeX 时有意义")
def test_build_raises_tex_not_installed(tmp_path):
    tex = tmp_path / "main.tex"
    tex.write_text(r"\documentclass{article}\begin{document}hi\end{document}")
    with pytest.raises(latex_build.TeXNotInstalled):
        latex_build.build(tex)
