"""latex_workflow 机械断言：日志解析 / chktex 输出解析（固定样例）+ 硬件标记编译。

``-m hardware`` 用例真正调用 latexmk/chktex（TeX Live 2026 在 deps/tex junction），
tests.cmd all 默认排除；解析器断言为纯函数，恒跑。
"""

from __future__ import annotations

import pytest

from pysci.skills.document_writing import latex_workflow
from pysci.skills.document_writing.toolpaths import TEXLIVE_BIN

_needs_tex = pytest.mark.hardware
_needs_texlive = pytest.mark.skipif(
    not (TEXLIVE_BIN / "latexmk.exe").is_file(), reason="TeX Live 未落盘"
)

# 固定样例 1：经典风格日志（! 错误 + l.<n> 指针 + 括号文件栈）
CLASSIC_LOG = (
    "This is pdfTeX, Version 3.14159\n"
    "(./main.tex (./chapters/intro.tex\n"
    "! Undefined control sequence.\n"
    "l.12 \\foo\n"
    "?\n"
    "! LaTeX Error: File 'weird.sty' not found.\n"
    "\n"
    "See the LaTeX manual for explanation.\n"
    "l.20 \\usepackage\n"
    ")\n"
    ")\n"
)

# 固定样例 2：-file-line-error 风格（本模块 build 统一开启）
FILELINE_LOG = (
    "(./t.tex\n"
    "./t.tex:3: Undefined control sequence.\n"
    "l.3 Hello \\undefinedcmd\n"
    ")\n"
)

# 固定样例 3：警告（行号在后续行的 on input line）
WARN_LOG = (
    "LaTeX Warning: Reference `fig:1' on page 1 undefined on input line 12.\n"
    "\n"
    "Package foo Warning: something odd on input line 34.\n"
)


def test_parse_classic_errors_with_file_stack():
    issues = latex_workflow.parse_log(CLASSIC_LOG)
    errors = [i for i in issues if i.severity == "error"]
    assert len(errors) == 2
    assert errors[0].message == "Undefined control sequence."
    assert errors[0].file == "./chapters/intro.tex"
    assert errors[0].line == 12
    assert errors[1].message == "LaTeX Error: File 'weird.sty' not found."
    assert errors[1].file == "./chapters/intro.tex"
    assert errors[1].line == 20


def test_parse_fileline_error_style():
    issues = latex_workflow.parse_log(FILELINE_LOG)
    errors = [i for i in issues if i.severity == "error"]
    assert len(errors) == 1
    assert errors[0].file == "./t.tex"
    assert errors[0].line == 3
    assert errors[0].message == "Undefined control sequence."


def test_parse_warnings_collect_line_numbers():
    issues = latex_workflow.parse_log(WARN_LOG)
    warnings = [i for i in issues if i.severity == "warning"]
    assert len(warnings) == 2
    assert warnings[0].message.startswith("Reference `fig:1'")
    assert warnings[0].line == 12
    assert warnings[1].message.startswith("something odd")
    assert warnings[1].line == 34
    assert all(i.severity == "warning" for i in issues)


def test_parse_log_noise_no_issues():
    assert latex_workflow.parse_log("This is pdfTeX\nNo pages of output.\n") == []


def test_parse_chktex_output():
    text = "t.tex:3:20: Command terminated with space.\nt.tex:9:5: Wrong length."
    issues = latex_workflow.parse_chktex_output(text)
    assert len(issues) == 2
    assert all(i.severity == "warning" for i in issues)
    assert (issues[0].file, issues[0].line) == ("t.tex", 3)
    assert issues[0].message == "Command terminated with space."
    assert (issues[1].file, issues[1].line) == ("t.tex", 9)


def test_parse_chktex_output_ignores_noise():
    assert latex_workflow.parse_chktex_output("chktex warning banner\n") == []


def test_lint_command_uses_real_newline_separator():
    # 回归（2026-11-17 冒烟）：chktex -f 格式串的记录分隔必须是真换行符；
    # 字面 `\n` 两字符会让多条 warning 并成一条、后续 file:line:col 被吞进 message。
    cmd = latex_workflow.lint_command("t.tex")
    fmt = cmd[cmd.index("-f") + 1]
    assert fmt.endswith("\n") and "\\n" not in fmt


def test_build_rejects_unknown_engine(tmp_path):
    with pytest.raises(latex_workflow.LatexError, match="引擎"):
        latex_workflow.build(tmp_path / "x.tex", engine="typst")


@_needs_tex
@_needs_texlive
def test_build_ok_document(tmp_path):
    tex = tmp_path / "ok.tex"
    tex.write_text(
        "\\documentclass{article}\n\\begin{document}\nHello.\n\\end{document}\n",
        encoding="utf-8",
    )
    result = latex_workflow.build(tex, run_lint=True)
    assert result.ok and result.pdf is not None and result.pdf.is_file()
    assert not any(i.severity == "error" for i in result.issues)


@_needs_tex
@_needs_texlive
def test_build_reports_error_location(tmp_path):
    tex = tmp_path / "bad.tex"
    tex.write_text(
        "\\documentclass{article}\n\\begin{document}\n\\undefinedcmd\n\\end{document}\n",
        encoding="utf-8",
    )
    result = latex_workflow.build(tex, run_lint=False)
    assert not result.ok and result.pdf is None
    errors = [i for i in result.issues if i.severity == "error"]
    assert errors and errors[0].line is not None
    assert "Undefined control sequence" in errors[0].message


@_needs_tex
@_needs_texlive
def test_chktex_lint_smoke(tmp_path):
    tex = tmp_path / "lint.tex"
    tex.write_text(
        "\\documentclass{article}\n\\begin{document}\nA. B\n\\end{document}\n",
        encoding="utf-8",
    )
    issues = latex_workflow.lint(tex)
    assert all(i.severity == "warning" for i in issues)
