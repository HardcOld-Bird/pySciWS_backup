"""pandoc 包装机械断言：命令拼装（纯）+ 实机 smoke + pin 版本断言。

实机用例依赖 vendored ``pandoc.exe``（deps/tools/pandoc/pandoc-3.14）；缺失时跳过，
命令拼装断言不受影响。
"""

from __future__ import annotations

import subprocess

import pytest

from pysci.skills.document_writing import pandoc_wrapper
from pysci.skills.document_writing.toolpaths import PANDOC_EXE, pandoc_exe

_need_pandoc = pytest.mark.skipif(
    not PANDOC_EXE.is_file(), reason="vendored pandoc 未落盘（未跑 init-deps）"
)


def test_build_command_basic_shape():
    cmd = pandoc_wrapper.build_command("in.md", to="docx", output="out.docx")
    assert cmd[0] == pandoc_exe()
    assert cmd[1:4] == ["in.md", "-t", "docx"]
    assert cmd[-2:] == ["-o", "out.docx"]


def test_build_command_full_options():
    cmd = pandoc_wrapper.build_command(
        "s.md",
        to="pptx",
        output="s.pptx",
        reference_doc="house.pptx",
        slide_level=2,
        toc=True,
        number_sections=True,
        citeproc=True,
        bib="refs.bib",
        csl="style.csl",
        resource_path="figs",
        standalone=True,
    )
    joined = " ".join(cmd)
    assert "--reference-doc house.pptx" in joined
    assert "--slide-level 2" in joined
    assert "--toc" in joined and "--number-sections" in joined
    assert "--bibliography refs.bib" in joined
    assert "--csl style.csl" in joined
    assert "--citeproc" in joined
    assert "--resource-path figs" in joined
    assert "--standalone" in joined


def test_citeproc_without_bib_raises():
    with pytest.raises(ValueError, match="citeproc"):
        pandoc_wrapper.build_command("in.md", citeproc=True)


def test_no_output_targets_stdout():
    cmd = pandoc_wrapper.build_command("in.md", to="html")
    assert "-o" not in cmd


@_need_pandoc
def test_pandoc_version_pin_3_14():
    """pin 版本断言：选型定稿 pandoc 3.14（pptx/xlsx reader 依赖该系列）。"""
    result = subprocess.run(
        [pandoc_exe(), "--version"], capture_output=True, text=True, encoding="utf-8"
    )
    assert result.stdout.splitlines()[0].startswith("pandoc 3.14")


@_need_pandoc
def test_convert_md_to_docx_smoke(tmp_path):
    src = tmp_path / "doc.md"
    src.write_text("# 标题\n\n正文 **加粗**。\n", encoding="utf-8")
    out = tmp_path / "doc.docx"
    got = pandoc_wrapper.convert(src, to="docx", output=out)
    assert got == out and out.is_file() and out.stat().st_size > 0


@_need_pandoc
def test_convert_failure_raises(tmp_path):
    with pytest.raises(pandoc_wrapper.PandocError):
        pandoc_wrapper.convert(tmp_path / "missing.md", to="docx",
                               output=tmp_path / "o.docx")
