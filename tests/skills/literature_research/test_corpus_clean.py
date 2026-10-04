"""``corpus_clean``（rag 语料的删除式规范化）离线单测。

被测契约分三层：

- **三条删除规则各自独立**：REVTeX ``thebibliography`` 宏残渣、MinerU 的空 alt 图片占位行、
  连续编号参考文献表。每条都要钉住「删什么」与**「不删什么」**——后者才是容易做错的
  方向：图注（``FIG. 1. (a) …``）、段中行内引用（``$[1-8]$``）、带 alt 的图片、不足
  ``MIN_REF_BLOCK`` 条的短列表，都必须原样保留。
- **``clean_markdown`` 的组合行为**：幂等（``materialize_clean`` 的「内容未变则不重写」
  判断以此为前提）、删除后压空行（``chunk_code_text`` 按字符计预算，空行同样占额）。
- **派生副本的落盘语义**：写到 ``<dest_root>/<name>.md``、恒为 LF、内容未变时不刷新
  mtime；``reset_corpus_dir`` 的越界守卫（只肯删名为 ``corpus`` 的目录）。

全部纯本地、不触网、不依赖 paperqa。
"""

from __future__ import annotations

import os
from pathlib import Path

from pysci.skills.literature_research.tools import corpus_clean
from pysci.skills.literature_research.tools.corpus_clean import (
    CleanStats,
    clean_markdown,
    materialize_clean,
    reset_corpus_dir,
    strip_bibliography_envs,
    strip_image_placeholders,
    strip_reference_blocks,
)

# ---------------------------------------------------------------------------
#  样本：形态取自真实的 MinerU 产物（cpa_ep/higher_order_perfect_absorption_no_ep_prl.md）
# ---------------------------------------------------------------------------
#: 一篇 APS 论文的典型形态：小节标题是**行内接排**（``Introduction—…``，无 Markdown
#: 标题），行内引用是段中的 ``$[1-8]$``，图注另起一行，图片占位行是 64 位哈希。
MINERU_PAPER = """# High-Order Perfect Absorption in the Absence of Exceptional Point

Huisheng Xu $^{1}$ , Luojia Wang $^{2}$

(Received 9 October 2025; published 31 March 2026)

DOI: 10.1103/nkls-pgkf

Introduction—Coherent perfect absorption (CPA) is a hallmark non-Hermitian interference phenomenon $[1-8]$ . When properly superposed input waves arrive simultaneously, they can be completely absorbed $[9,10]$ .

![](images/2a1996510c1d32b9360d9ae00136abf963b55636ba871c7965a8a86d70337b92.jpg)
FIG. 1. (a) Coherent input without relative delay. (b) With relative delay.

Delay formalism—We briefly review the concept of Wigner delay $[22,23]$ , a fundamental scattering phenomenon.

[1] Y. D. Chong, L. Ge, H. Cao, and A. D. Stone, Coherent perfect absorbers: Time-reversed lasers, Phys. Rev. Lett. 105, 053901 (2010).

[2] S. Longhi, PT-symmetric laser absorber, Phys. Rev. A 82, 031801(R) (2010).

[3] W. Wan, Y. Chong, L. Ge, H. Noh, A. D. Stone, and H. Cao, Time-reversed lasing and interferometric control of absorption, Science 331, 889 (2011).
"""

#: arXiv LaTeX 源码路径（``pdf_extract._tex_to_markdown``）会原样留下的 REVTeX 书目环境。
REVTEX_BIB = """# Real Title

Body text stays.

\\begin{thebibliography}{2}
\\providecommand \\BibitemOpen [0]{}
\\bibitem [{\\citenamefont {Bender}\\ \\emph {et~al.}(1999)}]{bender99}
\\bibinfo {year} {1999}\\BibitemShut {NoStop}
\\end{thebibliography}
"""


#: REVTeX 书目环境里的宏残渣——只要环境没被整段删干净，它们就会泄到产物里。
_REVTEX_MACROS = (
    "bibitem",
    "citenamefont",
    "bibinfo",
    "BibitemShut",
    "providecommand",
)


# ===========================================================================
#  规则一：REVTeX 书目环境
# ===========================================================================
def test_bibliography_environment_is_removed_with_its_macros():
    out, n = strip_bibliography_envs(REVTEX_BIB)

    assert n == 1
    assert "thebibliography" not in out
    assert "Body text stays." in out
    for macro in _REVTEX_MACROS:
        assert macro not in out


def test_unterminated_bibliography_swallows_to_eof():
    """缺 ``\\end{thebibliography}`` 时吞到文件末尾。

    宁可少一段正文，也不要让半篇宏定义进 embedding——后者是**静默**的污染，
    前者至少在本函数的计数里看得见。
    """
    text = "keep\n\\begin{thebibliography}{1}\n\\bibitem{a} x\ntrailing macro soup\n"

    out, n = strip_bibliography_envs(text)

    assert n == 1
    assert out == "keep\n"


def test_text_without_a_bibliography_is_returned_unchanged():
    out, n = strip_bibliography_envs(MINERU_PAPER)

    assert n == 0
    assert out == MINERU_PAPER


# ===========================================================================
#  规则二：图片占位行
# ===========================================================================
def test_standalone_image_placeholder_lines_are_removed():
    out, n = strip_image_placeholders(MINERU_PAPER)

    assert n == 1
    assert "![](images/" not in out
    assert "2a1996510c1d32b9360d9ae00136abf963b55636ba871c7965a8a86d70337b92" not in out


def test_figure_captions_survive_the_image_pass():
    """图注常含结论性表述，是正文的一部分——删图片行不得连它一起删。

    MinerU 把图注另起一行输出成 ``FIG. 1. (a) …`` 纯文本，不匹配图片占位行的形状。
    """
    out, _ = strip_image_placeholders(MINERU_PAPER)

    assert "FIG. 1. (a) Coherent input without relative delay." in out


def test_images_with_alt_text_are_left_alone():
    """只删 alt **为空**的占位行；带 alt 的说明有人写了描述，保守起见不动。"""
    text = "before\n![FIG. 1 unit cell](images/aa.jpg)\nafter\n"

    out, n = strip_image_placeholders(text)

    assert n == 0
    assert out == text


def test_image_line_with_trailing_spaces_is_still_removed():
    """实测 COMSOL 手册里存在 ``![](images/<hash>.jpg)␣␣``（尾随空格）的形态。"""
    text = "before\n![](images/aa.jpg)   \nafter\n"

    out, n = strip_image_placeholders(text)

    assert n == 1
    assert out == "before\nafter\n"


# ===========================================================================
#  规则三：编号参考文献表
# ===========================================================================
def test_three_consecutive_numbered_references_are_removed_as_a_block():
    out, n_blocks, n_lines = strip_reference_blocks(MINERU_PAPER)

    assert n_blocks == 1
    assert n_lines == 3
    assert "Y. D. Chong" not in out and "S. Longhi" not in out and "W. Wan" not in out


def test_blank_lines_between_entries_do_not_break_the_block():
    """MinerU 常在每条参考文献之间插一个空行——那仍是一块表，不是三处孤立引用。"""
    text = "Body.\n\n[1] A, J. Foo 1, 1 (2001).\n[2] B, J. Foo 2, 2 (2002).\n[3] C, J. Foo 3, 3 (2003).\n"

    _out, n_blocks, n_lines = strip_reference_blocks(text)

    assert (n_blocks, n_lines) == (1, 3)


def test_fewer_than_min_ref_block_entries_are_kept():
    """不足 :data:`MIN_REF_BLOCK` 条时**不删**：短列表分不清是参考文献还是正文编号项。"""
    text = "Body.\n\n[1] A, J. Foo 1, 1 (2001).\n\n[2] B, J. Foo 2, 2 (2002).\n\nMore body.\n"

    out, n_blocks, n_lines = strip_reference_blocks(text)

    assert (n_blocks, n_lines) == (0, 0)
    assert out == text
    assert corpus_clean.MIN_REF_BLOCK == 3


def test_inline_citations_in_body_text_are_not_touched():
    """行内引用（``$[1-8]$`` / ``$[22,23]$``）不在行首、也不带 ``[n] `` 的形状。

    这是本规则最容易误伤的方向：MinerU 把正文里的引用渲染成段中的方括号数字，
    与参考文献条目长得像。判据是「行首 + ``[n]`` + 空白 + 非空白」且**连续三条**。
    """
    text = (
        "A hallmark phenomenon $[1-8]$ . More text $[9,10]$ here.\n\n"
        "We review the Wigner delay $[22,23]$ , a fundamental concept.\n"
    )

    out, n_blocks, _ = strip_reference_blocks(text)

    assert n_blocks == 0
    assert out == text


def test_two_separate_blocks_are_both_removed():
    """MinerU 读双栏 PDF 时会把参考文献表**切断**插到正文中间（实测某产物如此）。

    两块各自独立命中，不必相邻。
    """
    text = (
        "Body A.\n\n"
        "[1] A, J. Foo 1, 1 (2001).\n\n[2] B, J. Foo 2, 2 (2002).\n\n[3] C, J. Foo 3, 3 (2003).\n"
        "\nBody B.\n\n"
        "[4] D, J. Foo 4, 4 (2004).\n\n[5] E, J. Foo 5, 5 (2005).\n\n[6] F, J. Foo 6, 6 (2006).\n"
    )

    out, n_blocks, n_lines = strip_reference_blocks(text)

    assert (n_blocks, n_lines) == (2, 6)
    assert "Body A." in out and "Body B." in out
    assert "J. Foo" not in out


# ===========================================================================
#  组合：clean_markdown
# ===========================================================================
def test_clean_markdown_removes_all_three_noise_classes_at_once():
    text = REVTEX_BIB + "\n" + MINERU_PAPER

    out, stats = clean_markdown(text)

    assert stats.n_bib_envs == 1
    assert stats.n_image_lines == 1
    assert stats.n_ref_blocks == 1 and stats.n_ref_lines == 3
    # 三类噪声各自消失
    assert "BibitemShut" not in out
    assert "![](images/" not in out
    assert "Y. D. Chong" not in out
    # 正文一字不改
    assert "# High-Order Perfect Absorption in the Absence of Exceptional Point" in out
    assert "Introduction—Coherent perfect absorption (CPA) is a hallmark" in out
    assert "Delay formalism—We briefly review the concept of Wigner delay" in out
    assert "FIG. 1. (a) Coherent input without relative delay." in out
    assert stats.chars_in == len(text)
    assert stats.chars_out == len(out)
    assert stats.removed > 0


def test_clean_markdown_exact_output_on_a_minimal_case():
    """精确断言（含空行压缩）：删掉一行图片后留下的三连换行要压回一个空行。"""
    src = "# T\n\nBody one.\n\n![](images/aa.jpg)\n\nBody two.\n"

    out, stats = clean_markdown(src)

    assert out == "# T\n\nBody one.\n\nBody two.\n"
    assert stats.chars_in == 46 and stats.chars_out == 26
    assert stats.removed == 20 and stats.n_image_lines == 1


def test_clean_markdown_is_idempotent():
    """幂等是 ``materialize_clean``「内容未变则不重写」判断的前提，必须钉住。"""
    once, _ = clean_markdown(REVTEX_BIB + "\n" + MINERU_PAPER)
    twice, stats2 = clean_markdown(once)

    assert twice == once
    assert stats2.removed == 0
    assert (stats2.n_bib_envs, stats2.n_image_lines, stats2.n_ref_blocks) == (0, 0, 0)


def test_clean_markdown_collapses_runs_of_blank_lines():
    """``chunk_code_text`` 按**字符**计预算，删除留下的空行同样占额。"""
    src = "A.\n\n\n\n\nB.\n"

    out, _ = clean_markdown(src)

    assert out == "A.\n\nB.\n"


def test_clean_markdown_on_clean_text_is_a_noop():
    src = "# Title\n\nJust prose.\n\nMore prose.\n"

    out, stats = clean_markdown(src)

    assert out == src
    assert stats == CleanStats(chars_in=len(src), chars_out=len(src))
    assert stats.removed == 0 and stats.removed_ratio == 0.0


def test_clean_markdown_of_empty_text_does_not_divide_by_zero():
    out, stats = clean_markdown("")

    assert out == ""
    assert stats.removed_ratio == 0.0


# ===========================================================================
#  派生副本落盘
# ===========================================================================
def test_materialize_clean_writes_a_named_copy(tmp_path: Path):
    src = tmp_path / "extracted" / "cpa_ep" / "noisy.md"
    src.parent.mkdir(parents=True)
    src.write_text(MINERU_PAPER, encoding="utf-8")
    dest_root = tmp_path / "rag" / corpus_clean.CORPUS_DIRNAME

    dest, stats = materialize_clean(src, dest_root, name="cpa_ep__noisy")

    assert dest == dest_root / "cpa_ep__noisy.md"
    assert dest.exists()
    assert "![](images/" not in dest.read_text(encoding="utf-8")
    assert stats.n_image_lines == 1
    # **源文件绝不被修改**
    assert src.read_text(encoding="utf-8") == MINERU_PAPER


def test_materialize_clean_always_emits_lf_even_from_a_crlf_source(tmp_path: Path):
    """派生产物必须跨平台字节确定。

    ``read_text`` 的通用换行翻译会把 CRLF 归一成 LF，再配 ``newline="\\n"`` 写盘，
    于是同一份语料在 Windows 与 Linux 上洗出的副本一致。（少写 ``newline`` 的话，
    Windows 会把 LF 又翻回 CRLF——这个坑在 ``journal_metrics.build_scimago_index``
    上实测过一次，索引字节因此多了 1 B。）
    """
    src = tmp_path / "crlf.md"
    src.write_text(MINERU_PAPER.replace("\n", "\r\n"), encoding="utf-8", newline="")
    assert b"\r\n" in src.read_bytes()

    dest, _ = materialize_clean(src, tmp_path / "corpus", name="d")

    assert b"\r" not in dest.read_bytes()


def test_materialize_clean_does_not_rewrite_an_identical_copy(tmp_path: Path):
    """内容未变时不重写：重写会刷新 mtime，让副本看起来比源文件新，白占一次 I/O。"""
    src = tmp_path / "a.md"
    src.write_text(MINERU_PAPER, encoding="utf-8")
    dest_root = tmp_path / "corpus"
    dest, _ = materialize_clean(src, dest_root, name="a")
    old = dest.stat().st_mtime_ns
    os.utime(dest, ns=(old - 10**9, old - 10**9))  # 造一个明显偏旧的 mtime
    before = dest.stat().st_mtime_ns

    materialize_clean(src, dest_root, name="a")

    assert dest.stat().st_mtime_ns == before


def test_materialize_clean_rewrites_when_the_source_changes(tmp_path: Path):
    src = tmp_path / "a.md"
    src.write_text(MINERU_PAPER, encoding="utf-8")
    dest_root = tmp_path / "corpus"
    dest, _ = materialize_clean(src, dest_root, name="a")
    first = dest.read_text(encoding="utf-8")

    src.write_text(MINERU_PAPER + "\nA new paragraph.\n", encoding="utf-8")
    dest2, _ = materialize_clean(src, dest_root, name="a")

    assert dest2 == dest
    assert dest2.read_text(encoding="utf-8") != first
    assert "A new paragraph." in dest2.read_text(encoding="utf-8")


def test_reset_corpus_dir_removes_only_the_corpus_directory(tmp_path: Path):
    home = tmp_path / "rag"
    corpus = home / corpus_clean.CORPUS_DIRNAME
    corpus.mkdir(parents=True)
    (corpus / "a.md").write_text("x", encoding="utf-8")
    index = home / "index.pkl"
    index.write_bytes(b"pickle")

    reset_corpus_dir(corpus)

    assert not corpus.exists()
    assert index.exists(), "同级的 index.pkl / index_meta.json 绝不能被牵连"


def test_reset_corpus_dir_refuses_any_other_directory_name(tmp_path: Path):
    """越界守卫：``dest_root`` 万一被传成 ``pqa_home`` 本身，必须是 no-op 而不是事故。"""
    home = tmp_path / "rag"
    home.mkdir(parents=True)
    (home / "index.pkl").write_bytes(b"pickle")
    (home / "index_meta.json").write_text("{}", encoding="utf-8")

    reset_corpus_dir(home)

    assert home.exists()
    assert (home / "index.pkl").exists()
    assert (home / "index_meta.json").exists()


def test_reset_corpus_dir_on_a_missing_directory_is_a_noop(tmp_path: Path):
    reset_corpus_dir(tmp_path / "rag" / corpus_clean.CORPUS_DIRNAME)  # 不抛即通过
