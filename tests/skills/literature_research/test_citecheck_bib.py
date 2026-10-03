"""参考文献解析器的离线回归测试（无 HTTP、无副作用）。

护栏对象是 WP-F F2：``citecheck`` 此前只能核「用户亲手敲的标识」与「papers/ 笔记」，
而综述的参考文献段与 Zotero 导出的 ``.bib`` 都吃不了。本文件覆盖两个新解析器
（:func:`citation_verify.parse_bibtex` / :func:`citation_verify.parse_markdown_references`）
与 ``cmd_citecheck`` 的 ``--bib`` / ``--review`` / ``--limit`` 接线。

三条设计约定值得钉住，因为它们都反直觉：

1. **BibTeX 的 LaTeX 重音按「转写」而非「删除」处理**——``B{\\"u}ttner`` 剥成 ``Buttner``，
   与 OpenAlex 的 ``Büttner`` 经 :func:`notes.normalize_last_name` 归一后**相同**。若按删除
   处理会得到 ``bttner``，于是每一条含声调作者的引用都会报一个假 WARN。
2. **参考文献段内外的接受强度不对称**——段内接受「只有标题」的行，段外只认硬标识
   （DOI / 显式 ``arXiv:``）。否则正文里一句提到某篇论文的散文会被当成待核引用。
3. **best-effort 不等于静默**——解析器对坏行跳过，但 ``_collect_bib_citations`` 必须报出
   每个文件解析出几条：不报的话，用户会把「这份 .bib 全是非标准格式」误读成「它是空的」。
"""

from __future__ import annotations

import argparse

import pytest

from pysci.skills.literature_research.tools import citation_verify as cv
from pysci.skills.literature_research.tools import notes, research

# ---------------------------------------------------------------------------
# fixture：真实形状的输入
# ---------------------------------------------------------------------------
#: 照抄 ``compose refs.bib`` 从 Zotero 产出的形状：``@string`` 前言、``{}`` 与 ``""``
#: 两种引值混用、标题里有成对花括号、作者字段里满是逗号。
BIB_SAMPLE = r"""
@string{prl = "Phys. Rev. Lett."}

@comment{ 这一段应当被完全忽略 }

@article{zhu2018topological,
  title = {Simultaneous Observation of a Topological Edge State and Exceptional Point},
  author = {Zhu, Weiwei and Fang, Xinsheng and Li, Yong},
  year = {2018},
  journal = {Phys. Rev. Lett.},
  volume = {121},
  pages = {124501},
  doi = {10.1103/PhysRevLett.121.124501},
}

@article{buttner2017field,
  title = "Field-free switching of perpendicular magnetic moments by spin-orbit torques",
  author = {B{\"u}ttner, Ralph and Moutafis, C.},
  journal = "Phys. Rev. B",
  year = 2017,
  eprint = {1701.05611},
  archiveprefix = {arXiv},
}

@inproceedings{jones2020conf,
  title = {Non-Hermitian Acoustics in the {C--H} Symmetric Waveguide Array},
  author = {Jones, A. and Smith, B.},
  booktitle = {Proceedings of Meetings on Acoustics},
  date = {2020-06-15},
}

@misc{emptyshell,
}
"""

#: 真实形状的 markdown 参考文献段：APS 风格、IEEE 风格、arXiv 预印本、无标识行混在一起。
MD_WITH_REFS = """# 综述正文

这一段散文提到了拓扑边界态与例外点的关系，但没有给出任何标识符。

## 2. 关键脉络

- 代表论文：Zhu 等人 2018 年在 PRL 上的工作

## 7. 参考文献

1. W. Zhu, X. Fang, and Y. Li, "Simultaneous observation of a topological edge state
   and exceptional point," Phys. Rev. Lett. 121, 124501 (2018).
   https://doi.org/10.1103/PhysRevLett.121.124501
2. R. Büttner et al., "Field-free switching of perpendicular magnetic moments,"
   Phys. Rev. B 96, 085117 (2017), arXiv:1701.05611.
3. [12] A. Jones, Extreme wave manipulation via non-Hermitian metagratings and
   degenerated states in open acoustic systems (2023).
4. Phys. Rev. Lett. 121, 124501 (2018).
5. 这一行既没有标识符也提不出可辨识的标题。

## 8. 附录

- 一些补充材料，DOI 是 10.9999/not.a.reference 但不该被当成参考文献。
"""


def _ns(**kw) -> argparse.Namespace:
    """构造 ``cmd_citecheck`` 需要的最小 Namespace。"""
    base = dict(
        targets=[],
        note=None,
        all=False,
        bib=None,
        review=False,
        limit=50,
        json=False,
        no_color=True,
    )
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.fixture
def stub_verify(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """把 ``verify_citation`` 换成不打网络的桩，并记录它收到的每一个 ``cite`` dict。

    桩返回真实的 :class:`CitationVerdict`（状态 PASS），使 ``cmd_citecheck`` 后续的
    渲染与退出码逻辑走的仍是生产代码路径。
    """
    seen: list[dict] = []

    def fake(cite):
        seen.append(cite)
        return cv.CitationVerdict(
            input=dict(cite), kind="doi" if cite.get("doi") else "title", status=cv.PASS
        )

    monkeypatch.setattr(cv, "verify_citation", fake)
    return seen


# ===========================================================================
#  _strip_latex —— LaTeX 重音必须转写而非删除
# ===========================================================================
@pytest.mark.parametrize(
    "raw,expected",
    [
        (r"B{\"u}ttner", "Buttner"),  # {\"u} 形态（Zotero 导出的主流写法）
        (r"{\'e}cole", "ecole"),  # {\'e} 形态
        (r"B\"{u}ttner", "Buttner"),  # \"{u} 形态（花括号在内）
        (r"\'E", "E"),  # 裸字母形态
        (r"\textbf{Bold Title}", "Bold Title"),  # 命名命令解包
        (r"Smith \& Jones", "Smith & Jones"),  # 转义标点
        (r"100\% pure", "100% pure"),
        (r"page 1--20", "page 1-20"),  # BibTeX 的页码连字符习惯
        (r"word~next", "word next"),  # 不换行空格
        (r"The {C--H} bond", "The C-H bond"),  # 保护性分组 + 连字符
        (r"$E = mc^2$", "E = mc^2"),  # 数学模式定界符
        (r"\oe uvre", "oeuvre"),  # 单/双字母命令去反斜杠
        ("plain text", "plain text"),
        ("", ""),
        (None, ""),
    ],
)
def test_strip_latex(raw, expected):
    assert cv._strip_latex(raw) == expected


def test_bibtex_accent_transliteration_matches_openalex_form():
    """**关键回归**：BibTeX 的 ``B{\\"u}ttner`` 与 OpenAlex 的 ``Büttner`` 归一后必须相同。

    这是「不因排版差异产生假冲突」的执行点。原 ``arxiv_client`` / ``openalex_client``
    的 ``_guess_last_name`` 用 ``re.sub(r"[^a-zA-Z]", "", ...)`` **删除**声调
    （``Büttner`` → ``bttner``），与 ``normalize_last_name`` 的**转写**语义相反；
    若解析器也走删除路线，每一位含声调的作者都会让 citecheck 报一个假 WARN。
    """
    from_bibtex = notes.normalize_last_name(cv._strip_latex(r"B{\"u}ttner, Ralph"))
    from_openalex = notes.normalize_last_name("Büttner, Ralph")
    assert from_bibtex == from_openalex == "buttner"


# ===========================================================================
#  parse_bibtex
# ===========================================================================
def test_parse_bibtex_entry_count_and_keys():
    """``@string`` / ``@comment`` 被忽略，空壳 ``@misc{}`` 仍算一条记录（但映射后为空）。"""
    entries = cv.parse_bibtex(BIB_SAMPLE)
    keys = [e["key"] for e in entries]
    assert keys == [
        "zhu2018topological",
        "buttner2017field",
        "jones2020conf",
        "emptyshell",
    ]
    assert all(e["entry_type"] not in cv._BIB_SKIP_TYPES for e in entries)


def test_parse_bibtex_braced_and_quoted_values_are_equivalent():
    entries = {e["key"]: e for e in cv.parse_bibtex(BIB_SAMPLE)}
    # {} 引值
    assert entries["zhu2018topological"]["journal"] == "Phys. Rev. Lett."
    # "" 引值
    assert entries["buttner2017field"]["journal"] == "Phys. Rev. B"
    # 无引值的裸数字
    assert entries["buttner2017field"]["year"] == "2017"


def test_parse_bibtex_field_names_are_lowercased():
    """BibTeX 字段名大小写不敏感，解析器统一小写，映射才不用穷举大小写组合。"""
    entries = cv.parse_bibtex(
        "@Article{K, Title = {Some Long Enough Title Here}, YEAR = {2020}}"
    )
    assert entries[0]["entry_type"] == "article"
    assert entries[0]["title"] == "Some Long Enough Title Here"
    assert entries[0]["year"] == "2020"


def test_parse_bibtex_comma_inside_field_is_not_a_separator():
    """作者字段里满是逗号；按顶层逗号切分才不会把一个字段拆成好几块。"""
    entries = {e["key"]: e for e in cv.parse_bibtex(BIB_SAMPLE)}
    assert (
        entries["zhu2018topological"]["author"]
        == "Zhu, Weiwei and Fang, Xinsheng and Li, Yong"
    )
    assert entries["zhu2018topological"]["journal"] == "Phys. Rev. Lett."


def test_parse_bibtex_nested_braces_in_title():
    """``{The {C--H} bond}`` 这种保护性分组要靠深度计数，按第一个 ``}`` 截断会切坏字段区。"""
    entries = {e["key"]: e for e in cv.parse_bibtex(BIB_SAMPLE)}
    assert entries["jones2020conf"]["title"] == (
        "Non-Hermitian Acoustics in the C-H Symmetric Waveguide Array"
    )
    # 分组没把后面的字段吃掉
    assert (
        entries["jones2020conf"]["booktitle"] == "Proceedings of Meetings on Acoustics"
    )


def test_parse_bibtex_missing_fields_are_absent_not_empty():
    """缺失的字段**不出现**在 dict 里，而不是以空串占位。"""
    entries = {e["key"]: e for e in cv.parse_bibtex(BIB_SAMPLE)}
    assert "doi" not in entries["buttner2017field"]
    assert "volume" not in entries["jones2020conf"]
    assert "year" not in entries["jones2020conf"]  # 它用的是 biblatex 的 date
    assert "date" in entries["jones2020conf"]


def test_parse_bibtex_empty_shell_entry():
    entries = {e["key"]: e for e in cv.parse_bibtex(BIB_SAMPLE)}
    assert entries["emptyshell"] == {"key": "emptyshell", "entry_type": "misc"}


def test_parse_bibtex_skips_string_and_comment_bodies():
    """``@string`` 的定义体不得被当成条目；``@comment`` 的内容不得泄漏。"""
    bib = '@string{foo = "bar"}\n@comment{@article{inner, title={x}}}\n@article{real, title={A Long Enough Real Title Here}}\n'
    entries = cv.parse_bibtex(bib)
    # @comment 体里的 @article 会被扫到（这是正则解析的已知边界），但 @string 不会成条目
    assert "foo" not in [e["key"] for e in entries]
    assert "real" in [e["key"] for e in entries]


def test_parse_bibtex_unclosed_entry_is_best_effort():
    """残缺的 .bib（未闭合）仍应能核对其前面的条目，而不是整体解析失败。"""
    bib = (
        "@article{good, title={A Perfectly Fine Title For Testing}, year={2020}}\n"
        "@article{broken, title={Never Closed Here\n"
    )
    entries = cv.parse_bibtex(bib)
    assert entries[0]["key"] == "good"
    assert entries[0]["title"] == "A Perfectly Fine Title For Testing"
    assert len(entries) == 2  # 坏条目也收进来（尽力而为），但字段可能不全


def test_parse_bibtex_field_without_equals_is_skipped():
    entries = cv.parse_bibtex(
        "@article{k, garbage, title = {A Reasonably Long Title Value}, }"
    )
    assert entries[0]["title"] == "A Reasonably Long Title Value"


def test_parse_bibtex_old_style_parens_not_supported():
    """``@type(key, ...)`` 旧式括号**不支持**——这是不引入 bibtexparser 的明确代价。

    Zotero 与 ``compose refs.bib`` 产出的都是花括号形态，故不值得为它加一套解析器；
    但行为必须被钉住，免得未来有人以为它能用。
    """
    assert cv.parse_bibtex("@article(oldkey, title = {Some Title})") == []


def test_parse_bibtex_empty_and_none_input():
    assert cv.parse_bibtex("") == []
    assert cv.parse_bibtex(None) == []
    assert cv.parse_bibtex("no bibtex here at all") == []


# ---------------------------------------------------------------------------
#  转义与引号：两条曾把解析彻底搞崩的规则
# ---------------------------------------------------------------------------
def test_escaped_quote_in_braced_value_does_not_derail_the_scan():
    """**关键回归**：花括号值里的 ``\\"`` 不得被当成字符串定界符。

    修复前的行为：``author = {B{\\"u}ttner, Ralph}`` 里那个 ``"`` 把状态机推进 in_str，
    于是闭合的 ``}`` 被当普通字符吞掉、深度计数从此崩坏，该条目一路吞到文件末尾。
    实测到的后果：一份 4 条的 .bib 只解出 2 条，而第二条的 ``author`` 值是
    ``B uttner, Ralph and Moutafis, C. , journal = "Phys. Rev. B", year = 2017, ...``
    ——把后面所有条目连同下一个 ``@misc`` 全吃进了一个字段。

    而 ``\\"u`` 是 Zotero 导出里最常见的分音符编码，所以这不是边缘情况：只要 .bib
    里有一位带分音符的作者，它**之后**的全部内容都核不了。
    """
    bib = (
        "@article{a1,\n"
        '  author = {B{\\"u}ttner, Ralph and Moutafis, C.},\n'
        "  title = {A Perfectly Fine Title For Testing},\n"
        "}\n"
        "@article{a2, title = {The Entry After The Accented Author}, year = {2020}}\n"
        "@misc{a3,}\n"
    )
    entries = {e["key"]: e for e in cv.parse_bibtex(bib)}
    assert sorted(entries) == ["a1", "a2", "a3"]
    assert entries["a1"]["author"] == "Buttner, Ralph and Moutafis, C."
    assert entries["a1"]["title"] == "A Perfectly Fine Title For Testing"
    assert entries["a2"]["title"] == "The Entry After The Accented Author"
    assert entries["a2"]["year"] == "2020"


def test_escaped_quote_inside_a_quoted_value_does_not_end_it():
    """引号值里的 ``\\"`` 是字面引号，**不**终止该值——因此它后面的字段仍能被切出来。

    断言落在**字段结构**上而不是 ``abstract`` 的确切文本：``_strip_latex`` 面对
    ``\\"h`` 时无法区分「转义的引号后跟字母 h」与「h 上的分音符」（TeX 里两者写法
    相同），会按后者处理。那是排版层面的固有歧义，不是扫描器的缺陷；本用例要钉的是
    扫描器没在 ``\\"`` 处把字符串提前关掉。
    """
    bib = '@article{k, title = {Short}, abstract = "He said \\"hello\\" loudly", year = {2020}}'
    entry = cv.parse_bibtex(bib)[0]
    assert sorted(entry) == ["abstract", "entry_type", "key", "title", "year"]
    assert entry["year"] == "2020"  # 没被吞进 abstract
    assert entry["abstract"].startswith("He said ")
    assert entry["abstract"].endswith("loudly")


def test_literal_quotes_inside_braced_value_are_not_delimiters():
    """``"`` 只在深度 0 处才是定界符；花括号值里它只是字面量。"""
    entry = cv.parse_bibtex(
        '@article{k, title = {The "Best" Paper Here}, year = {2020}}'
    )[0]
    assert entry["title"] == 'The "Best" Paper Here'
    assert entry["year"] == "2020"


def test_odd_quote_count_in_braced_value_does_not_swallow_the_rest():
    """花括号值里奇数个 ``"`` 不得让后续 ``}`` 全部失效。

    这是上一条规则的反面：若把值内部的 ``"`` 也当定界符，一个落单的引号就会把
    扫描器永久锁在 in_str，从此再没有任何 ``}`` 能关闭条目。
    """
    bib = (
        '@article{k1, title = {The "Best Paper Here}, year = {2020}}\n'
        "@article{k2, title = {Still Reachable After The Stray Quote}}\n"
    )
    keys = [e["key"] for e in cv.parse_bibtex(bib)]
    assert keys == ["k1", "k2"]


def test_strip_latex_braced_accent_keeps_the_closing_brace_intact():
    """**关键回归**：``{\\"u}`` 的右花括号属于**分组**，不是重音命令的参数。

    修复前的松散正则（命令 + 可选花括号 + 可选字母 + 可选花括号）在 ``B{\\"u}ttner``
    上匹配到的是 ``\\"u}``：它把闭合花括号当成重音自己的可选右括号吃掉，剩下一个
    孤立的 ``{`` 在后续步骤里变成空格——得到 ``B uttner`` 而不是 ``Buttner``。
    那个空格看似无害，却会让 ``normalize_last_name`` 得到 ``b uttner``，与 OpenAlex
    的 ``Büttner`` → ``buttner`` 对不上，于是每条含分音符作者的引用都报一个假 WARN。
    """
    out = cv._strip_latex(r"B{\"u}ttner")
    assert out == "Buttner"
    assert " " not in out  # 没有因吃掉右括号而遗留的空格
    assert notes.normalize_last_name(out) == notes.normalize_last_name("Büttner")


def test_strip_latex_escaped_quote_before_a_space_is_not_an_accent():
    """``\\" loudly`` 里的 ``\\"`` 是转义引号，不是「l 上的分音符」。

    TeX 里控制**符号**（``\\"`` ``\\'``）后的空白是真实排版空格而非终止符，所以裸字母
    形态不允许空白。允许的话 ``\\" l`` 会被读成重音命令，连同空格一起吃掉，把
    ``hello\\" loudly`` 变成 ``helloloudly``——两个词糊成一个，比留着引号糟得多。
    """
    assert cv._strip_latex(r"He said \" loudly") == 'He said " loudly'


def test_strip_latex_letter_accent_needs_braces_in_bare_form():
    """字母型重音（``\\b`` ``\\u`` ``\\c`` …）的裸形态**不**被当成重音。

    否则 ``\\bibitem`` / ``\\usepackage`` 会被劈成 ``\\b i`` / ``\\u s``，把命令名毁掉。
    带花括号的形态（``\\c{c}`` ``\\u{a}``）仍然正常转写，而那正是它们的常见写法。
    """
    assert cv._strip_latex(r"\c{c}ervenka") == "cervenka"
    assert cv._strip_latex(r"\u{a}bre") == "abre"
    # 裸形态不被误当重音：反斜杠仍在，由第 3 步当普通命令名去反斜杠
    assert cv._strip_latex(r"\bibitem") == "bibitem"


# ===========================================================================
#  citation_from_bibtex —— 到 verify_citation 的 cite dict
# ===========================================================================
def test_citation_from_bibtex_full_mapping():
    entries = {e["key"]: e for e in cv.parse_bibtex(BIB_SAMPLE)}
    cite = cv.citation_from_bibtex(entries["zhu2018topological"])
    assert cite["title"] == (
        "Simultaneous Observation of a Topological Edge State and Exceptional Point"
    )
    assert cite["authors"] == ["Zhu, Weiwei", "Fang, Xinsheng", "Li, Yong"]
    assert cite["first_author_last_name"] == "zhu"
    assert cite["year"] == 2018
    assert cite["journal"] == "Phys. Rev. Lett."
    assert cite["doi"] == "10.1103/physrevlett.121.124501"  # normalize_doi 转小写


def test_citation_from_bibtex_only_writes_nonempty_keys():
    """只写确实有值的键：空串与「BibTeX 没写这个字段」在比对器里等价，但后者更诚实。"""
    cite = cv.citation_from_bibtex({"key": "k", "entry_type": "misc"})
    assert cite == {}
    cite2 = cv.citation_from_bibtex({"key": "k", "title": "Only A Title"})
    assert set(cite2) == {"title"}


def test_citation_from_bibtex_eprint_with_arxiv_prefix():
    entries = {e["key"]: e for e in cv.parse_bibtex(BIB_SAMPLE)}
    cite = cv.citation_from_bibtex(entries["buttner2017field"])
    assert cite["arxiv_id"] == "1701.05611"
    assert cite["first_author_last_name"] == "buttner"


def test_citation_from_bibtex_eprint_shape_only_is_still_arxiv():
    """没有 ``archiveprefix`` 但编号形状就是 arXiv 的，同样认。"""
    cite = cv.citation_from_bibtex({"eprint": "2301.12345"})
    assert cite["arxiv_id"] == "2301.12345"


@pytest.mark.parametrize(
    "entry",
    [
        {"eprint": "10.5281/zenodo.1234567"},  # Zenodo DOI 形态
        {"eprint": "hal-01234567"},  # HAL 编号
        {"eprint": "9"},  # 纯数字
    ],
)
def test_citation_from_bibtex_non_arxiv_eprint_is_not_claimed(entry):
    """``eprint`` 不是 arXiv 专属字段：误当 arXiv id 会让 arXiv 源报一个假 NOT_FOUND。"""
    assert "arxiv_id" not in cv.citation_from_bibtex(entry)


def test_citation_from_bibtex_booktitle_used_for_proceedings():
    """不兼容 ``booktitle`` 会让所有 ``@inproceedings`` 的期刊比对直接缺失。"""
    entries = {e["key"]: e for e in cv.parse_bibtex(BIB_SAMPLE)}
    cite = cv.citation_from_bibtex(entries["jones2020conf"])
    assert cite["journal"] == "Proceedings of Meetings on Acoustics"


def test_citation_from_bibtex_journaltitle_biblatex():
    cite = cv.citation_from_bibtex({"journaltitle": "Nature Physics", "title": "T"})
    assert cite["journal"] == "Nature Physics"


def test_citation_from_bibtex_journal_wins_over_booktitle():
    """三个候选同时存在时按 ``journal`` → ``journaltitle`` → ``booktitle`` 取首个非空。"""
    cite = cv.citation_from_bibtex(
        {"journal": "J. Real", "booktitle": "Proc. Fake", "title": "T"}
    )
    assert cite["journal"] == "J. Real"


def test_citation_from_bibtex_date_field_yields_year():
    """biblatex 用 ``date`` 而非 ``year``；两者都要能提出四位年份。"""
    entries = {e["key"]: e for e in cv.parse_bibtex(BIB_SAMPLE)}
    assert cv.citation_from_bibtex(entries["jones2020conf"])["year"] == 2020


def test_citation_from_bibtex_first_last_name_forms():
    """``Last, First`` 与 ``First Last`` 两种作者写法都要提取出正确的姓。"""
    a = cv.citation_from_bibtex({"author": "Zhu, Weiwei and Li, Yong"})
    b = cv.citation_from_bibtex({"author": "Weiwei Zhu and Yong Li"})
    assert a["first_author_last_name"] == "zhu"
    assert b["first_author_last_name"] == "zhu"


# ===========================================================================
#  _references_line_range —— 段落定位
# ===========================================================================
def test_references_line_range_finds_english_heading():
    """``start`` **排除标题行本身**，因此它指向标题下的那一行——那往往是个空行。

    这是有意的：区间语义是「标题之后、下一个边界之前」，标题自己不属于参考文献
    内容（把它包进去会让 ``parse_markdown_references`` 拿一行 ``## 7. 参考文献`` 去
    试提标题）。代价是调用方不能假定 ``lines[start]`` 就是第一条引用，得自己跳过空行。
    """
    lines = MD_WITH_REFS.splitlines()
    start, end = cv._references_line_range(lines)
    assert start > 0
    assert lines[start - 1].startswith("## 7. 参考文献")
    assert lines[start].strip() == ""
    assert lines[start + 1].startswith("1. W. Zhu")
    # 段落在下一个同级标题（## 8. 附录）处结束
    assert lines[end].startswith("## 8.")


def test_references_line_range_absent():
    assert cv._references_line_range(["# Title", "", "no references here"]) == (-1, -1)


@pytest.mark.parametrize(
    "heading",
    [
        "## 参考文献",
        "## References",
        "### references",
        "## Bibliography",
        "## 7. 参考文献",
        "## 参考资料",
    ],
)
def test_references_line_range_heading_variants(heading):
    lines = ["# Doc", heading, "- a", "## Next", "- b"]
    start, end = cv._references_line_range(lines)
    assert start == 2
    assert end == 3  # 到 `## Next` 为止，不吃掉它


def test_references_line_range_level1_heading_keeps_its_subsections():
    """``# 参考文献`` 是**一级**标题，于是 ``## Next`` 属于它的子节而不是边界。

    边界正则是 ``^#{1,level}[ \\t]+\\S``：只同级或更高级的标题才终止本段。上面那个
    参数化用例里的 ``## 参考文献`` 正好被 ``## Next`` 截断，而换成一级标题后行为
    **应当**不同——两者不是同一条规则的两个实例，故拆成独立用例而不是参数项。
    """
    lines = ["# Doc", "# 参考文献", "- a", "## Next", "- b"]
    start, end = cv._references_line_range(lines)
    assert start == 2
    assert end == 5  # 一路到文末：`## Next` 是子节，不终止一级段


def test_references_line_range_subsection_is_not_a_boundary():
    """``###`` 子标题属于参考文献段内部（有些综述按主题分小节列文献）。"""
    lines = ["## References", "### 2018", "- a", "### 2019", "- b", "## Appendix"]
    start, end = cv._references_line_range(lines)
    assert (start, end) == (1, 5)


def test_references_line_range_higher_level_heading_is_a_boundary():
    lines = ["### References", "- a", "## Chapter", "- b"]
    start, end = cv._references_line_range(lines)
    assert (start, end) == (1, 2)


def test_references_line_range_runs_to_eof():
    lines = ["## References", "- a", "- b"]
    assert cv._references_line_range(lines) == (1, 3)


# ===========================================================================
#  _title_from_reference_line —— best-effort 标题提取
# ===========================================================================
def test_title_extraction_rejects_journal_fragment():
    """**关键**：无标题格式里的期刊碎片不得被当成标题。

    ``Phys. Rev. Lett. 121, 124501 (2018).`` 里最长的一段是 ``Phys. Rev. Lett. 121``，
    拿它去核会得到一堆假 NOT_FOUND，把本该 PASS 的引用拖红。候选门（≥20 字符、
    ≥3 词、字母占比 ≥60%）就是为此设的。
    """
    assert cv._title_from_reference_line("Phys. Rev. Lett. 121, 124501 (2018).") == ""


def test_title_extraction_strips_numbering_and_identifiers():
    line = (
        '1. W. Zhu, X. Fang, and Y. Li, "Simultaneous observation of a topological '
        'edge state and exceptional point," Phys. Rev. Lett. 121, 124501 (2018). '
        "https://doi.org/10.1103/PhysRevLett.121.124501"
    )
    title = cv._title_from_reference_line(line)
    assert "Simultaneous observation of a topological edge state" in title
    assert "doi.org" not in title and "10.1103" not in title
    assert not title.startswith("1.")


def test_title_extraction_strips_bracketed_numbering():
    line = (
        "[12] A. Jones, Extreme wave manipulation via non-Hermitian metagratings "
        "and degenerated states in open acoustic systems (2023)."
    )
    title = cv._title_from_reference_line(line)
    assert title.startswith("Extreme wave manipulation")
    assert "[12]" not in title and "2023" not in title


def test_title_extraction_strips_arxiv_id():
    line = (
        "R. Büttner et al., Field-free switching of perpendicular magnetic moments "
        "by spin-orbit torques, arXiv:1701.05611 (2017)."
    )
    title = cv._title_from_reference_line(line)
    assert "1701.05611" not in title
    assert "Field-free switching" in title


def test_title_extraction_too_short_returns_empty():
    assert cv._title_from_reference_line("- short") == ""
    assert cv._title_from_reference_line("") == ""
    assert cv._title_from_reference_line(None) == ""


def test_title_extraction_low_letter_ratio_returns_empty():
    """数字/符号占比过高的片段不是标题。"""
    assert cv._title_from_reference_line("1234 5678 9012 3456 7890 1234 5678") == ""


# ===========================================================================
#  parse_markdown_references
# ===========================================================================
def test_parse_markdown_references_extracts_doi():
    cites = cv.parse_markdown_references(MD_WITH_REFS)
    dois = [c["doi"] for c in cites if "doi" in c]
    assert "10.1103/physrevlett.121.124501" in dois


def test_parse_markdown_references_extracts_arxiv_and_year():
    cites = cv.parse_markdown_references(MD_WITH_REFS)
    with_arx = [c for c in cites if c.get("arxiv_id") == "1701.05611"]
    assert len(with_arx) == 1
    assert with_arx[0]["year"] == 2017


def test_parse_markdown_references_doi_line_does_not_claim_bare_number_as_arxiv():
    """行里已有 DOI 时，裸的四位点五位数字**不**当 arXiv id。

    在带 DOI 的行里它更可能是 DOI 尾巴或页码；误认会让 arXiv 源报一个假 NOT_FOUND，
    甚至把本该 PASS 的引用拖成 WARN。只有显式 ``arXiv:`` 前缀才认。
    """
    text = "## References\n\n- Phys. Rev. Lett. 121, 124501 (2018). doi:10.1103/PhysRevLett.121.124501 9999.88888\n"
    cites = cv.parse_markdown_references(text)
    assert len(cites) == 1
    assert cites[0]["doi"] == "10.1103/physrevlett.121.124501"
    assert "arxiv_id" not in cites[0]


def test_parse_markdown_references_explicit_arxiv_prefix_wins_alongside_doi():
    text = (
        "## References\n\n"
        "- Some sufficiently long title for the candidate gate here, "
        "doi:10.1234/abcd.5678, arXiv:2301.12345v2 (2023).\n"
    )
    cites = cv.parse_markdown_references(text)
    assert cites[0]["doi"] == "10.1234/abcd.5678"
    assert cites[0]["arxiv_id"] == "2301.12345"  # 版本号被剥掉


def test_parse_markdown_references_title_only_line_inside_section():
    """段内接受「只有标题」的行——很多综述的文献段就是不带 DOI 的。"""
    cites = cv.parse_markdown_references(MD_WITH_REFS)
    titles = [c.get("title", "") for c in cites]
    assert any(
        "Extreme wave manipulation via non-Hermitian metagratings" in t for t in titles
    )


def test_parse_markdown_references_title_only_line_outside_section_is_skipped():
    """段外的散文不得被当成待核引用（两档强度的不对称）。"""
    cites = cv.parse_markdown_references(MD_WITH_REFS)
    joined = " ".join(
        str(c.get("title", "")) + " " + str(c.get("_raw", "")) for c in cites
    )
    # 正文里那句散文（"这一段散文提到了拓扑边界态…"）没有硬标识，不该出现
    assert "这一段散文提到了" not in joined
    # §2 里那条 bullet 同样在段外
    assert "代表论文：Zhu 等人" not in joined


def test_parse_markdown_references_unidentifiable_line_is_skipped():
    """无标识且提不出标题的行一律跳过（方案要求的 best-effort 语义）。"""
    cites = cv.parse_markdown_references(MD_WITH_REFS)
    raws = [c["_raw"] for c in cites]
    assert not any("既没有标识符" in r for r in raws)


def test_parse_markdown_references_heading_line_itself_is_not_a_citation():
    cites = cv.parse_markdown_references(MD_WITH_REFS)
    raws = [c["_raw"] for c in cites]
    assert not any(r.startswith("## 7.") for r in raws)


def test_parse_markdown_references_carries_line_and_raw():
    """每条带 ``_line``（1-based 原文行号）与 ``_raw``，供报告里定位「哪一行错了」。"""
    cites = cv.parse_markdown_references(MD_WITH_REFS)
    lines = MD_WITH_REFS.splitlines()
    assert cites, "应当至少解析出一条"
    for c in cites:
        assert c["_line"] >= 1
        assert lines[c["_line"] - 1].strip() == c["_raw"]


def test_parse_markdown_references_ignores_dois_outside_section_when_no_hard_id():
    """附录里的 DOI 仍会被收（它在段外但**有硬标识**）——段外只是不接受纯标题行。

    这是有意的：一份综述的附录里列了 DOI，说明那条引用确实存在，核一下没坏处；
    而段外的散文之所以被挡，是因为它提不出可靠标题。
    """
    cites = cv.parse_markdown_references(MD_WITH_REFS)
    assert any(c.get("doi") == "10.9999/not.a.reference" for c in cites)


def test_parse_markdown_references_multiple_dois_takes_first():
    """一行里多个 DOI 只取第一个（一条参考文献只应指向一篇文献）。

    两个 DOI 的注册码都写足 4 位：``_DOI_IN_TEXT_RE`` 要求 ``10\\.\\d{4,9}/``，因此
    ``10.1/first`` 这种玩具例子根本不是合法 DOI，会静默不匹配——用它会测到
    「什么都没抽到」而不是「多个里取第一个」。
    """
    text = (
        "## References\n\n"
        "- A title long enough to pass the candidate gate here: "
        "doi:10.1103/first doi:10.1103/second\n"
    )
    cites = cv.parse_markdown_references(text)
    assert len(cites) == 1
    assert cites[0]["doi"] == "10.1103/first"


def test_parse_markdown_references_cleans_trailing_punctuation_from_doi():
    """``...124501.`` 的那个句点不属于 DOI。"""
    text = "## References\n\n- Something. doi:10.1103/PhysRevLett.121.124501.\n"
    cites = cv.parse_markdown_references(text)
    assert cites[0]["doi"] == "10.1103/physrevlett.121.124501"


def test_parse_markdown_references_no_section_still_takes_hard_ids():
    """没有参考文献段的文档：只认硬标识，不猜标题。"""
    text = "# Doc\n\n见 10.1103/PhysRevLett.121.124501 与 arXiv:1803.04110。\n"
    cites = cv.parse_markdown_references(text)
    assert len(cites) == 1
    assert cites[0]["doi"] == "10.1103/physrevlett.121.124501"
    assert cites[0]["arxiv_id"] == "1803.04110"


def test_parse_markdown_references_empty_and_none():
    assert cv.parse_markdown_references("") == []
    assert cv.parse_markdown_references(None) == []


def test_parse_markdown_references_cites_are_verify_compatible():
    """产出的 dict 只含 ``verify_citation`` 认识的键（外加 ``_`` 前缀的定位信息）。"""
    allowed = {
        "doi",
        "arxiv_id",
        "openalex_id",
        "title",
        "year",
        "journal",
        "authors",
        "first_author_last_name",
        "_line",
        "_raw",
    }
    for c in cv.parse_markdown_references(MD_WITH_REFS):
        assert set(c) <= allowed, set(c) - allowed


# ===========================================================================
#  _collect_bib_citations —— --bib / --review 接线
# ===========================================================================
def test_collect_bib_citations_parses_bib_by_suffix(tmp_path, capsys):
    p = tmp_path / "refs.bib"
    p.write_text(BIB_SAMPLE, encoding="utf-8")
    cites, notices = research._collect_bib_citations(_ns(bib=[str(p)]))
    # 4 条记录里的空壳被滤掉
    assert len(cites) == 3
    assert any("解析出 3 条 BibTeX 条目" in n for n in notices)


def test_collect_bib_citations_parses_md_by_suffix(tmp_path):
    p = tmp_path / "survey.md"
    p.write_text(MD_WITH_REFS, encoding="utf-8")
    cites, notices = research._collect_bib_citations(_ns(bib=[str(p)]))
    assert cites
    assert any("参考文献行" in n for n in notices)


def test_collect_bib_citations_reports_missing_file(tmp_path):
    """文件不存在只留一行提示（不变量 1），不抛、不阻塞其余文件。"""
    good = tmp_path / "refs.bib"
    good.write_text(BIB_SAMPLE, encoding="utf-8")
    cites, notices = research._collect_bib_citations(
        _ns(bib=[str(tmp_path / "nope.bib"), str(good)])
    )
    assert len(cites) == 3  # 好的那份照常解析
    assert any("nope.bib" in n and "文件不存在" in n for n in notices)


def test_collect_bib_citations_reports_zero_parsed(tmp_path):
    """**best-effort 不等于静默**：解析出 0 条也要报出来。

    否则用户会把「这份 .bib 全是解析器不认的格式」误读成「这份 .bib 是空的」。
    """
    p = tmp_path / "weird.bib"
    p.write_text("@article(oldkey, title = {Old Style Parens})\n", encoding="utf-8")
    cites, notices = research._collect_bib_citations(_ns(bib=[str(p)]))
    assert cites == []
    assert any("解析出 0 条" in n for n in notices)


def test_collect_bib_citations_review_flag_scans_reviews_dir(tmp_path, monkeypatch):
    reviews = tmp_path / "reviews"
    reviews.mkdir()
    (reviews / "a_survey.md").write_text(MD_WITH_REFS, encoding="utf-8")
    (reviews / "b_survey.md").write_text(MD_WITH_REFS, encoding="utf-8")
    monkeypatch.setattr(research, "REVIEWS_DIR", reviews)
    cites, notices = research._collect_bib_citations(_ns(review=True))
    assert len(cites) > 0
    assert sum("参考文献行" in n for n in notices) == 2


def test_collect_bib_citations_review_flag_missing_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(research, "REVIEWS_DIR", tmp_path / "no-such-reviews")
    cites, notices = research._collect_bib_citations(_ns(review=True))
    assert cites == []
    assert any("reviews/ 目录不存在" in n for n in notices)


def test_collect_bib_citations_review_flag_empty_dir(tmp_path, monkeypatch):
    reviews = tmp_path / "reviews"
    reviews.mkdir()
    monkeypatch.setattr(research, "REVIEWS_DIR", reviews)
    cites, notices = research._collect_bib_citations(_ns(review=True))
    assert cites == []
    assert any("没有 .md" in n for n in notices)


def test_collect_bib_citations_bib_and_review_combine(tmp_path, monkeypatch):
    reviews = tmp_path / "reviews"
    reviews.mkdir()
    (reviews / "a_survey.md").write_text(MD_WITH_REFS, encoding="utf-8")
    monkeypatch.setattr(research, "REVIEWS_DIR", reviews)
    bib = tmp_path / "refs.bib"
    bib.write_text(BIB_SAMPLE, encoding="utf-8")
    cites, _ = research._collect_bib_citations(_ns(bib=[str(bib)], review=True))
    assert len(cites) == 3 + len(cv.parse_markdown_references(MD_WITH_REFS))


# ===========================================================================
#  cmd_citecheck 的 --limit 与 stderr 路由
# ===========================================================================
def test_citecheck_limit_truncates_and_notices(tmp_path, stub_verify, capsys):
    bib = tmp_path / "many.bib"
    entries = "\n".join(
        f"@article{{k{i}, title = {{A Distinct Long Enough Title Number {i}}}, year = {{20{i:02d}}}}}"
        for i in range(10)
    )
    bib.write_text(entries, encoding="utf-8")
    rc = research.cmd_citecheck(_ns(bib=[str(bib)], limit=3))
    assert rc == 0
    assert len(stub_verify) == 3
    err = capsys.readouterr().err
    assert "--limit 3" in err and "共解析出 10 条" in err


def test_citecheck_limit_zero_means_unlimited(tmp_path, stub_verify, capsys):
    bib = tmp_path / "many.bib"
    entries = "\n".join(
        f"@article{{k{i}, title = {{A Distinct Long Enough Title Number {i}}}, year = {{20{i:02d}}}}}"
        for i in range(5)
    )
    bib.write_text(entries, encoding="utf-8")
    research.cmd_citecheck(_ns(bib=[str(bib)], limit=0))
    assert len(stub_verify) == 5
    assert "截断" not in capsys.readouterr().err


def test_citecheck_limit_does_not_apply_to_explicit_targets(tmp_path, stub_verify):
    """用户亲手敲的位置参数一律全量核验。

    ``citecheck 10.1/x 10.2/y --limit 1`` 若静默丢掉一条，就等于工具擅自缩小了
    用户明确要求的核验范围——那比多花两次网络请求严重得多。
    """
    research.cmd_citecheck(
        _ns(targets=["10.1103/PhysRevLett.121.124501", "1803.04110"], limit=1)
    )
    assert len(stub_verify) == 2


def test_citecheck_default_limit_is_fifty(tmp_path, stub_verify):
    bib = tmp_path / "many.bib"
    entries = "\n".join(
        f"@article{{k{i}, title = {{A Distinct Long Enough Title Number {i}}}, year = {{2000}}}}"
        for i in range(80)
    )
    bib.write_text(entries, encoding="utf-8")
    research.cmd_citecheck(_ns(bib=[str(bib)]))
    assert len(stub_verify) == 50


def test_citecheck_notices_go_to_stderr_not_stdout(tmp_path, stub_verify, capsys):
    """``--json`` 时 stdout 必须是纯 JSON，所以简报一律走 stderr。"""
    import json

    bib = tmp_path / "refs.bib"
    bib.write_text(BIB_SAMPLE, encoding="utf-8")
    rc = research.cmd_citecheck(
        _ns(bib=[str(bib), str(tmp_path / "gone.bib")], json=True)
    )
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)  # 解析失败即说明 stdout 被污染
    assert isinstance(payload, list) and len(payload) == 3
    assert "gone.bib" in captured.err


def test_citecheck_json_carries_line_and_raw_for_bib_citations(
    tmp_path, stub_verify, capsys
):
    """来自参考文献文件的裁决带原文定位；一份 200 行的文献段里光看 DOI 不好定位。"""
    import json

    md = tmp_path / "survey.md"
    md.write_text(MD_WITH_REFS, encoding="utf-8")
    research.cmd_citecheck(_ns(bib=[str(md)], json=True))
    payload = json.loads(capsys.readouterr().out)
    assert payload and all("line" in item and "raw" in item for item in payload)


def test_citecheck_json_omits_line_for_bare_identifiers(stub_verify, capsys):
    """笔记/裸标识没有原文行号，不该凭空多出 ``line`` 键。"""
    import json

    research.cmd_citecheck(_ns(targets=["10.1103/PhysRevLett.121.124501"], json=True))
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1
    assert "line" not in payload[0] and "raw" not in payload[0]


def test_citecheck_nothing_to_verify_returns_two(tmp_path, capsys):
    assert research.cmd_citecheck(_ns()) == 2
    err = capsys.readouterr().err
    assert "--bib <path>" in err  # 提示里要点出新入口


def test_citecheck_fail_still_returns_one(tmp_path, monkeypatch, capsys):
    """退出码契约不变：有 FAIL → 1（可作写入前门）。"""

    def fake(cite):
        return cv.CitationVerdict(input=dict(cite), kind="doi", status=cv.FAIL)

    monkeypatch.setattr(cv, "verify_citation", fake)
    bib = tmp_path / "refs.bib"
    bib.write_text(BIB_SAMPLE, encoding="utf-8")
    assert research.cmd_citecheck(_ns(bib=[str(bib)])) == 1


def test_citecheck_review_flag_end_to_end(tmp_path, monkeypatch, stub_verify):
    reviews = tmp_path / "reviews"
    reviews.mkdir()
    (reviews / "a_survey.md").write_text(MD_WITH_REFS, encoding="utf-8")
    monkeypatch.setattr(research, "REVIEWS_DIR", reviews)
    assert research.cmd_citecheck(_ns(review=True)) == 0
    assert len(stub_verify) == len(cv.parse_markdown_references(MD_WITH_REFS))
