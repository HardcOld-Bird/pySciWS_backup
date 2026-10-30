"""notes 叶子模块离线测试（无 HTTP、无副作用）。

护栏对象是 WP-B 引入的 frontmatter 统一层，覆盖三件最容易悄悄坏掉的事：

1. **列表字段不再丢失**——旧的手写解析器跳过所有以 ``-`` 开头的行，block-style 的
   ``authors`` / ``topics`` 会全部读不出来，连带 ``cache_manager.referenced_paths()``
   返回空集、``prune --keep-referenced`` 保护失效。
2. **序列化幂等且形状稳定**——同一 dict 永远产生同一字节序列，``index --fix`` 的 diff
   才只包含真实的值变化。
3. **正文逐字节不变**——``split_note`` → ``render_note`` 往返必须字节等价，含正文
   开头的空行（``2023_fang`` 的 ``---`` 与 ``# 标题`` 之间就有一个）。

两份 fixture 直接照抄仓库里真实存在的笔记格式：``BLOCK_STYLE`` 取自
``papers/2018_zhu_*.md``（block-style 列表、无 ``extracted_md_path`` 键），
``FLOW_STYLE`` 取自 ``papers/2023_fang_*.md``（flow-style 列表、含中文路径）。
"""

from __future__ import annotations

import pytest

from pysci.skills.literature_research.tools import notes

# ---------------------------------------------------------------------------
# fixture：两种真实存在的 frontmatter 写法
# ---------------------------------------------------------------------------
BLOCK_STYLE = """---
title: Simultaneous Observation of Topological Edge State and Exceptional Point in an Open and Non-Hermitian System
short_title: Simultaneous Observation of Topological Edge State
authors:
  - Weiwei Zhu
  - Xinsheng Fang
  - Yong Li
first_author_last_name: zhu
year: 2018
publication_date: "2018-03-12T04:08:19Z"
journal: Phys. Rev. Lett. 121, 124501 (2018)
doi: "10.1103/PhysRevLett.121.124501"
local_pdf_path: D:\\XXXIIIGGG\\projects\\pySci\\pySciWS\\data\\cache\\pdfs\\1803.04110.pdf
topics:
  - cond-mat.mes-hall
methods: []
status: unread
review_count: 0
keywords_auto:
  - cond-mat.mes-hall
---
# Simultaneous Observation of Topological Edge State (zhu 2018)

## Changelog

- YYYY-MM-DD: Created (AI auto-fill from OpenAlex/arXiv)
"""

FLOW_STYLE = """---
title: "Extreme Wave Manipulation via Non-Hermitian Metagratings on Degenerated States"
authors: ["Xinsheng Fang", "Nengyin Wang", "Yong Li"]
first_author_last_name: "fang"
year: 2023
journal: "Physical Review Applied"
local_pdf_path: "D:/XiGPrograms/zotero/data/storage/35ZQTHWE/Fang 等 - 2023 - Extreme Wave.pdf"
jif: 4.08
topics: ["non-hermitian", "exceptional-point", "metagrating"]
status: "unread"
---

# Non-Hermitian Metagratings (Fang 2023)

## TLDR

_（待精读后填写）_

## Changelog

- 2026-09-18: Created (AI auto-fill from OpenAlex)
- 2026-09-18: PDF fetched via Zotero; extracted full text (39457 chars)
"""


# ===========================================================================
#  split_note / load_frontmatter —— 列表字段不再丢失
# ===========================================================================
def test_load_frontmatter_block_style_reads_lists():
    """block-style 的 ``authors`` / ``topics`` 必须读成真正的列表（回归旧解析器缺陷）。"""
    fm = notes.load_frontmatter(BLOCK_STYLE)
    assert fm["authors"] == ["Weiwei Zhu", "Xinsheng Fang", "Yong Li"]
    assert fm["topics"] == ["cond-mat.mes-hall"]
    assert fm["methods"] == []
    assert fm["keywords_auto"] == ["cond-mat.mes-hall"]


def test_load_frontmatter_flow_style_reads_lists():
    """flow-style（``["a", "b"]``）同样要读成列表。"""
    fm = notes.load_frontmatter(FLOW_STYLE)
    assert fm["authors"] == ["Xinsheng Fang", "Nengyin Wang", "Yong Li"]
    assert fm["topics"] == ["non-hermitian", "exceptional-point", "metagrating"]


def test_load_frontmatter_scalar_types():
    fm = notes.load_frontmatter(BLOCK_STYLE)
    assert fm["year"] == 2018 and isinstance(fm["year"], int)
    assert fm["review_count"] == 0
    assert fm["status"] == "unread"
    assert fm["doi"] == "10.1103/PhysRevLett.121.124501"
    # 未加引号的 Windows 反斜杠路径按字面量保留
    assert fm["local_pdf_path"].endswith("1803.04110.pdf")
    assert "\\" in fm["local_pdf_path"]


def test_load_frontmatter_missing_key_stays_missing():
    """``2018_zhu`` 那篇确实没有 ``extracted_md_path`` 键——不能凭空造出来。"""
    assert "extracted_md_path" not in notes.load_frontmatter(BLOCK_STYLE)


def test_load_frontmatter_flow_style_numeric_and_unicode_path():
    fm = notes.load_frontmatter(FLOW_STYLE)
    assert fm["jif"] == 4.08
    assert "Fang 等 - 2023" in fm["local_pdf_path"]  # 中文与空格未被破坏


def test_split_note_returns_body_verbatim():
    fm, body = notes.split_note(FLOW_STYLE)
    assert fm["year"] == 2023
    # 正文开头那个空行必须留在 body 里（render_note 只补回被定界符吃掉的那一个换行）
    assert body.startswith("\n# Non-Hermitian Metagratings")


def test_split_note_block_style_body_has_no_leading_blank():
    _, body = notes.split_note(BLOCK_STYLE)
    assert body.startswith("# Simultaneous Observation")


def test_split_note_without_frontmatter():
    fm, body = notes.split_note("just body text")
    assert fm == {}
    assert body == "just body text"


def test_split_note_bad_yaml_degrades_silently():
    """YAML 解析失败时返回 ``({}, 原文)`` 而非抛异常——坏文件不该让 index 崩掉。"""
    broken = "---\ntitle: [unclosed\n---\nbody\n"
    fm, body = notes.split_note(broken)
    assert fm == {}
    assert body == broken


def test_split_note_top_level_list_is_not_a_mapping():
    fm, body = notes.split_note("---\n- a\n- b\n---\nbody\n")
    assert fm == {}
    assert body == "---\n- a\n- b\n---\nbody\n"


# ===========================================================================
#  dump_frontmatter —— 幂等、字段序、缩进形状
# ===========================================================================
def test_dump_is_idempotent():
    fm = notes.load_frontmatter(BLOCK_STYLE)
    once = notes.dump_frontmatter(fm)
    twice = notes.dump_frontmatter(notes.load_frontmatter(f"---\n{once}---\nbody\n"))
    assert once == twice


def test_dump_is_idempotent_flow_style():
    fm = notes.load_frontmatter(FLOW_STYLE)
    once = notes.dump_frontmatter(fm)
    twice = notes.dump_frontmatter(notes.load_frontmatter(f"---\n{once}---\nbody\n"))
    assert once == twice


def test_dump_follows_field_order():
    """已知字段按 FIELD_ORDER 输出，与插入序无关。"""
    out = notes.dump_frontmatter({"status": "read", "title": "T", "year": 2020})
    keys = [ln.split(":")[0] for ln in out.strip().splitlines()]
    assert keys == ["title", "year", "status"]


def test_dump_unknown_keys_appended_in_insertion_order():
    out = notes.dump_frontmatter(
        {"title": "T", "zzz_custom": 1, "aaa_custom": 2, "year": 2020}
    )
    keys = [ln.split(":")[0] for ln in out.strip().splitlines()]
    assert keys == ["title", "year", "zzz_custom", "aaa_custom"]


def test_dump_uses_indented_block_sequences():
    """列表项缩进两格，与既有笔记/模板的形状一致（避免 --fix 制造纯缩进 diff）。"""
    out = notes.dump_frontmatter({"authors": ["A", "B"]})
    assert out == "authors:\n  - A\n  - B\n"


def test_dump_does_not_fold_long_scalars():
    """长标题保持一行，git diff 与 Select-String 才可用。"""
    out = notes.dump_frontmatter(notes.load_frontmatter(BLOCK_STYLE))
    title_line = next(ln for ln in out.splitlines() if ln.startswith("title:"))
    assert title_line.startswith("title: Simultaneous Observation")
    assert title_line.endswith("Non-Hermitian System")


def test_dump_keeps_unicode_literal():
    out = notes.dump_frontmatter(
        {"related_to_my_work_reason": "与增益 EP 项目直接相关"}
    )
    assert "与增益 EP 项目直接相关" in out
    assert "\\u" not in out


def test_dump_quotes_date_like_strings():
    """``2018-03-12`` 必须加引号，否则 safe_load 会把它读成 datetime.date。"""
    out = notes.dump_frontmatter({"publication_date": "2018-03-12"})
    assert out == "publication_date: '2018-03-12'\n"
    assert (
        notes.load_frontmatter(f"---\n{out}---\nb\n")["publication_date"]
        == "2018-03-12"
    )


def test_dump_preserves_false_and_zero():
    out = notes.dump_frontmatter({"esi_hot_paper": False, "review_count": 0})
    assert "esi_hot_paper: false" in out
    assert "review_count: 0" in out


def test_dump_empty_dict():
    assert notes.dump_frontmatter({}) == ""


def test_dump_roundtrip_preserves_values():
    for src in (BLOCK_STYLE, FLOW_STYLE):
        fm = notes.load_frontmatter(src)
        back = notes.load_frontmatter(f"---\n{notes.dump_frontmatter(fm)}---\nx\n")
        assert back == fm


# ===========================================================================
#  render_note —— 正文逐字节不变
# ===========================================================================
@pytest.mark.parametrize("src", [BLOCK_STYLE, FLOW_STYLE])
def test_render_note_roundtrip_preserves_body_and_values(src):
    """不变量 2：正文逐字节保持不变；frontmatter 只被规范化，值不丢。

    注意不能直接断言 ``render_note(fm, body) == src``：frontmatter 是**有意**被重新
    序列化的（引号风格、flow/block 形状都会统一到 :data:`notes.FIELD_ORDER` 与 block
    style，这正是 ``index --fix`` 要做的规范化）。真正的不变量是：

    * body 逐字节相同（含正文开头那个空行）；
    * frontmatter 的**值**相同；
    * 已规范化的文本再渲染一次字节相同（幂等，``--fix`` 重跑不产生新 diff）。
    """
    fm, body = notes.split_note(src)
    out = notes.render_note(fm, body)

    assert out.endswith(body)  # 正文原样拼回，未被规范化
    fm2, body2 = notes.split_note(out)
    assert body2 == body
    assert fm2 == fm
    assert notes.render_note(fm2, body2) == out


def test_render_note_body_is_not_normalized():
    body = "\n# 标题\n\n正文里有一个 --- 分隔线\n\n---\n\n以及尾随空白   \n"
    out = notes.render_note({"title": "T"}, body)
    assert out.endswith(body)


# ===========================================================================
#  note_filename
# ===========================================================================
def test_note_filename_shape():
    """与仓库里真实存在的 ``papers/2018_zhu_simultaneous-observation-of-topological.md`` 一致：
    slug 取 ``short_title``（而非超长全标题），截到 40 字后去掉尾随连字符。"""
    fm = notes.load_frontmatter(BLOCK_STYLE)
    assert notes.note_filename(fm) == (
        "2018_zhu_simultaneous-observation-of-topological.md"
    )


def test_note_filename_prefers_short_title_and_truncates():
    fm = {
        "year": 2020,
        "first_author_last_name": "zhu",
        "title": "A Very Long Full Title That Should Not Be Used",
        "short_title": "Short One",
    }
    assert notes.note_filename(fm) == "2020_zhu_short-one.md"


def test_note_filename_falls_back_to_title_without_short_title():
    fm = {"year": 2020, "first_author_last_name": "zhu", "title": "Only Full Title"}
    assert notes.note_filename(fm) == "2020_zhu_only-full-title.md"


def test_note_filename_missing_year_and_author():
    assert notes.note_filename({}) == "nd_unknown_paper.md"


def test_note_filename_falls_back_to_authors_for_non_latin_names():
    """``normalize_last_name`` 对纯中文名返回 ``''``；文件名不该退化成 ``unknown``。"""
    fm = {
        "year": 2024,
        "first_author_last_name": "",
        "authors": ["朱某某"],
        "title": "声学超表面",
    }
    assert notes.note_filename(fm) == "2024_朱某某_声学超表面.md"


def test_note_filename_falls_back_to_authors_when_field_absent():
    fm = {"year": 2024, "authors": [{"name": "Zheng Zhu"}], "title": "Gain induced EP"}
    assert notes.note_filename(fm) == "2024_zhu_gain-induced-ep.md"


# ===========================================================================
#  merge_frontmatter —— 只填空，绝不覆盖
# ===========================================================================
def test_merge_fills_only_blank_keys():
    existing = {
        "title": "Old",
        "jif": None,
        "doi": "",
        "topics": [],
        "status": "unread",
    }
    incoming = {
        "title": "New",
        "jif": 4.08,
        "doi": "10.1/x",
        "topics": ["ep"],
        "status": "read",
    }
    merged, changed = notes.merge_frontmatter(existing, incoming)
    assert merged["title"] == "Old"  # 非空 → 不动
    assert merged["status"] == "unread"  # 非空 → 不动
    assert merged["jif"] == 4.08
    assert merged["doi"] == "10.1/x"
    assert merged["topics"] == ["ep"]
    assert changed == ["jif", "doi", "topics"]


def test_merge_never_touches_user_filled_evaluation_fields():
    """用户手填的评价字段是 merge 的红线。"""
    existing = {"my_rating": 4, "status": "read", "related_to_my_work": "high"}
    incoming = {"my_rating": 1, "status": "unread", "related_to_my_work": "none"}
    merged, changed = notes.merge_frontmatter(existing, incoming)
    assert merged == existing
    assert changed == []


def test_merge_treats_zero_and_false_as_real_values():
    """``0`` / ``False`` 不是空值，不能被机器值顶掉。"""
    existing = {"review_count": 0, "esi_hot_paper": False}
    incoming = {"review_count": 5, "esi_hot_paper": True}
    merged, changed = notes.merge_frontmatter(existing, incoming)
    assert merged == {"review_count": 0, "esi_hot_paper": False}
    assert changed == []


def test_merge_blank_incoming_values_are_skipped():
    existing = {"title": "T"}
    incoming = {"title": "", "jif": None, "topics": [], "doi": "10.1/x"}
    merged, changed = notes.merge_frontmatter(existing, incoming)
    assert merged == {"title": "T", "doi": "10.1/x"}
    assert changed == ["doi"]


def test_merge_returns_changed_keys_accurately_and_does_not_mutate_inputs():
    existing = {"a": "", "b": 1}
    incoming = {"a": "x", "c": "y"}
    snapshot = dict(existing)
    merged, changed = notes.merge_frontmatter(existing, incoming)
    assert changed == ["a", "c"]
    assert merged == {"a": "x", "b": 1, "c": "y"}
    assert existing == snapshot  # 入参未被就地修改
    assert merged is not existing


def test_merge_nothing_to_do():
    merged, changed = notes.merge_frontmatter({"a": 1}, {"a": 2})
    assert changed == [] and merged == {"a": 1}


def test_merge_handles_none_inputs():
    assert notes.merge_frontmatter(None, None) == ({}, [])
    merged, changed = notes.merge_frontmatter({}, {"a": 1})
    assert merged == {"a": 1} and changed == ["a"]


# ===========================================================================
#  normalize_frontmatter（template_defaults / template_body 的测试见文件末尾）
# ===========================================================================
def test_normalize_adds_missing_template_keys():
    out, added = notes.normalize_frontmatter({"title": "T"})
    assert out["title"] == "T"
    assert out["extracted_md_path"] == ""  # 模板默认值
    assert out["review_count"] == 0
    assert out["status"] == "unread"
    assert "extracted_md_path" in added
    assert set(notes.template_defaults()) <= set(out)


def test_normalize_keeps_existing_values_untouched():
    """只补键，绝不改值——即使既有值与模板默认值冲突。"""
    fm = {
        "status": "read",  # 模板默认 unread
        "review_count": 3,  # 模板默认 0
        "my_rating": 5,  # 模板默认 null
        "esi_highly_cited": True,  # 模板默认 null（未知）
        "related_to_my_work": "high",
    }
    out, added = notes.normalize_frontmatter(fm)
    for k, v in fm.items():
        assert out[k] == v
    assert not (set(added) & set(fm))


def test_normalize_added_keys_are_sorted_by_field_order():
    _, added = notes.normalize_frontmatter({"keywords_auto": [], "title": "T"})
    assert added == sorted(added, key=notes.FIELD_ORDER.index)
    assert "title" not in added and "keywords_auto" not in added


def test_normalize_output_key_order_matches_field_order():
    out, _ = notes.normalize_frontmatter({"keywords_auto": [], "title": "T", "zzz": 1})
    keys = list(out)
    known = [k for k in notes.FIELD_ORDER if k in keys]
    assert keys == known + ["zzz"]  # 未知键追加在后


def test_normalize_is_idempotent():
    """跑第二遍不该产生任何新变更，``index --fix`` 的 diff 才可读。"""
    once, added1 = notes.normalize_frontmatter({"title": "T"})
    twice, added2 = notes.normalize_frontmatter(once)
    assert added1  # 第一遍确实补了东西
    assert twice == once
    assert added2 == []


def test_normalize_with_missing_template_only_orders():
    """模板读不到时静默降级为「只排序」（不变量 1）。"""
    out, added = notes.normalize_frontmatter(
        {"status": "read", "title": "T"}, template="nope.md"
    )
    assert added == []
    assert list(out) == ["title", "status"]


def test_normalize_handles_none_input():
    out, added = notes.normalize_frontmatter(None)
    assert set(notes.template_defaults()) <= set(out)
    assert added


def test_normalize_and_merge_are_not_interchangeable():
    """锁定两者的语义差别：``merge`` 跳过空值，``normalize`` 补的是键的存在性。"""
    incoming = {"jif": None, "extracted_md_path": ""}
    merged, changed = notes.merge_frontmatter({"title": "T"}, incoming)
    assert changed == [] and "jif" not in merged  # merge：空值不写进去
    norm, added = notes.normalize_frontmatter({"title": "T"})
    assert {"jif", "extracted_md_path"} <= set(added)  # normalize：键照样补上


# ===========================================================================
#  append_changelog
# ===========================================================================
def test_append_changelog_appends_at_section_end():
    body = (
        "# T\n\n## TLDR\n\nx\n\n## Changelog\n\n"
        "- 2026-01-01: Created\n- 2026-01-02: Read\n"
    )
    out = notes.append_changelog(body, "- 2026-02-01: 补齐字段 jif")
    assert out.endswith("- 2026-01-02: Read\n- 2026-02-01: 补齐字段 jif\n")
    # 前面的段落逐字节不变
    assert out.startswith(
        "# T\n\n## TLDR\n\nx\n\n## Changelog\n\n- 2026-01-01: Created\n"
    )


def test_append_changelog_stops_at_next_heading():
    """Changelog 不是最后一段时，新行必须插在段内而非文末。"""
    body = "## Changelog\n\n- a\n\n## Appendix\n\nstuff\n"
    out = notes.append_changelog(body, "- b")
    assert out == "## Changelog\n\n- a\n- b\n\n## Appendix\n\nstuff\n"


def test_append_changelog_creates_section_when_missing():
    body = "# T\n\n正文，没有 Changelog 段。\n"
    out = notes.append_changelog(body, "- 2026-02-01: 补齐字段 jif")
    assert out == body + "\n## Changelog\n\n- 2026-02-01: 补齐字段 jif\n"


def test_append_changelog_on_empty_section():
    out = notes.append_changelog("## Changelog\n", "- x")
    assert out == "## Changelog\n\n- x\n"


def test_append_changelog_heading_without_trailing_newline():
    """标题就是文末且无换行时，同样只补一个空行。"""
    assert notes.append_changelog("## Changelog", "- x") == "## Changelog\n\n- x\n"


def test_append_changelog_tolerates_trailing_spaces_on_heading():
    """``## Changelog   ``（尾随空格）仍要能认出标题，且不因 ``\\s*`` 吐掉换行而多插空行。"""
    assert (
        notes.append_changelog("## Changelog   \n", "- x") == "## Changelog   \n\n- x\n"
    )


def test_append_changelog_subsection_is_not_a_boundary():
    """``###`` 属于 Changelog 段内部，新行应附在段末而非插到子标题之前。"""
    body = "## Changelog\n\n- a\n\n### 备注\n\nstuff\n"
    out = notes.append_changelog(body, "- b")
    assert out == "## Changelog\n\n- a\n\n### 备注\n\nstuff\n- b\n"


def test_append_changelog_empty_line_is_noop():
    body = "## Changelog\n\n- a\n"
    assert notes.append_changelog(body, "") == body
    assert notes.append_changelog(body, "\n") == body


def test_append_changelog_on_real_block_style_body():
    _, body = notes.split_note(BLOCK_STYLE)
    out = notes.append_changelog(body, "- 2026-02-01: 补齐字段 extracted_md_path")
    assert out.endswith(
        "- YYYY-MM-DD: Created (AI auto-fill from OpenAlex/arXiv)\n"
        "- 2026-02-01: 补齐字段 extracted_md_path\n"
    )


def test_append_changelog_recognizes_numbered_heading():
    """``## 8. Changelog``（``templates/review_note.md`` 的写法）同样要能认出。

    不认编号的后果不是报错而是**静默重复**：函数会在文末新建一个裸的 ``## Changelog``
    段，而第 8 节永远为空——两份 Changelog 并存，人与机器各写一处，审计轨迹直接失效。
    """
    body = "# T\n\n## 7. 检索式记录\n\nstuff\n\n## 8. Changelog\n\n- a\n"
    assert notes.append_changelog(body, "- b") == (
        "# T\n\n## 7. 检索式记录\n\nstuff\n\n## 8. Changelog\n\n- a\n- b\n"
    )


@pytest.mark.parametrize(
    "head",
    ["## 8. Changelog", "## 8、Changelog", "## 8) Changelog", "## 12.Changelog"],
)
def test_append_changelog_numbered_heading_variants(head):
    """编号分隔符的三种写法与无空格形式都能认出，且**不新建重复段**。"""
    out = notes.append_changelog(f"{head}\n\n- a\n", "- b")
    assert out == f"{head}\n\n- a\n- b\n"
    assert out.count("Changelog") == 1


def test_append_changelog_on_real_review_template_body():
    """对仓库里真实的 ``templates/review_note.md`` 正文追加：只长在第 8 节内部。

    直接用真模板而不是手写夹具，是为了把「模板的 Changelog 带编号」这个事实钉在测试里：
    未来有人把模板改成裸标题或改成 ``## 附录``，本用例会立即失败而不是一直静默重复建段。
    """
    body = notes.template_body("review_note.md")
    assert "## 8. Changelog" in body
    line = "- 2026-02-01: 补充 §4 争议判断"
    out = notes.append_changelog(body, line)
    # 没有新建裸标题的重复段
    assert out.count("## 8. Changelog") == 1
    assert "\n## Changelog" not in out
    assert line in out
    # 第 8 节之前的正文逐字节保留
    cut = body.index("## 8. Changelog")
    assert out.startswith(body[:cut])


# ===========================================================================
#  文本工具
# ===========================================================================
def test_normalize_last_name_transliterates_accents():
    """R4 语义统一：**转写**声调（Büttner → buttner），而非删除（→ bttner）。"""
    assert notes.normalize_last_name("Büttner") == "buttner"
    assert notes.normalize_last_name("Kai Büttner") == "buttner"
    assert notes.normalize_last_name("José García") == "garcia"


def test_normalize_last_name_forms():
    assert notes.normalize_last_name("Zhu, Zheng") == "zhu"
    assert notes.normalize_last_name("Zheng Zhu") == "zhu"
    assert notes.normalize_last_name("Zheng") == "zheng"
    assert notes.normalize_last_name("Ludwig van Beethoven") == "beethoven"
    assert notes.normalize_last_name("Jean-Luc Picard") == "picard"
    assert notes.normalize_last_name("") == ""
    assert notes.normalize_last_name(None) == ""


def test_strip_accents():
    assert notes.strip_accents("Büttner") == "Buttner"
    assert notes.strip_accents("éàç") == "eac"
    assert notes.strip_accents("") == ""


def test_slugify_preserves_chinese():
    assert notes.slugify("声学超表面 EP") == "声学超表面-ep"


def test_slugify_folds_punctuation_and_caps_length():
    assert notes.slugify("Hello,   World!!") == "hello-world"
    assert notes.slugify("a" * 200, max_len=48) == "a" * 48
    assert notes.slugify("---", 10) == "untitled"
    assert notes.slugify("") == "untitled"


# ===========================================================================
#  fmt_size 边界
# ===========================================================================
@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, "0 B"),
        (1, "1 B"),
        (1023, "1023 B"),
        (1024, "1.0 KB"),
        (1536, "1.5 KB"),
        (1024**2, "1.0 MB"),
        (int(1.5 * 1024**2), "1.5 MB"),
        (1024**3, "1.0 GB"),
        (1024**4, "1.0 TB"),
        (5 * 1024**4, "5.0 TB"),  # TB 是上限，不再进位
    ],
)
def test_fmt_size(n, expected):
    assert notes.fmt_size(n) == expected


# ===========================================================================
#  template_defaults / template_body
# ===========================================================================
def test_template_defaults_reads_real_template():
    """默认值直接来自 ``templates/paper_note.md``，不是硬编码副本（防漂移）。"""
    d = notes.template_defaults("paper_note.md")
    # 模板里只有两个**非空**默认值；其余字段一律空，交给数据源与 AI 填。
    assert d.get("status") == "unread"
    assert d.get("review_count") == 0
    assert d.get("my_rating") is None
    assert d.get("authors") == []
    assert d.get("title") == ""
    assert d.get("extracted_md_path") == ""
    assert d.get("journal_ref") == ""
    # WP-D：ESI 默认值必须是 null（未知）而不是 false（已确认不是）
    assert d.get("esi_highly_cited") is None
    assert d.get("esi_hot_paper") is None
    # 模板里的 ``# ===`` 分节横幅与行内语义注释被 yaml 丢弃——这是可接受的：
    # 机器生成的笔记本来就没有 frontmatter 注释，``index --fix`` 不构成信息损失。
    assert not any(str(k).startswith("#") for k in d)


def test_template_defaults_missing_file_is_empty():
    assert notes.template_defaults("no_such_template.md") == {}


def test_template_body_has_placeholders():
    body = notes.template_body("paper_note.md")
    assert "{{short_title}}" in body  # 占位符留给 research._fill_placeholders
    assert "## Changelog" in body
    for head in ("## TLDR", "## Key Claims", "## Journal-tier Justification"):
        assert head in body
    assert body.rstrip().endswith("- YYYY-MM-DD: User review")


def test_template_body_missing_file_is_empty():
    assert notes.template_body("no_such_template.md") == ""


def test_field_order_covers_template_defaults():
    """FIELD_ORDER 必须覆盖模板里的全部字段，否则 --fix 会把它们甩到末尾。"""
    d = notes.template_defaults("paper_note.md")
    if not d:  # 模板缺失时跳过（不在本测试的关注范围）
        pytest.skip("paper_note.md 模板不可用")
    missing = [k for k in d if k not in notes.FIELD_ORDER]
    assert missing == []
