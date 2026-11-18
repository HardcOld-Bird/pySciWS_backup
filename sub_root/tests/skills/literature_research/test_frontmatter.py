"""frontmatter 固定样例断言：平面子集解析/归一化合同与 INDEX.md 生成。"""

from __future__ import annotations

import pytest

from pysci.skills.literature_research import frontmatter

NOTE = (
    "---\n"
    "title: A Paper  \n"
    "authors: Doe et al.\n"
    "year: 2020\n"
    "status: Read\n"
    "tags: [Graph, graph, ML]\n"
    "---\n"
    "\n"
    "## TLDR\n"
    "Body stays.\n"
)


def test_parse_frontmatter_roundtrip_fields():
    meta, body = frontmatter.parse_frontmatter(NOTE)
    assert meta["title"] == "A Paper"
    assert meta["tags"] == ["Graph", "graph", "ML"]
    assert meta["year"] == "2020"
    assert body == "## TLDR\nBody stays."


def test_parse_frontmatter_without_fence_is_identity():
    meta, body = frontmatter.parse_frontmatter("no frontmatter here")
    assert meta == {}
    assert body == "no frontmatter here"


def test_parse_frontmatter_unterminated_fence_is_identity():
    meta, body = frontmatter.parse_frontmatter("---\ntitle: x\nno closing")
    assert meta == {}
    assert body.startswith("---")


def test_normalize_meta_defaults_and_coercions():
    meta, _ = frontmatter.parse_frontmatter(NOTE)
    norm = frontmatter.normalize_meta(meta)
    assert norm["year"] == 2020 and isinstance(norm["year"], int)
    assert norm["tags"] == ["graph", "ml"]  # 小写去重排序
    assert norm["status"] == "read"  # 状态机小写
    assert "venue" not in norm  # 可选字段缺省不补（字段表仅补 status/tags）
    assert list(norm)[:3] == ["title", "authors", "year"]  # 规范键序


def test_normalize_meta_rejects_bad_input():
    with pytest.raises(ValueError, match="缺必填字段"):
        frontmatter.normalize_meta({"title": "x"})
    with pytest.raises(ValueError, match="year 必须是数字"):
        frontmatter.normalize_meta({"title": "x", "authors": "y", "year": "n/a"})
    with pytest.raises(ValueError, match="status 不在"):
        frontmatter.normalize_meta({"title": "x", "authors": "y", "year": "2020", "status": "done"})


def test_normalize_note_keeps_body_and_fences():
    out = frontmatter.normalize_note(NOTE)
    assert out.startswith("---\n")
    assert "## TLDR\nBody stays." in out
    assert "tags: [graph, ml]" in out
    meta, body = frontmatter.parse_frontmatter(out)
    assert frontmatter.normalize_meta(meta)["year"] == 2020
    assert body == "## TLDR\nBody stays."


def test_build_index_sorted_and_invalid_rows(tmp_path):
    (tmp_path / "b2021.md").write_text(
        "---\ntitle: Newer\nauthors: Roe\nyear: 2021\n---\nbody", encoding="utf-8"
    )
    (tmp_path / "a2020.note.md").write_text(NOTE, encoding="utf-8")
    (tmp_path / "bad.md").write_text("---\ntitle: Broken\n---\nbody", encoding="utf-8")
    (tmp_path / "extract.md").write_text("# Raw extract\n\nno frontmatter", encoding="utf-8")
    (tmp_path / "INDEX.md").write_text("stale", encoding="utf-8")
    index = frontmatter.build_index(tmp_path)
    assert "stale" not in index  # INDEX.md 本身被跳过、不自引用
    assert "extract" not in index  # 无 frontmatter 的提取稿不入索引、不算坏笔记
    lines = [ln for ln in index.splitlines() if ln.startswith("|")]
    # 表头分隔行之后：2021 → 2020 → INVALID(bad, year 0)
    assert lines[2].startswith("| b2021 |")
    assert lines[3].startswith("| a2020.note |")
    assert "INVALID" in lines[4]
    assert "A Paper" in lines[3]
