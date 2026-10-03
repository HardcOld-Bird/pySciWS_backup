"""``_merge_note`` 与 ``index --fix`` 的离线回归测试（全部在 tmp_path 下，不碰真实 papers/）。

护栏对象是 WP-C 修掉的 P0 缺陷：旧 ``_write_note`` 是全有或全无——文件已存在且非
``--overwrite`` 时**直接 return、什么都不写**。后果链是：``cmd_read`` 算出的
``local_pdf_path`` / ``extracted_md_path`` 在标准工作流（先 ``add`` 建骨架、后 ``read``
抓全文）下永远写不进笔记；连带 ``cache_manager.referenced_paths()`` 取不到这两个字段，
``cache prune --keep-referenced`` 的保护对现存全部笔记完全失效。

锁定的四条不变量：

1. ``merged`` 时**只填空键**——用户手填的 ``my_rating`` / ``status`` /
   ``related_to_my_work`` 永不被机器值顶掉；
2. ``merged`` / ``--fix`` 时**正文逐字节保留**（``merged`` 只允许在 ``## Changelog``
   段末追加一行，``--fix`` 连那一行都不加）；
3. ``unchanged`` 时**完全不触碰文件**（保留 mtime，免得无谓刷新 prune 的 LRU 依据）；
4. 无法解析 frontmatter 的既有文件（人工笔记）**绝不被自动改写**。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytest

from pysci.skills.literature_research.tools import notes, research

# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------
BASE_FM: dict = {
    "title": "Topological Edge State and Exceptional Point in an Open System",
    "short_title": "Topological Edge State",
    "authors": ["Weiwei Zhu", "Yong Li"],
    "first_author_last_name": "zhu",
    "year": 2018,
    "journal": "Phys. Rev. Lett.",
    "doi": "10.1103/PhysRevLett.121.124501",
    "status": "unread",
}
BASE_NAME = "2018_zhu_topological-edge-state.md"

BODY = (
    "# Topological Edge State (zhu 2018)\n"
    "\n"
    "## TLDR\n"
    "\n"
    "_（待精读后填写）_\n"
    "\n"
    "## Changelog\n"
    "\n"
    "- 2026-09-21: Created (AI auto-fill from arXiv)\n"
)

#: flow-style 笔记（照抄 ``papers/2023_fang_*.md`` 的写法）：旧解析器读不出它的列表。
FLOW_NOTE = (
    "---\n"
    'title: "Extreme Wave Manipulation via Non-Hermitian Metagratings"\n'
    'authors: ["Xinsheng Fang", "Yong Li"]\n'
    'first_author_last_name: "fang"\n'
    "year: 2023\n"
    'local_pdf_path: "D:/XiGPrograms/zotero/data/storage/35ZQTHWE/Fang 等 - 2023.pdf"\n'
    'status: "unread"\n'
    "---\n"
    "\n"
    "# Non-Hermitian Metagratings (Fang 2023)\n"
    "\n"
    "## Changelog\n"
    "\n"
    "- 2026-09-18: Created\n"
)


def _seed_note(
    papers_dir: Path, fm: dict, body: str = BODY, *, newline: str = "\r\n"
) -> Path:
    """按 :func:`notes.note_filename` 的命名规范落一篇既有笔记（默认 CRLF，同真实数据）。"""
    path = papers_dir / notes.note_filename(fm)
    path.write_text(notes.render_note(fm, body), encoding="utf-8", newline=newline)
    return path


def _write_raw_note(papers_dir: Path, name: str, text: str) -> Path:
    """原文落盘（不重渲染），用于 flow-style 之类的形状夹具。"""
    path = papers_dir / name
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _index_args(**kw) -> argparse.Namespace:
    base = {"check": False, "fix": False, "dry_run": False, "force": False}
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.fixture
def papers_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把 ``research.PAPERS_DIR`` 指到 tmp_path，测试绝不触碰真实 papers/。"""
    d = tmp_path / "papers"
    d.mkdir()
    monkeypatch.setattr(research, "PAPERS_DIR", d)
    return d


@pytest.fixture
def index_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """同时重定向 ``PAPERS_DIR`` 与 ``INDEX_PATH``（``cmd_index`` 两个都要）。"""
    d = tmp_path / "papers"
    d.mkdir()
    idx = tmp_path / "INDEX.md"
    monkeypatch.setattr(research, "PAPERS_DIR", d)
    monkeypatch.setattr(research, "INDEX_PATH", idx)
    return d, idx


# ===========================================================================
#  _merge_note —— 四种 action
# ===========================================================================
def test_merge_note_creates_when_absent(papers_dir):
    path, action, changed = research._merge_note(dict(BASE_FM))
    assert action == research.NOTE_CREATED
    assert changed == []
    assert path == papers_dir / BASE_NAME
    fm, body = notes.split_note(path.read_text(encoding="utf-8"))
    assert fm["doi"] == BASE_FM["doi"]
    assert "added_date" in fm  # build_note_markdown 补的
    # 正文来自真实模板骨架（templates/paper_note.md 的 12 个 ## 段）
    assert "## TLDR" in body and "## Changelog" in body
    assert "## Journal-tier Justification" in body


def test_merge_note_fills_blank_fields(papers_dir):
    """P0 回归：``cmd_read`` 算出的两个路径字段必须真能写进既有笔记。"""
    path = _seed_note(papers_dir, dict(BASE_FM))
    incoming = dict(BASE_FM)
    incoming["extracted_md_path"] = "data/cache/extracted/topological_fulltext.md"
    incoming["local_pdf_path"] = "data/cache/pdfs/1803.04110.pdf"

    p2, action, changed = research._merge_note(incoming)

    assert p2 == path
    assert action == research.NOTE_MERGED
    assert set(changed) == {"extracted_md_path", "local_pdf_path"}
    fm = notes.load_frontmatter(p2.read_text(encoding="utf-8"))
    assert fm["extracted_md_path"] == incoming["extracted_md_path"]
    assert fm["local_pdf_path"] == incoming["local_pdf_path"]


def test_merge_note_preserves_body_byte_for_byte(papers_dir):
    """不变量 2：正文除 Changelog 末尾多出的那一行外，逐字节不变。"""
    path = _seed_note(papers_dir, dict(BASE_FM))
    incoming = dict(BASE_FM)
    incoming["jif"] = 8.97

    research._merge_note(incoming)

    _, body = notes.split_note(path.read_text(encoding="utf-8"))
    assert body.split("## Changelog")[0] == BODY.split("## Changelog")[0]
    old_lines = [ln for ln in BODY.splitlines() if ln.startswith("- ")]
    new_lines = [ln for ln in body.splitlines() if ln.startswith("- ")]
    assert len(new_lines) == len(old_lines) + 1  # 只追加，不重写
    assert new_lines[: len(old_lines)] == old_lines
    assert "补齐字段 jif" in new_lines[-1]
    assert research._today() in new_lines[-1]


def test_merge_note_unchanged_does_not_touch_file(papers_dir):
    """不变量 3：无空键可补时文件内容与 mtime 都不动。"""
    fm = dict(BASE_FM)
    fm["jif"] = 8.97
    path = _seed_note(papers_dir, fm)
    before = path.read_bytes()
    old = 1_000_000.0
    os.utime(path, (old, old))
    mtime_before = path.stat().st_mtime

    p2, action, changed = research._merge_note(dict(fm))

    assert action == research.NOTE_UNCHANGED
    assert changed == []
    assert p2 == path
    assert path.read_bytes() == before
    assert path.stat().st_mtime == mtime_before


def test_merge_note_skips_blank_incoming_values(papers_dir):
    """incoming 的空值不该被写进去（那只是把缺失键变成显式的 null 噪声）。"""
    path = _seed_note(papers_dir, dict(BASE_FM))
    incoming = dict(BASE_FM)
    incoming["jif"] = None
    incoming["wos_id"] = ""
    incoming["topics"] = []

    _, action, changed = research._merge_note(incoming)

    assert action == research.NOTE_UNCHANGED
    assert changed == []
    fm = notes.load_frontmatter(path.read_text(encoding="utf-8"))
    assert "jif" not in fm and "wos_id" not in fm and "topics" not in fm


def test_merge_note_overwrite_resets_body(papers_dir):
    path = _seed_note(papers_dir, dict(BASE_FM))
    p2, action, _ = research._merge_note(dict(BASE_FM), overwrite=True)
    assert action == research.NOTE_OVERWRITTEN
    assert p2 == path
    _, body = notes.split_note(p2.read_text(encoding="utf-8"))
    assert "## TLDR" in body  # 正文被模板重置
    assert "- 2026-09-21: Created (AI auto-fill from arXiv)" not in body


def test_merge_note_does_not_clobber_user_fields(papers_dir):
    """不变量 1：用户手填的判断字段永不被机器值顶掉。"""
    fm = dict(BASE_FM)
    fm.update(
        my_rating=5,
        status="read",
        related_to_my_work="high",
        related_to_my_work_reason="与增益 EP 项目直接相关",
    )
    path = _seed_note(papers_dir, fm)
    incoming = dict(BASE_FM)
    incoming.update(my_rating=1, status="unread", related_to_my_work="none", jif=8.97)

    _, action, changed = research._merge_note(incoming)

    assert action == research.NOTE_MERGED
    assert changed == ["jif"]
    got = notes.load_frontmatter(path.read_text(encoding="utf-8"))
    assert got["my_rating"] == 5
    assert got["status"] == "read"
    assert got["related_to_my_work"] == "high"
    assert got["related_to_my_work_reason"] == "与增益 EP 项目直接相关"
    assert got["jif"] == 8.97


def test_merge_note_merges_flow_style_note(papers_dir):
    """``2023_fang`` 那类 flow-style 笔记同样能被 merge（旧解析器读不出它的列表）。"""
    fm = notes.load_frontmatter(FLOW_NOTE)
    # 文件名必须合命名规范，否则 _merge_note 会另建一篇而不是合并到它
    path = _write_raw_note(papers_dir, notes.note_filename(fm), FLOW_NOTE)

    incoming = dict(fm)
    incoming["extracted_md_path"] = "cache/extracted/fang_fulltext.md"
    p2, action, changed = research._merge_note(incoming)

    assert p2 == path
    assert action == research.NOTE_MERGED and changed == ["extracted_md_path"]
    got = notes.load_frontmatter(p2.read_text(encoding="utf-8"))
    assert got["authors"] == ["Xinsheng Fang", "Yong Li"]
    assert got["extracted_md_path"] == "cache/extracted/fang_fulltext.md"
    # 人工手补的 Zotero 存储路径不得被覆盖
    assert got["local_pdf_path"].endswith("Fang 等 - 2023.pdf")


def test_merge_note_skips_file_without_frontmatter(papers_dir, capsys):
    """不变量 4：人工笔记（无 frontmatter）绝不被自动改写。"""
    path = papers_dir / BASE_NAME
    path.write_text("# 我的手写笔记\n\n正文。\n", encoding="utf-8", newline="\n")
    before = path.read_bytes()

    p2, action, changed = research._merge_note(dict(BASE_FM))

    assert p2 == path
    assert action == research.NOTE_UNCHANGED and changed == []
    assert path.read_bytes() == before
    assert "无可解析的 frontmatter" in capsys.readouterr().err


@pytest.mark.parametrize("newline", ["\r\n", "\n"])
def test_merge_note_preserves_newline_style(papers_dir, newline):
    """LF-only 的笔记不能被 merge 顺手翻译成 CRLF——那是「正文逐字节不变」的磁盘含义。"""
    path = _seed_note(papers_dir, dict(BASE_FM), newline=newline)
    incoming = dict(BASE_FM)
    incoming["jif"] = 8.97

    research._merge_note(incoming)

    raw = path.read_bytes().decode("utf-8")
    assert ("\r\n" in raw) is (newline == "\r\n")


# ===========================================================================
#  _report_note_action —— 四种 action 必须可分辨
# ===========================================================================
@pytest.mark.parametrize(
    ("action", "expect"),
    [
        (research.NOTE_CREATED, "新建"),
        (research.NOTE_OVERWRITTEN, "--overwrite 重建"),
        (research.NOTE_MERGED, "补齐 1 个空字段"),
        (research.NOTE_UNCHANGED, "笔记未变"),
    ],
)
def test_report_note_action_distinguishes_all_four(tmp_path, capsys, action, expect):
    """旧措辞把 merged 与 unchanged 压成同一句「已存在，未覆盖」，LLM 会误判成写入失败。"""
    research._report_note_action("read", tmp_path / BASE_NAME, action, ["jif"])
    out = capsys.readouterr().out
    assert expect in out
    assert "未覆盖" not in out


def test_report_note_action_merged_lists_field_names(tmp_path, capsys):
    research._report_note_action(
        "add",
        tmp_path / BASE_NAME,
        research.NOTE_MERGED,
        ["extracted_md_path", "local_pdf_path"],
    )
    out = capsys.readouterr().out
    assert "extracted_md_path" in out and "local_pdf_path" in out
    assert "Changelog" in out


# ===========================================================================
#  index --check —— 构建日期不参与比对
# ===========================================================================
def test_index_comparable_ignores_build_date():
    """回归：--check 曾把渲染时刻的日期烘进比对内容，隔天必报红。"""
    a = "# Literature Index\n\n> 由 `research index` 自动重建于 2026-09-21；共 3 篇。\n"
    b = "# Literature Index\n\n> 由 `research index` 自动重建于 2026-10-03；共 3 篇。\n"
    assert research._index_comparable(a) == research._index_comparable(b)


def test_index_comparable_still_sees_real_change():
    a = "> 由 `research index` 自动重建于 2026-09-21；共 3 篇。\n| 1 | x |\n"
    b = "> 由 `research index` 自动重建于 2026-09-21；共 4 篇。\n| 1 | x |\n"
    assert research._index_comparable(a) != research._index_comparable(b)


def test_cmd_index_check_passes_on_a_different_day(index_dirs, capsys):
    papers, idx = index_dirs
    _seed_note(papers, dict(BASE_FM))
    assert research.cmd_index(_index_args()) == 0
    # 把 INDEX.md 的构建日期改成过去的日子，模拟「上次构建是几天前」
    text = idx.read_text(encoding="utf-8")
    idx.write_text(text.replace(research._today(), "2001-01-01"), encoding="utf-8")

    assert research.cmd_index(_index_args(check=True)) == 0
    assert "已是最新" in capsys.readouterr().out


def test_cmd_index_check_fails_on_real_change(index_dirs, capsys):
    papers, idx = index_dirs
    _seed_note(papers, dict(BASE_FM))
    assert research.cmd_index(_index_args()) == 0
    # 新增一篇 → papers/ 的实际内容变了 → 必须报红
    other = dict(BASE_FM)
    other.update(
        year=2021, first_author_last_name="gu", short_title="Exceptional Point"
    )
    _seed_note(papers, other)

    assert research.cmd_index(_index_args(check=True)) == 1
    assert "不一致" in capsys.readouterr().out


# ===========================================================================
#  index --fix —— 存量回填
# ===========================================================================
def test_cmd_index_fix_backfills_missing_keys(index_dirs, capsys):
    """三篇现存笔记都缺 ``extracted_md_path`` 键；--fix 应按模板补齐。"""
    papers, _ = index_dirs
    path = _seed_note(papers, dict(BASE_FM))
    before_fm, before_body = notes.split_note(path.read_text(encoding="utf-8"))
    assert "extracted_md_path" not in before_fm

    assert research.cmd_index(_index_args(fix=True)) == 0

    fm, body = notes.split_note(path.read_text(encoding="utf-8"))
    assert body == before_body  # 不变量 2：--fix 连 Changelog 都不动
    assert "extracted_md_path" in fm
    assert fm["extracted_md_path"] == ""  # 模板默认值
    assert fm["doi"] == before_fm["doi"]  # 既有值一律不动
    assert fm["status"] == "unread"
    assert fm["review_count"] == 0
    out = capsys.readouterr().out
    assert "已补齐" in out and "extracted_md_path" in out


def test_cmd_index_fix_orders_fields_by_field_order(index_dirs):
    papers, _ = index_dirs
    fm = dict(BASE_FM)
    fm["zzz_custom"] = 1
    path = _seed_note(papers, fm)

    assert research.cmd_index(_index_args(fix=True)) == 0

    keys = list(notes.load_frontmatter(path.read_text(encoding="utf-8")))
    known = [k for k in notes.FIELD_ORDER if k in keys]
    unknown = [k for k in keys if k not in notes.FIELD_ORDER]
    assert keys == known + unknown
    assert keys[: len(known)] == known  # 已知字段严格按 FIELD_ORDER
    assert unknown == ["zzz_custom"]


def test_cmd_index_fix_normalizes_flow_style_to_block(index_dirs):
    papers, _ = index_dirs
    fm = notes.load_frontmatter(FLOW_NOTE)
    path = _write_raw_note(papers, notes.note_filename(fm), FLOW_NOTE)

    assert research.cmd_index(_index_args(fix=True)) == 0

    text = path.read_text(encoding="utf-8")
    assert "authors:\n  - Xinsheng Fang\n  - Yong Li\n" in text
    assert 'authors: ["' not in text
    got = notes.load_frontmatter(text)
    assert got["authors"] == ["Xinsheng Fang", "Yong Li"]  # 值没丢


def test_cmd_index_fix_dry_run_writes_nothing(index_dirs, capsys):
    papers, idx = index_dirs
    path = _seed_note(papers, dict(BASE_FM))
    before = path.read_bytes()

    assert research.cmd_index(_index_args(fix=True, dry_run=True)) == 0

    assert path.read_bytes() == before
    assert not idx.exists()  # dry-run 连 INDEX.md 也不写
    out = capsys.readouterr().out
    assert "将补齐" in out and "--dry-run" in out


def test_cmd_index_fix_is_idempotent(index_dirs):
    """跑第二遍不该产生任何新变更（--fix 的 diff 才可读）。"""
    papers, _ = index_dirs
    path = _seed_note(papers, dict(BASE_FM))
    assert research.cmd_index(_index_args(fix=True)) == 0
    once = path.read_bytes()
    assert research.cmd_index(_index_args(fix=True)) == 0
    assert path.read_bytes() == once


def test_cmd_index_fix_reports_but_does_not_rename(index_dirs, capsys):
    """改名会破坏 reviews/ 的 [[wiki-link]]，所以只报告。"""
    papers, _ = index_dirs
    path = papers / "wrong_name.md"
    path.write_text(
        notes.render_note(dict(BASE_FM), BODY), encoding="utf-8", newline="\n"
    )

    assert research.cmd_index(_index_args(fix=True)) == 0

    assert path.exists()  # 原名保留
    assert not (papers / BASE_NAME).exists()  # 没有被自动改名
    assert "不自动重命名" in capsys.readouterr().err


def test_cmd_index_fix_skips_note_without_frontmatter(index_dirs, capsys):
    papers, _ = index_dirs
    path = papers / "handwritten.md"
    path.write_text("# 我的手写笔记\n\n正文。\n", encoding="utf-8", newline="\n")
    before = path.read_bytes()

    assert research.cmd_index(_index_args(fix=True)) == 0

    assert path.read_bytes() == before
    captured = capsys.readouterr()
    assert "无可解析的 frontmatter" in captured.err


def test_cmd_index_dry_run_without_fix_is_ignored(index_dirs, capsys):
    """``--dry-run`` 单独给出时只提醒一句，然后照常重建 INDEX.md（不做半吊子预演）。"""
    papers, idx = index_dirs
    _seed_note(papers, dict(BASE_FM))

    assert research.cmd_index(_index_args(dry_run=True)) == 0

    assert idx.exists()
    assert "需与 --fix 同用" in capsys.readouterr().err
