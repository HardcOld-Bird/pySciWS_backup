"""tree_hash 行尾无关回归（backlog 20261010-003127-devops）。

锁定核心修复：同内容、不同行尾（CRLF vs LF）的目录树得**同一**哈希——消除
worktree(CRLF)/main(LF) 合并后 skills-deployed.json 的假漂移；同时内容/文件名
差异仍须改变哈希（规范化不得掩盖真实变更）。
"""

from __future__ import annotations

from pysci.skills.orchestration.tools.sync import tree_hash


def _write(path, text: str, *, crlf: bool) -> None:
    """写文本文件；crlf=True 时把 LF 转 CRLF（模拟 git autocrlf smudge）。"""
    data = text.replace("\n", "\r\n") if crlf else text
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode("utf-8"))


def test_same_content_different_eol_same_hash(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _write(a / "SKILL.md", "# 真本\n正文一行\n", crlf=False)
    _write(a / "references" / "ref.md", "参考\n", crlf=False)
    _write(b / "SKILL.md", "# 真本\n正文一行\n", crlf=True)
    _write(b / "references" / "ref.md", "参考\n", crlf=True)
    assert tree_hash(a) == tree_hash(b)


def test_mixed_eol_within_tree_normalized(tmp_path):
    """main 是混合行尾、worktree 全 CRLF——逐文件交叉后仍须同哈希。"""
    a, b = tmp_path / "a", tmp_path / "b"
    _write(a / "x.md", "line1\nline2\n", crlf=False)  # LF
    _write(a / "y.md", "line3\nline4\n", crlf=True)  # CRLF
    _write(b / "x.md", "line1\nline2\n", crlf=True)  # CRLF
    _write(b / "y.md", "line3\nline4\n", crlf=False)  # LF
    assert tree_hash(a) == tree_hash(b)


def test_crlf_folds_to_lf_at_byte_level(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "x.md").write_bytes(b"a\r\nb\n")  # CRLF + LF
    (b / "x.md").write_bytes(b"a\nb\n")  # 全 LF
    assert tree_hash(a) == tree_hash(b)


def test_content_difference_still_changes_hash(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _write(a / "x.md", "内容A\n", crlf=False)
    _write(b / "x.md", "内容B\n", crlf=True)
    assert tree_hash(a) != tree_hash(b)


def test_filename_difference_changes_hash(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _write(a / "x.md", "同内容\n", crlf=False)
    _write(b / "y.md", "同内容\n", crlf=False)
    assert tree_hash(a) != tree_hash(b)


def test_missing_dir_marker(tmp_path):
    assert tree_hash(tmp_path / "nope") == "MISSING"
