"""drain 销账的 commit 定位三级回退（backlog 20261011-220621-leader 要求 4）。

首战台账实锤：E0-B 条目 ``commit=""``，而合并 commit f2e0840 一直在 ``git log`` 里——
正文措辞 ``合并 commit hash：f2e0840`` 里 "hash" 的 a 是十六进制字母，正则的间隔类跨不过
去，整体失配。修复：正文抓不到就按 ``wt-<backlog-id>`` 分支回溯 merge commit（worktree
名=条目 id 是任务书纪律），彻底查不到才写 ``unresolved`` 注记——**commit 字段不留空串**。
"""

from __future__ import annotations

import subprocess

import pytest

from pysci.skills.orchestration.tools import dispatch, drain, workflow
from pysci.skills.orchestration.tools.dispatch import DispatchOutcome

from .conftest import drain_batch, read_backlog, seed_backlog

# 交付正文的两条真实措辞：正则抓得到的 / 抓不到的（E0-B 原形状）
BODY_PARSEABLE = "已完成，合并 commit 3f2a1b9，backlog id=s1"
BODY_MISSED = "改动清单略；合并 commit hash：f2e0840；backlog id=s1"


# ---------------------------------------------------------------------------
# resolve_commit：正文 → 分支回溯 → 注记
# ---------------------------------------------------------------------------
def test_extract_commit_misses_hash_wording():
    """钉住缺陷形状本身：正文写了 hash，正则却抓不到（所以才需要回退）。"""
    assert drain._extract_commit(BODY_PARSEABLE) == "3f2a1b9"
    assert drain._extract_commit(BODY_MISSED) == ""


def test_resolve_commit_uses_body_hash_without_touching_git(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(
        drain, "_run_git", lambda *a: calls.append(a) or (0, "deadbeef")
    )
    assert drain.resolve_commit(BODY_PARSEABLE, "s1") == "3f2a1b9"
    assert calls == [], "正文已抓到 → 不必查 git"


def test_resolve_commit_falls_back_to_branch_merge_commit(monkeypatch):
    calls: list[tuple] = []

    def fake_git(*args: str) -> tuple[int, str]:
        calls.append(args)
        return 0, "f2e0840"

    monkeypatch.setattr(drain, "_run_git", fake_git)
    got = drain.resolve_commit(BODY_MISSED, "20261011-210317-leader")

    assert got == "f2e0840", "正则失配 → 按 wt-<id> 分支回溯 merge commit"
    assert calls == [
        (
            "log",
            "--merges",
            "-n",
            "1",
            "--format=%h",
            "--grep=wt-20261011-210317-leader",
        )
    ]


def test_resolve_commit_unresolved_when_both_paths_miss(monkeypatch):
    monkeypatch.setattr(drain, "_run_git", lambda *a: (0, ""))
    assert drain.resolve_commit("没有给出哈希", "s9") == drain.COMMIT_UNRESOLVED


def test_resolve_commit_skips_git_without_worktree_name(monkeypatch):
    monkeypatch.setattr(
        drain, "_run_git", lambda *a: pytest.fail(f"无名不该查 git：{a}")
    )
    assert drain.resolve_commit("正文无哈希", "") == drain.COMMIT_UNRESOLVED


def test_merge_commit_of_ignores_non_hash_output(monkeypatch):
    """git 报错/输出脏行（如空仓库的 stderr 混入）→ 返回空串交给下一级注记。"""
    monkeypatch.setattr(
        drain, "_run_git", lambda *a: (128, "fatal: not a git repository")
    )
    assert drain.merge_commit_of("s1") == ""
    monkeypatch.setattr(drain, "_run_git", lambda *a: (0, "用法: git log [<options>]"))
    assert drain.merge_commit_of("s1") == ""


# ---------------------------------------------------------------------------
# _drain_batch：.done 的 items 与 backlog 注记都带上定位到的 commit
# ---------------------------------------------------------------------------
def test_drain_batch_sets_per_id_commit_from_branch_lookup(iso_state, monkeypatch):
    seed_backlog(
        iso_state,
        [
            {"id": "s1", "status": "pending", "member": "devops"},
            {"id": "s2", "status": "pending", "member": "devops"},
        ],
    )
    seed, group = workflow.backlog_take_batch(4)
    monkeypatch.setattr(
        dispatch,
        "do_dispatch",
        lambda *a, **k: DispatchOutcome(code=0, kind="result", body=BODY_MISSED),
    )
    asked: list[str] = []

    def fake_git(*args: str) -> tuple[int, str]:
        name = str(args[-1]).removeprefix("--grep=wt-")
        asked.append(name)
        return 0, f"cafe00{name[-1]}"

    monkeypatch.setattr(drain, "_run_git", fake_git)

    items = drain_batch(seed, group, drain.new_run_log(), dry=False)

    assert asked == ["s1"], "只查本批实际完成项（s2 未被选中 → 不查）"
    assert [i["commit"] for i in items] == ["cafe001"]
    assert all(i["commit"] for i in items), "commit 字段不留空串"
    by_id = {i["id"]: i for i in read_backlog(iso_state)}
    assert by_id["s1"]["status"] == "done" and "cafe001" in by_id["s1"]["note"]
    assert by_id["s2"]["status"] == "pending"


def test_drain_batch_unresolved_annotation_not_empty(iso_state, monkeypatch):
    """两条路都失败（正文无哈希 + 分支查无 merge）→ 注记 unresolved，仍非空串。"""
    seed_backlog(iso_state, [{"id": "s1", "status": "pending", "member": "devops"}])
    seed, group = workflow.backlog_take_batch(4)
    monkeypatch.setattr(
        dispatch,
        "do_dispatch",
        lambda *a, **k: DispatchOutcome(code=0, kind="result", body="正文什么都没写"),
    )
    monkeypatch.setattr(drain, "_run_git", lambda *a: (0, ""))

    items = drain_batch(seed, group, drain.new_run_log(), dry=False)

    assert items[0]["commit"] == drain.COMMIT_UNRESOLVED
    assert drain.COMMIT_UNRESOLVED in read_backlog(iso_state)[0]["note"]


# ---------------------------------------------------------------------------
# 真 git 面：--grep 形状必须匹配 git merge --no-ff 的默认消息（不靠 mock 自证）
# ---------------------------------------------------------------------------
@pytest.fixture
def real_repo(tmp_path):
    """最小真仓库：main 上 ``git merge --no-ff wt-item-1``（默认 merge 消息）。"""

    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )

    run("init", "-b", "main", ".")
    run("config", "user.email", "test@example.com")
    run("config", "user.name", "test")
    (tmp_path / "base.txt").write_text("base\n", encoding="utf-8")
    run("add", "base.txt")
    run("commit", "-m", "feat: base")
    run("checkout", "-b", "wt-item-1")
    (tmp_path / "work.txt").write_text("work\n", encoding="utf-8")
    run("add", "work.txt")
    run("commit", "-m", "feat: 工作")
    run("checkout", "main")
    run("merge", "--no-ff", "wt-item-1")
    return tmp_path, run("log", "--merges", "-1", "--format=%h").stdout.strip()


def test_merge_commit_of_matches_real_no_ff_message(real_repo, monkeypatch):
    repo, expected = real_repo
    monkeypatch.setattr(drain, "PROJECT_ROOT", repo)
    assert drain.merge_commit_of("item-1") == expected


def test_merge_commit_of_empty_for_unknown_name(real_repo, monkeypatch):
    repo, _expected = real_repo
    monkeypatch.setattr(drain, "PROJECT_ROOT", repo)
    assert drain.merge_commit_of("nope") == ""
