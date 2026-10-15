"""「已提交未合并」检出与告警（backlog 20261015-145045 f4a）。

实案（2026-10-15 两起，用户征询后裁定必修）：devops 尾段「提交→合并→推送」被进程杀死/
跳截断/重启时停在**已提交未合并**——成果卡在 worktree 里静默躺平（6304 卡 3 小时），零告警。

判据**复用** ``worktree_salvage_state`` 的 ``unmerged``（``git rev-list --count main..branch``），
不新造第二份；**只检出出声，不自动合并**（组长裁决：patrol 自动 merge 进 main 是护栏级写操作）。
"""

from __future__ import annotations

from pysci.skills.orchestration.tools import drain, events, workflow


def _fake_git(monkeypatch, *, unmerged: str, worktrees: list[str], head="abc123456789"):
    """假 git：只回答本判据要用的那几条命令（面清单 / 分支存在 / head / 领先数 / 分叉点 / 脏检）。"""
    calls: list[tuple[str, ...]] = []

    def _run(*args: str):
        calls.append(tuple(args))
        if args[:3] == ("worktree", "list", "--porcelain"):
            return 0, "\n\n".join(f"worktree {p}" for p in worktrees) + "\n"
        if args[:1] == ("rev-parse",) and "--verify" in args:
            return 0, ""
        if args[:2] == ("rev-parse", "--short"):
            return 0, head
        if args[:2] == ("rev-list", "--count"):
            return 0, unmerged
        if args[:1] == ("merge-base",):
            return 0, "deadbeef1234"
        if args[:1] == ("-C",):
            return 0, ""  # 面内干净
        return 1, ""

    monkeypatch.setattr(drain, "_run_git", _run)
    return calls


def _face(name: str) -> str:
    return str(workflow.WORKTREES_ROOT / name)


def test_unmerged_face_is_detected_with_readings(iso_state, monkeypatch):
    """分支领先 main → 检出并给出读数（分支/提交数/落后点/现成命令）。"""
    _fake_git(monkeypatch, unmerged="3", worktrees=[_face("face-a")])

    alerts = drain.unmerged_worktree_alerts()

    assert len(alerts) == 1
    a = alerts[0]
    assert a["worktree"] == "face-a" and a["branch"] == "wt-face-a"
    assert a["unmerged"] == 3
    assert a["head"] == "abc123456789"
    assert a["behind_point"] == "deadbeef1234", "落后点＝与 main 的分叉点"
    assert a["merge_cmd"] == "pysci-dev worktree merge face-a", "须附现成命令供一键执行"
    assert "3 个提交未合并" in a["title"]


def test_merged_face_is_silent(iso_state, monkeypatch):
    """已合并（领先 0）→ 不误报。"""
    _fake_git(monkeypatch, unmerged="0", worktrees=[_face("face-a")])
    assert drain.unmerged_worktree_alerts() == []


def test_no_worktree_is_silent(iso_state, monkeypatch):
    _fake_git(monkeypatch, unmerged="3", worktrees=[])
    assert drain.unmerged_worktree_alerts() == []


def test_non_orchestration_worktrees_are_ignored(iso_state, monkeypatch):
    """仓外/主树的 worktree 不算本编排的面（面根之外一律忽略）。"""
    _fake_git(monkeypatch, unmerged="3", worktrees=[str(iso_state / "elsewhere")])
    assert drain.unmerged_worktree_alerts() == []


def test_announce_emits_once_per_branch_head(iso_state, monkeypatch):
    """出声：同面同 head 只喊一次；head 变了（又有新提交）再喊。"""
    _fake_git(monkeypatch, unmerged="2", worktrees=[_face("face-b")])
    sent: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        events,
        "emit",
        lambda kind, **kw: sent.append(
            (kind, kw.get("subject", ""), kw.get("action", ""))
        ),
    )
    seen: set = set()

    first = drain.announce_unmerged_worktrees(seen)
    second = drain.announce_unmerged_worktrees(seen)

    assert len(first) == 1 and second == [], "同一 (branch, head) 不得每分钟重复喊"
    assert len(sent) == 1
    kind, subject, action = sent[0]
    assert kind == events.NEEDS_LEADER and subject == "wt-face-b"
    assert action == "pysci-dev worktree merge face-b", "出声须点名现成合并命令"

    _fake_git(
        monkeypatch, unmerged="5", worktrees=[_face("face-b")], head="cafe00001111"
    )
    third = drain.announce_unmerged_worktrees(seen)
    assert len(third) == 1 and len(sent) == 2, "head 变了＝又有新提交，应再喊一次"


def test_watchdog_tick_calls_the_alert(iso_state, monkeypatch):
    """接线：检出挂在既有巡检 tick（不新造定时器）——改回不挂即红。"""
    from pathlib import Path

    from pysci.skills.orchestration.tools import watchdog

    src = Path(watchdog.__file__).read_text(encoding="utf-8")
    assert "_drain.announce_unmerged_worktrees(unmerged_seen)" in src
    assert "unmerged_seen" in src
