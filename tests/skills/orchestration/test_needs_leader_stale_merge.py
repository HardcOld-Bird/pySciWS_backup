"""NEEDS-LEADER「已提交未合并」的**消费时旁证**（backlog 20261017-111500）。

patrol 在 job 在途期发的「已提交未合并」报警，组长 drain 时 job 往往已自行合并（实测一晚
3 次全部是陈旧报）。消费时复查分支：已并入 main / 分支已删 → 降级为 status 注记（不唤醒）；
确有未合并提交、或查不出 → 照常呈报（兜底防漏唤）。
"""

from __future__ import annotations


def _ev(subject: str) -> dict:
    from pysci.skills.orchestration.tools import events

    return {"event": events.NEEDS_LEADER, "subject": subject, "detail": "d"}


def test_merged_branch_is_stale(monkeypatch):
    from pysci.skills.orchestration.tools import watch

    monkeypatch.setattr(watch, "_branch_still_unmerged", lambda b: False)
    note = watch._suspect_acked(_ev("wt-foo"))
    assert "已合并陈旧报" in note and "wt-foo" in note


def test_unmerged_branch_still_reports(monkeypatch):
    from pysci.skills.orchestration.tools import watch

    monkeypatch.setattr(watch, "_branch_still_unmerged", lambda b: True)
    assert watch._suspect_acked(_ev("wt-foo")) == ""


def test_unreadable_state_keeps_reporting(monkeypatch):
    """查不出（None）→ 空串（宁可报，不静默）。"""
    from pysci.skills.orchestration.tools import watch

    monkeypatch.setattr(watch, "_branch_still_unmerged", lambda b: None)
    assert watch._suspect_acked(_ev("wt-foo")) == ""


def test_other_needs_leader_subjects_are_unchanged(monkeypatch):
    """非 ``wt-*`` 的 NEEDS-LEADER 仍无旁证可查（维持唤醒）。"""
    from pysci.skills.orchestration.tools import watch

    monkeypatch.setattr(watch, "_branch_still_unmerged", lambda b: False)
    assert watch._suspect_acked(_ev("20261017-030200-devops")) == ""


def test_branch_helper_reads_git(tmp_path, monkeypatch):
    """真 git：已并入的分支 → False；有独有提交 → True；分支不存在 → False。"""
    import subprocess

    from pysci.skills.orchestration.tools import watch

    repo = tmp_path / "r"
    repo.mkdir()

    def g(*a):
        return subprocess.run(
            ["git", *a], cwd=repo, capture_output=True, text=True, encoding="utf-8"
        )

    g("init", "-q", "-b", "main")
    g("config", "user.email", "t@t")
    g("config", "user.name", "t")
    (repo / "a.txt").write_text("a", encoding="utf-8")
    g("add", "-A")
    g("commit", "-q", "-m", "init")
    monkeypatch.setattr(watch, "PROJECT_ROOT", repo)
    assert watch._branch_still_unmerged("nope") is False
    assert watch._branch_still_unmerged("main") is False  # 与自身无独有提交
    g("checkout", "-q", "-b", "wt-x")
    (repo / "b.txt").write_text("b", encoding="utf-8")
    g("add", "-A")
    g("commit", "-q", "-m", "work")
    assert watch._branch_still_unmerged("wt-x") is True
