"""NEEDS-LEADER「已提交未合并」的消费时新鲜度重核（adhoc-20261023-083549）。

实证（drain 排空 seq≈317 两条 wt-bilingual-pipeline-*）：报警按发射时刻快照出声，成员常在
交付前已自行合并（分支删除），事件因聋期迟到唤醒/事后手动 drain 才被消费——连续两次虚惊。
修复：消费呈现/唤醒前重核——**分支存在且 main..branch 仍有未合并提交才算 live**；已合并/
分支已删 → 自动销账 + 留痕一行，不作待处理呈现、不触发组长唤醒。两条消费通道共用单一判据
``watch.unmerged_stale_note``：自动唤醒（queue_pressure → _suspect_acked）与手动
``orch events drain``。**绝不自动合并**（护栏级语义）；真实未合并照常呈现+唤醒。
"""

from __future__ import annotations

import time
from argparse import Namespace

from pysci.skills.orchestration.tools import drain, events, orch, watch, workflow


def _ev(subject: str, *, kind: str = events.NEEDS_LEADER) -> dict:
    return {"event": kind, "subject": subject, "detail": "d"}


def _emit(subject: str, detail: str = "「已提交未合并」示例") -> None:
    events.emit(events.NEEDS_LEADER, subject=subject, detail=detail, action="a")


# ---------------------------------------------------------------------------
# 判据（单一实现 unmerged_stale_note）
# ---------------------------------------------------------------------------
def test_stale_branch_is_marked_autocancel(monkeypatch):
    monkeypatch.setattr(watch, "_branch_still_unmerged", lambda b: False)
    note = watch.unmerged_stale_note(_ev("wt-foo"))
    assert "陈旧自动销：已合并/分支已删" in note and "wt-foo" in note


def test_live_unmerged_keeps_silent_note(monkeypatch):
    """真实未合并 → 空串：照常呈现+唤醒（不得误杀）。"""
    monkeypatch.setattr(watch, "_branch_still_unmerged", lambda b: True)
    assert watch.unmerged_stale_note(_ev("wt-foo")) == ""


def test_unreadable_state_keeps_reporting(monkeypatch):
    """查不出（None）→ 空串（宁可报，不静默——兜底防漏唤）。"""
    monkeypatch.setattr(watch, "_branch_still_unmerged", lambda b: None)
    assert watch.unmerged_stale_note(_ev("wt-foo")) == ""


def test_other_subjects_and_types_are_untouched(monkeypatch):
    """非 wt-* 主题或非 NEEDS-LEADER：不查 git、不销（维持原呈现）。"""

    def _boom(_b):
        raise AssertionError("不该查 git")

    monkeypatch.setattr(watch, "_branch_still_unmerged", _boom)
    assert watch.unmerged_stale_note(_ev("20261023-083549")) == ""
    assert watch.unmerged_stale_note(_ev("wt-foo", kind=events.JOB_DONE)) == ""


def test_suspect_acked_reuses_single_verdict(monkeypatch):
    """自动唤醒通道（_suspect_acked）与 drain 呈现面同一判据、同一留痕。"""
    monkeypatch.setattr(watch, "_branch_still_unmerged", lambda b: False)
    ev = _ev("wt-foo")
    assert watch.unmerged_stale_note(ev) == watch._suspect_acked(ev)


# ---------------------------------------------------------------------------
# 手动通道：orch events drain 呈现面
# ---------------------------------------------------------------------------
def test_drain_autocancels_stale_and_presents_live(monkeypatch, capsys):
    """陈旧只留痕一行（不作待处理呈现、不带唤醒策略）；真实未合并照常呈现+唤醒策略。"""
    _emit("wt-stale")
    _emit("wt-live", "「已提交未合并」：面 live 有 2 个提交未合并")
    monkeypatch.setattr(
        watch,
        "_branch_still_unmerged",
        lambda b: {"wt-stale": False, "wt-live": True}[b],
    )

    rc = orch.cmd_events(Namespace(e_action="drain"))
    out = capsys.readouterr().out

    assert rc == 0
    assert "排空 2 起积压事件（含 1 条陈旧自动销）" in out
    stale_line = next(ln for ln in out.splitlines() if "wt-stale" in ln)
    live_line = next(ln for ln in out.splitlines() if "wt-live" in ln)
    assert "陈旧自动销：已合并/分支已删" in stale_line
    assert "唤醒策略" not in stale_line, "陈旧事件不得带唤醒策略"
    assert "陈旧自动销" not in live_line and "唤醒策略" in live_line
    assert events.unacked() == [], "两事件都应随 drain 推水位线完成销账"


def test_drain_second_run_is_idempotent(monkeypatch, capsys):
    """排空后同命令再跑：无积压（陈旧事件不复读）。"""
    _emit("wt-stale")
    monkeypatch.setattr(watch, "_branch_still_unmerged", lambda b: False)
    orch.cmd_events(Namespace(e_action="drain"))
    capsys.readouterr()
    rc = orch.cmd_events(Namespace(e_action="drain"))
    out = capsys.readouterr().out
    assert rc == 0 and "无积压事件" in out and "陈旧自动销" not in out


# ---------------------------------------------------------------------------
# 自动唤醒通道（watch/Stop 钩子路径）：queue_pressure 的 reasons/notes 分流
# ---------------------------------------------------------------------------
def test_stale_unmerged_downgraded_to_note_not_wake(iso_state, monkeypatch):
    _emit("wt-stale")
    monkeypatch.setattr(watch, "_branch_still_unmerged", lambda b: False)
    press = watch.queue_pressure(now=time.time() + 10_000_000)
    assert not any("统一事件流" in r for r in press["reasons"]), "陈旧不得进 reasons"
    assert any("陈旧自动销" in n for n in press["notes"]), "留痕须落在 notes"


def test_live_unmerged_still_wakes(iso_state, monkeypatch):
    _emit("wt-live")
    monkeypatch.setattr(watch, "_branch_still_unmerged", lambda b: True)
    press = watch.queue_pressure(now=time.time() + 10_000_000)
    assert any("统一事件流" in r for r in press["reasons"]), "真实未合并必须照常唤醒"


# ---------------------------------------------------------------------------
# 发射面加固：缓存按 main head 失效
# ---------------------------------------------------------------------------
def _fake_git(monkeypatch, heads_line: str, unmerged: str, calls: list):
    def _run(*args):
        calls.append(tuple(args))
        if args[:3] == ("worktree", "list", "--porcelain"):
            return 0, f"worktree {workflow.WORKTREES_ROOT / 'face-a'}\n"
        if args[:1] == ("for-each-ref",):
            return 0, heads_line
        if args[:2] == ("rev-list", "--count"):
            return 0, unmerged
        if args[:1] == ("merge-base",):
            return 0, "deadbeef1234"
        return 1, ""

    monkeypatch.setattr(drain, "_run_git", _run)


def test_unmerged_cache_fails_over_when_main_advances(iso_state, monkeypatch):
    """main 前进（合入分支）即缓存失效——分支未摘的面不得复读陈旧读数。"""
    drain._UNMERGED_CACHE.clear()
    calls: list = []
    _fake_git(monkeypatch, "wt-face-a abc123456789\nmain m1", "3", calls)
    assert len(drain.unmerged_worktree_alerts()) == 1
    assert sum(1 for c in calls if c[:2] == ("rev-list", "--count")) == 1

    calls.clear()  # main 前进到 m2（head 未变）：缓存失效、重算
    _fake_git(monkeypatch, "wt-face-a abc123456789\nmain m2", "0", calls)
    assert drain.unmerged_worktree_alerts() == []
    assert sum(1 for c in calls if c[:2] == ("rev-list", "--count")) == 1

    calls.clear()  # 同 main head 再问：缓存命中、零 rev-list（空闲 tick 语义保留）
    _fake_git(monkeypatch, "wt-face-a abc123456789\nmain m2", "0", calls)
    assert drain.unmerged_worktree_alerts() == []
    assert sum(1 for c in calls if c[:2] == ("rev-list", "--count")) == 0
