"""drain 循环：dry 干跑、成功销账（含 commit 提取）、失败跳过 needs_leader 且不阻塞
后续项、锁忙时优雅退出、任务书含合并安全句。

覆盖任务书验证要求 2/3 的 drain 侧行为（以 monkeypatch 假 do_dispatch 避免真派发）。
"""

from __future__ import annotations

import json

from pysci.skills.orchestration.tools import dispatch, drain, workflow
from pysci.skills.orchestration.tools.dispatch import DispatchOutcome

from .conftest import read_backlog, seed_backlog


def test_dry_drain_consumes_all_and_writes_done(iso_state):
    seed_backlog(
        iso_state,
        [
            {"id": "b1", "status": "pending", "summary": "s1"},
            {"id": "b2", "status": "pending", "summary": "s2"},
        ],
    )
    run_log = drain.new_run_log()
    summary = drain.drain_backlog(run_log, dry=True)
    assert summary["count"] == 2
    assert all(i["status"] == "dry" for i in summary["items"])
    assert all(i["status"] == "done" for i in read_backlog(iso_state))
    done = run_log.with_suffix(".done")
    assert done.exists()
    assert json.loads(done.read_text(encoding="utf-8"))["count"] == 2
    assert run_log.exists() and "drain 启动" in run_log.read_text(encoding="utf-8")
    assert not drain.LOCK_PATH.exists()  # 结束已释放锁


def test_dry_drain_empty_backlog(iso_state):
    seed_backlog(iso_state, [])
    summary = drain.drain_backlog(drain.new_run_log(), dry=True)
    assert summary["count"] == 0
    assert not drain.LOCK_PATH.exists()


def test_success_marks_done_and_extracts_commit(iso_state, monkeypatch):
    seed_backlog(iso_state, [{"id": "b1", "status": "pending", "summary": "s1"}])
    seen = {}

    def fake_dispatch(member, **kw):
        seen["member"] = member
        seen["text"] = kw["text"]
        seen["quiet"] = kw.get("quiet")
        return DispatchOutcome(
            code=0, kind="result", body="改动完成，commit abc1234def 已合并回 main。"
        )

    monkeypatch.setattr(dispatch, "do_dispatch", fake_dispatch)
    summary = drain.drain_backlog(drain.new_run_log())
    assert seen["member"] == "devops"
    assert seen["quiet"] is True
    assert "backlog id=b1" in seen["text"]
    # 合并安全句（任务书 §5）注入 devops 派发
    assert "合并前 git status" in seen["text"]
    assert summary["items"][0]["status"] == "done"
    assert summary["items"][0]["commit"] == "abc1234def"
    assert read_backlog(iso_state)[0]["status"] == "done"


def test_failure_skips_to_needs_leader_and_continues(iso_state, monkeypatch):
    seed_backlog(
        iso_state,
        [
            {"id": "bad", "status": "pending", "summary": "会 blocked"},
            {"id": "good", "status": "pending", "summary": "会成功"},
        ],
    )

    def fake_dispatch(member, **kw):
        if "id=bad" in kw["text"]:
            return DispatchOutcome(
                code=2, kind="blocked", body="", error="main 工作区有未提交改动"
            )
        return DispatchOutcome(code=0, kind="result", body="done commit deadbeef")

    monkeypatch.setattr(dispatch, "do_dispatch", fake_dispatch)
    summary = drain.drain_backlog(drain.new_run_log())
    by_id = {i["id"]: i for i in summary["items"]}
    assert by_id["bad"]["status"] == "needs_leader"
    assert by_id["bad"]["kind"] == "blocked"
    assert by_id["good"]["status"] == "done"  # 一项卡住不阻塞后续
    items = {i["id"]: i for i in read_backlog(iso_state)}
    assert items["bad"]["status"] == "needs_leader"
    assert items["good"]["status"] == "done"


def test_drain_exits_gracefully_when_lock_busy(iso_state, monkeypatch):
    seed_backlog(iso_state, [{"id": "b1", "status": "pending", "summary": "s1"}])
    (iso_state / "devops.lock").write_text(
        json.dumps({"pid": 4242, "started": drain._now(), "run_log": "r.log"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: True)
    called = []
    monkeypatch.setattr(
        dispatch,
        "do_dispatch",
        lambda *a, **k: called.append(1) or DispatchOutcome(code=0, kind="result"),
    )
    run_log = drain.new_run_log()
    summary = drain.drain_backlog(run_log)
    assert summary.get("skipped") == "lock_busy"
    assert called == []  # 未消费任何 backlog 项
    assert read_backlog(iso_state)[0]["status"] == "pending"  # 仍 pending
    # 未持有锁的进程不得删除他人锁
    assert (iso_state / "devops.lock").exists()


def test_backlog_task_text_shared_builder():
    text = workflow.backlog_task_text(
        {"id": "x1", "summary": "摘要A", "evidence": "证据B", "note": "附注C"}
    )
    assert "id=x1" in text and "摘要A" in text and "证据B" in text and "附注C" in text
    assert "合并前 git status" in text  # 合并安全句
    # push 规程句与 charter 一致（限时 best-effort），旧「不 push」矛盾句不得回潮
    assert "push 规程" in text and "best-effort" in text
    assert "不 push" not in text
