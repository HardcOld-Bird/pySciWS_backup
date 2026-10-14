"""backlog 批量计划器 + FIFO 执行的端到端：dry 干跑、成功销账（含 commit 提取）、失败跳过
needs_leader 且不阻塞后续项、锁忙时优雅退出、任务书含合并安全句。

backlog-fifo-20261014 后的形态：计划器（drain.drain_backlog）只组包入队，执行在 devops
FIFO worker（queueworker.run_worker）；本文件经 :func:`conftest.drain_once` 端到端驱动，
以 monkeypatch 假 do_dispatch 避免真派发。
"""

from __future__ import annotations

import json
from argparse import Namespace

from pysci.skills.orchestration.tools import dispatch, drain, orch, queues, workflow
from pysci.skills.orchestration.tools.dispatch import DispatchOutcome

from .conftest import drain_once, read_backlog, seed_backlog


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
    assert summary["kind"] == "plan", "计划器记账面：kind=plan（执行账归 worker）"
    assert summary["count"] == 2
    assert all(i["status"] == "dry" for i in summary["items"])
    assert all(i["status"] == "done" for i in read_backlog(iso_state))
    done = run_log.with_suffix(".done")
    assert done.exists()
    assert json.loads(done.read_text(encoding="utf-8"))["count"] == 2
    assert run_log.exists() and "backlog 计划器启动" in run_log.read_text(
        encoding="utf-8"
    )
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
    summary = drain_once()
    # backlog 条目 → 队列条目的形态转换（backlog-fifo-20261014 验收项）
    qitem = queues.items("devops")[0]
    assert qitem["kind"] == drain.BACKLOG_BATCH_KIND
    assert qitem["backlog"]["seed_id"] == "b1"
    assert qitem["backlog"]["ids"] == ["b1"]
    assert "backlog id=b1" in qitem["task"]  # 批次正文（含销账清单要求）
    # 合并安全句（任务书 §5）注入批次正文
    assert "合并前 git status" in qitem["task"]
    # 执行面：假派发收到 devops + quiet 通路
    assert seen["member"] == "devops"
    assert seen["quiet"] is True
    assert "backlog id=b1" in seen["text"]
    # 账务：销账 done + commit 三级定位
    assert summary["count"] == 1
    assert summary["items"][0]["status"] == "done"
    assert summary["items"][0]["commit"] == "abc1234def"
    assert qitem["status"] == "done"
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
    summary = drain_once()
    by_id = {i["id"]: i for i in summary["items"]}
    assert by_id["bad"]["status"] == "needs_leader"
    assert by_id["bad"]["kind"] == "blocked"
    # 一项卡住不阻塞后续：组包项在下一轮（worker 续计划）被当新种子消化
    assert by_id["good"]["status"] == "done"
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


# ---------------------------------------------------------------------------
# 跨 run 失败计数（backlog 20261012-073700-leader 要求 4：防重派盲目）
# ---------------------------------------------------------------------------
def _done(state, name: str, items: list) -> None:
    """落一份 ``devops-runs/<name>.done`` run 摘要（只关心 items 面）。"""
    d = state / "devops-runs"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.done").write_text(
        json.dumps(
            {"count": len(items), "elapsed_s": 1.0, "items": items},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_failure_counts_accumulates_across_runs(iso_state):
    """同一 run 内一条 id 至多败一次（失败即 needs_leader，drain 不再重取），
    所以「重复失败」这个信号只能跨 ``.done`` 汇总——本测试钉住汇总口径。"""
    _done(iso_state, "20261012-010000.done", [{"id": "a", "status": "run_failed"}])
    _done(
        iso_state,
        "20261012-020000.done",
        [
            {"id": "a", "status": "blocked"},
            {"id": "b", "status": "needs_leader"},
            {"id": "c", "status": "done"},
        ],
    )
    got = drain.failure_counts()
    assert got == {"a": 2, "b": 1}  # 成功的 c 不入账
    assert drain.failure_counts(limit=1) == {"a": 1, "b": 1}  # 只看最近一份
    assert drain.failure_counts(limit=0) == got


def test_failure_counts_excludes_quota_exhausted(iso_state):
    """额度耗尽是渠道级故障、条目无罪复位——计入会把「渠道没了」误读成「单写坏了」。"""
    _done(
        iso_state,
        "20261012-030000.done",
        [
            {"id": "q", "status": "quota_exhausted"},
            {"id": "r", "status": "parse_error"},
        ],
    )
    assert drain.failure_counts() == {"r": 1}


def test_failure_counts_fail_soft_on_bad_files(iso_state):
    d = iso_state / "devops-runs"
    d.mkdir(parents=True, exist_ok=True)
    (d / "broken.done").write_text("{不是 json", encoding="utf-8")
    (d / "noitems.done").write_text(json.dumps({"count": 0}), encoding="utf-8")
    (d / "junk.done").write_text(
        json.dumps(
            {"items": [None, {"status": "run_failed"}, {"id": "", "status": "blocked"}]}
        ),
        encoding="utf-8",
    )
    assert drain.failure_counts() == {}  # 坏摘要/缺 id/非字典项一律跳过，不抛


def test_status_shows_repeat_failure_hint(iso_state, monkeypatch, capsys):
    """组长决定重派的那一眼，必须同时看到「同条目累计失败次数」与出声提示。"""
    seed_backlog(
        iso_state,
        [
            {"id": "a", "status": "needs_leader", "summary": "大单"},
            {"id": "b", "status": "needs_leader", "summary": "小单"},
        ],
    )
    _done(iso_state, "20261012-010000.done", [{"id": "a", "status": "run_failed"}])
    _done(iso_state, "20261012-020000.done", [{"id": "a", "status": "run_failed"}])
    _done(iso_state, "20261012-030000.done", [{"id": "b", "status": "blocked"}])
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: False)
    orch.cmd_status(Namespace())
    out = capsys.readouterr().out
    assert "累计失败 2 次，勿原样重派" in out
    assert "重复失败 1 条" in out  # 只有 a 达阈值（≥2）
    assert "小单（累计失败" not in out  # 单次失败是常态，不出声


def test_status_quiet_without_repeat_failures(iso_state, monkeypatch, capsys):
    seed_backlog(iso_state, [{"id": "a", "status": "needs_leader", "summary": "s"}])
    _done(iso_state, "20261012-010000.done", [{"id": "a", "status": "run_failed"}])
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: False)
    orch.cmd_status(Namespace())
    out = capsys.readouterr().out
    assert "needs_leader 条目 1" in out
    assert "累计失败" not in out and "重复失败" not in out
