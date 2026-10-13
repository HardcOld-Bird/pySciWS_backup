"""成员 FIFO 队列 + worker 生命周期（REFACTOR.md §3.1，M4）。

队列形状走 ``workq``；worker 的锁/停止/run 名走 ``drain.WorkerPaths`` 一族（抽取共用）。
测试用**临时 state 根** + mock ``do_dispatch``（真 LLM 派发不在本环节验收范围）。
"""

from __future__ import annotations

from argparse import Namespace

import pytest

from pysci.skills.orchestration.tools import (
    dispatch,
    drain,
    orch,
    queues,
    queueworker,
    workq,
)


class _Out:
    """dispatch.DispatchOutcome 的最小替身。"""

    def __init__(self, code: int = 0, kind: str = "result", body: str = "ok"):
        self.code, self.kind, self.body, self.error = code, kind, body, ""


@pytest.fixture
def state(tmp_path, monkeypatch):
    """隔离 state 根（queues/ 在它下面）；orch 面走模块常量 patch。"""
    root = tmp_path / "state"
    monkeypatch.setattr(queues, "QUEUES_DIR", root / "queues")
    monkeypatch.setattr(queueworker, "ORCH_STATE_ROOT", root)
    return root


def _mock_dispatch(monkeypatch, *, code: int = 0):
    calls: list[tuple[str, str]] = []

    def fake(member, **kw):
        calls.append((str(member), str(kw.get("text") or kw.get("task") or "")))
        return _Out(code=code, kind="result" if code == 0 else "blocked")

    monkeypatch.setattr(dispatch, "do_dispatch", fake)
    return calls


# ---------------------------------------------------------------------------
# ① 入队 → worker 消化 → 结果落盘；队空退出销毁
# ---------------------------------------------------------------------------
def test_worker_consumes_fifo_one_by_one(state, monkeypatch):
    calls = _mock_dispatch(monkeypatch)
    queues.enqueue("figure", text="甲", root=state)
    queues.enqueue("figure", text="乙", root=state)
    run_log = state / "r.log"

    summary = queueworker.run_worker("figure", run_log, root=state)

    assert [c[1] for c in calls] == ["甲", "乙"]  # FIFO、**逐条**（生产队列不组包）
    assert summary["count"] == 2 and not summary.get("error")
    items = queues.items("figure", root=state)
    assert [i["status"] for i in items] == ["done", "done"]
    assert items[0]["result"]["kind"] == "result"  # 结果摘要落盘
    assert run_log.with_suffix(".done").exists()


def test_failure_item_marked_failed(state, monkeypatch):
    _mock_dispatch(monkeypatch, code=2)
    queues.enqueue("theory", text="会失败", root=state)
    queueworker.run_worker("theory", state / "r.log", root=state)
    assert queues.items("theory", root=state)[0]["status"] == workq.STATUS_FAILED


def test_empty_queue_exits_and_writes_done(state, monkeypatch):
    calls = _mock_dispatch(monkeypatch)
    run_log = state / "empty.log"
    summary = queueworker.run_worker("sim", run_log, root=state)
    assert calls == [] and summary["count"] == 0
    assert run_log.with_suffix(".done").exists()  # 队空 → 收尾摘要（进程退出销毁）
    assert (
        queueworker.lock_status(queueworker.member_paths("sim", root=state))["state"]
        != "running"
    )  # 锁已释放


# ---------------------------------------------------------------------------
# ② per-member 锁防双 worker
# ---------------------------------------------------------------------------
def test_second_worker_is_refused_while_lock_held(state, monkeypatch):
    from pysci.skills.orchestration.tools import drain

    _mock_dispatch(monkeypatch)
    paths = queueworker.member_paths("lit", root=state)
    paths.runs.mkdir(parents=True, exist_ok=True)
    assert drain.acquire_lock("held", paths) is not None  # 本进程（存活）持锁

    queues.enqueue("lit", text="x", root=state)
    out = queueworker.run_worker("lit", state / "r2.log", root=state)
    assert out.get("skipped") == "lock_busy"  # 双 worker 被拒
    drain.release_lock(state / "r.log", paths)
    # 锁释放后可正常消化
    assert queueworker.run_worker("lit", state / "r3.log", root=state)["count"] == 1


# ---------------------------------------------------------------------------
# ③ 孤儿回收
# ---------------------------------------------------------------------------
def test_worker_reclaims_orphan_in_progress(state, monkeypatch):
    calls = _mock_dispatch(monkeypatch)
    item = queues.enqueue("drawing", text="孤儿", root=state)
    workq.mutate(
        queues.queue_path("drawing", root=state),
        item["id"],
        status=workq.STATUS_IN_PROGRESS,
        started_at="T",
    )

    summary = queueworker.run_worker("drawing", state / "r.log", root=state)

    assert summary.get("reclaimed") == [item["id"]]  # 复位并照常消化
    assert [c[1] for c in calls] == ["孤儿"]
    assert queues.items("drawing", root=state)[0]["status"] == workq.STATUS_DONE


# ---------------------------------------------------------------------------
# ④ reviewer 附属：父子联动 + 「宿主+review 才算完成」
# ---------------------------------------------------------------------------
def test_review_child_and_host_completion(state, monkeypatch):
    _mock_dispatch(monkeypatch)
    host = queues.enqueue("writing", text="宿主交付", review=True, root=state)

    queueworker.run_worker("writing", state / "rw.log", root=state)

    kids = queues.review_children(host["id"], host_member="writing", root=state)
    assert len(kids) == 1 and kids[0]["kind"] == "review"
    assert kids[0]["parent"] == host["id"] and kids[0]["parent_member"] == "writing"
    # 宿主自收尾，但 review 未收尾 → 完成语义仍不成立
    assert (
        queues.completion(host["id"], host_member="writing", root=state)[0]
        == queues.AWAITING_REVIEW
    )

    queueworker.run_worker("reviewer", state / "rr.log", root=state)  # 消化子 job

    state_now, _why = queues.completion(host["id"], host_member="writing", root=state)
    assert state_now == queues.COMPLETE  # 宿主 + review 均收尾
    assert queues.items("writing", root=state)[0].get("review_result")  # 尽力回填


def test_job_without_review_completes_immediately(state, monkeypatch):
    _mock_dispatch(monkeypatch)
    host = queues.enqueue("deputy", text="无需复审", root=state)
    queueworker.run_worker("deputy", state / "rd.log", root=state)
    assert (
        queues.completion(host["id"], host_member="deputy", root=state)[0]
        == queues.COMPLETE
    )
    assert queues.review_children(host["id"], host_member="deputy", root=state) == []


# ---------------------------------------------------------------------------
# ⑤ orch 命令面 + 观察面
# ---------------------------------------------------------------------------
def test_orch_queues_enqueue_and_list(state, capsys):
    assert orch.main(["queues", "enqueue", "figure", "--text", "甲", "--dry"]) == 0
    out = capsys.readouterr().out
    assert "已入队" in out and "figure" in out
    assert orch.main(["queues", "list"]) == 0
    assert "figure" in capsys.readouterr().out
    assert orch.main(["queues", "enqueue", "devops", "--text", "x", "--dry"]) == 2


def test_water_lines_marks_backlog(state):
    queues.enqueue("sim", text="a", root=state)
    lines = "\n".join(queues.water_lines(root=state))
    assert "[sim]" in lines and "有积压" in lines


def test_status_reads_new_queue_water(iso_state, monkeypatch, capsys):
    """`orch status` 队列段能读出新队列水位（观察面接入，任务书验收）。"""
    monkeypatch.setattr(queues, "QUEUES_DIR", iso_state / "queues")
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: False)
    queues.enqueue("figure", text="待做", root=iso_state)

    assert orch.cmd_status(Namespace()) == 0

    out = capsys.readouterr().out
    assert "== 成员队列 ==" in out
    assert "[figure]" in out and "有积压" in out


def test_queue_members_exclude_devops_and_bencher():
    assert "figure" in queues.QUEUE_MEMBERS
    assert "devops" not in queues.QUEUE_MEMBERS
    assert "bencher" not in queues.QUEUE_MEMBERS
