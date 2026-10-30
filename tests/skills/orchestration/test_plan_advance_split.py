"""advance **动作分离**（backlog 20261015-135000-f2-ledger）。

实案：`advance --evidence` 想收尾一个 self 环节，却先撞上在途结算——结算动作与收尾动作
搅在同一条命令里，输出叠在一起（c2 收尾被挤、b_last 误收 c2 文件都与此相关）。

原则：**一次命令一个动作，动作不被挤占**。带 ``--evidence`` 时只做证据收尾，并在输出里
如实提示「还有几项待结账」；结算另跑一次不带 ``--evidence`` 的 advance。
"""

from __future__ import annotations

from pysci.skills.orchestration.tools import plans, queues

from .test_plan_guards import _args, _finish, _new, _statuses

GOOD = "# 全流程报告：x\n\n## 任务概述\n\n…\n\n## 环节完成清单\n\n…\n"


def _seed_plan_with_a_finished_but_unsettled_job(qface, tmp_path) -> tuple[str, str]:
    """造「一个待收尾 self 环节 + 一个已派发且队列里已 done 的叶子」。

    Returns: ``(plan_id, evidence_path)``。
    """
    pid = "p-split"
    assert (
        _new(
            qface,
            pid,
            [
                {
                    "id": "s1",
                    "kind": "self",
                    "brief": "全流程报告",
                    "role": "final_report",
                },
                {"id": "s2", "member": "devops", "brief": "实施"},
            ],
        )
        == 0
    )
    plans.cmd_plan_advance(_args(pid))  # 引导 s1（self_awaiting），不入队 s2

    ev = tmp_path / "report.md"
    ev.write_text(GOOD, encoding="utf-8")
    # 把 s2 置成「已派发」并在队列里收尾（模拟：worker 已 JOB-DONE，只等结账）
    state = plans._load_state(pid)
    row = next(s for s in state["steps"] if s["id"] == "s2")
    row["status"] = "queued"
    plans._save_state(pid, state)
    queues.enqueue("devops", plan=pid, step="s2", task_file="")
    _finish(pid, "s2", "devops")
    return pid, str(ev)


def test_advance_with_evidence_does_not_settle_inflight(qface, tmp_path, capsys):
    pid, ev = _seed_plan_with_a_finished_but_unsettled_job(qface, tmp_path)
    assert plans._terminal_ready_ids(
        pid, plans._load_yaml(pid), plans._load_state(pid)
    ) == ["s2"], "前提：s2 已在队里收尾、待结账"
    capsys.readouterr()

    rc = plans.cmd_plan_advance(_args(pid, evidence=ev))

    out = capsys.readouterr().out
    assert rc == 0
    assert "[本次动作] 只做 evidence 收尾" in out, out
    assert "尚待结账" in out and "s2" in out, "须如实提示待结账项，而不是顺手结掉"
    st = _statuses(pid)
    assert st["s1"] == "done", "证据收尾本身要生效"
    assert st["s2"] == "queued", "在途结算**不得**被顺手做掉（动作不被挤占）"


def test_plain_advance_still_settles(qface, tmp_path, capsys):
    """不带 --evidence 的 advance 照旧结算（分离不是「取消结算」）。"""
    pid, _ev = _seed_plan_with_a_finished_but_unsettled_job(qface, tmp_path)
    capsys.readouterr()

    plans.cmd_plan_advance(_args(pid))

    assert _statuses(pid)["s2"] == "done", "普通 advance 仍要把待结账项结掉"


def test_terminal_ready_ids_is_read_only(qface, tmp_path):
    pid, _ev = _seed_plan_with_a_finished_but_unsettled_job(qface, tmp_path)
    plan, state = plans._load_yaml(pid), plans._load_state(pid)

    assert plans._terminal_ready_ids(pid, plan, state) == ["s2"]
    assert plans._terminal_ready_ids(pid, plan, state) == ["s2"], (
        "只读：连调两次不改变事实"
    )
    assert _statuses(pid)["s2"] == "queued"


def test_inflight_items_are_still_reported_with_evidence(qface, tmp_path, capsys):
    """在途（未收尾）项在 --evidence 路径下仍如实报出（不是静默）。"""
    pid = "p-inflight"
    assert (
        _new(
            qface,
            pid,
            [
                {"id": "s1", "kind": "self", "brief": "报告", "role": "final_report"},
                {"id": "s2", "member": "devops", "brief": "实施"},
            ],
        )
        == 0
    )
    plans.cmd_plan_advance(_args(pid))
    state = plans._load_state(pid)
    next(s for s in state["steps"] if s["id"] == "s2")["status"] = "queued"
    plans._save_state(pid, state)
    queues.enqueue("devops", plan=pid, step="s2", task_file="")  # 保持 pending＝在途
    ev = tmp_path / "r.md"
    ev.write_text(GOOD, encoding="utf-8")
    capsys.readouterr()

    plans.cmd_plan_advance(_args(pid, evidence=str(ev)))

    out = capsys.readouterr().out
    assert "在途 1 项仍在跑" in out and "s2" in out, out
