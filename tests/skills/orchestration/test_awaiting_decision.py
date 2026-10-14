"""「因缺授权/缺裁决 blocked → 组长补授权 → 重排」闭环（backlog 20261013-193058-devops）。

病灶：`<blocked>` 里点名要的授权，只有组长**主动**写附注才闭环；机械面上没有任何东西提示
「这条上轮是缺授权被拒的」。于是重排后 worker 若仍缺授权，只会照同一成因再 blocked 一次
——白烧一跳。本单给这类条目留可追踪痕迹：drain 在交付 `<blocked reason="needs_decision">`
时给条目置 `awaiting_decision` 标记 + 存原点名原文，`orch status` 单列，重排任务书自动注入
「原点名事项 ↔ 本次补齐情况」对照行。

判据（success_criteria）：条目带该标记 → `backlog_task_text()` 出现该对照行。
"""

from __future__ import annotations

from argparse import Namespace

from pysci.skills.orchestration.tools import dispatch, drain, orch, workflow
from pysci.skills.orchestration.tools.dispatch import DispatchOutcome
from pysci.skills.orchestration.tools.workflow import AWAITING_DECISION_KEY

from .conftest import drain_once, read_backlog, seed_backlog


def _item(**kw) -> dict:
    base = {
        "id": "20261013-193500-leader",
        "member": "leader",
        "summary": "带宽治理 B",
        "evidence": "state/suggestions/approved/x.md",
        "status": "pending",
    }
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# ① 任务书对照行（success_criteria 原文钉法）
# ---------------------------------------------------------------------------
def test_taskbook_shows_comparison_line_when_marked():
    text = workflow.backlog_task_text(
        _item(
            **{
                AWAITING_DECISION_KEY: True,
                "blocked_report": "点名授权：bandwidth 反转熔断、兜底 kind 名",
                "note": "组长裁决：熔断授权反转仅限 bandwidth；kind 名用 channel_saturated",
            }
        )
    )
    assert "补授权" in text and "重排" in text
    assert "原点名事项" in text and "bandwidth 反转熔断" in text
    assert "本次补齐情况" in text and "channel_saturated" in text


def test_taskbook_no_line_without_marker():
    assert "原点名事项" not in workflow.backlog_task_text(_item())


def test_taskbook_warns_when_authorization_still_missing():
    """标记在、附注空 → 出声「尚未补」：重排会再白烧，别让 worker 闷头再 blocked 一次。"""
    text = workflow.backlog_task_text(_item(**{AWAITING_DECISION_KEY: True}))
    assert "本次补齐情况" in text
    assert "尚未补" in text


def test_taskbook_falls_back_when_blocked_report_absent():
    """标记在、原点名原文缺 → 指回 note 全文，不崩、不留空。"""
    text = workflow.backlog_task_text(
        _item(**{AWAITING_DECISION_KEY: True, "note": "授权已给"})
    )
    assert "原点名事项" in text and "note" in text


# ---------------------------------------------------------------------------
# ② drain 侧：reason=needs_decision 才置标记（其余成因不需要授权）
# ---------------------------------------------------------------------------
def _run_blocked(iso_state, monkeypatch, *, reason: str, body: str = "缺判决记录"):
    seed_backlog(
        iso_state,
        [{"id": "b1", "status": "pending", "summary": "s", "member": "leader"}],
    )
    monkeypatch.setattr(
        dispatch,
        "do_dispatch",
        lambda *a, **k: DispatchOutcome(
            code=2, kind="blocked", body=body, reason=reason
        ),
    )
    drain_once()  # 计划器入队 → devops FIFO worker 执行并结算
    return read_backlog(iso_state)[0]


def test_drain_marks_awaiting_decision_for_needs_decision(iso_state, monkeypatch):
    item = _run_blocked(
        iso_state, monkeypatch, reason="needs_decision", body="缺判决记录：须补裁决"
    )
    assert item["status"] == "needs_leader"
    assert item[AWAITING_DECISION_KEY] is True
    assert "缺判决记录" in item["blocked_report"]


def test_drain_does_not_mark_for_capacity(iso_state, monkeypatch):
    item = _run_blocked(iso_state, monkeypatch, reason="capacity")
    assert AWAITING_DECISION_KEY not in item


def test_drain_does_not_mark_for_unlabeled_blocked(iso_state, monkeypatch):
    item = _run_blocked(iso_state, monkeypatch, reason="")
    assert AWAITING_DECISION_KEY not in item


def test_marked_item_flows_into_redispatched_taskbook(iso_state, monkeypatch):
    """闭环全链：blocked(reason=needs_decision) → 条目带标记 → 重排任务书出对照行。"""
    item = _run_blocked(
        iso_state, monkeypatch, reason="needs_decision", body="点名授权 X/Y"
    )
    # 组长补授权：把授权写进 note（重排前的动作）
    item["note"] = "授权 X/Y 均已给"
    text = workflow.backlog_task_text(item)
    assert "原点名事项" in text and "点名授权 X/Y" in text
    assert "本次补齐情况" in text and "X/Y 均已给" in text


# ---------------------------------------------------------------------------
# ③ orch status 单列
# ---------------------------------------------------------------------------
def test_status_lists_awaiting_decision(iso_state, monkeypatch, capsys):
    seed_backlog(
        iso_state,
        [
            {
                "id": "a",
                "status": "needs_leader",
                "summary": "缺授权卡住",
                AWAITING_DECISION_KEY: True,
            }
        ],
    )
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: False)
    orch.cmd_status(Namespace())
    out = capsys.readouterr().out
    assert "上轮因缺授权/裁决 blocked" in out


def test_status_quiet_when_no_awaiting_decision(iso_state, monkeypatch, capsys):
    seed_backlog(
        iso_state, [{"id": "a", "status": "needs_leader", "summary": "普通卡点"}]
    )
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: False)
    orch.cmd_status(Namespace())
    assert "上轮因缺授权" not in capsys.readouterr().out
