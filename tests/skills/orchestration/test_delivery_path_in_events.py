"""JOB-DONE 事件携带**交付全文留档**路径（backlog 20261015-205906）。

现场：JOB-DONE 唤醒行只给「…收尾：done（result）｜建议处置：events drain → plan advance」，
**不带**留档路径。组长若想在 `plan advance`（结算+派发）**之前**先读完整交付把关，只能猜目录
（旧 outbox）/翻 ledger/`find -newermt`，多跑 3 轮——而路径其实早在结果里（``body_file``），
只是没透到事件面。

修法：事件细节补「交付全文留档：<绝对路径>」。事件细节原样进 `watch.format_event`（唤醒行）
与 `events drain` 输出（两者都渲染 ``detail``），故一处补齐两处可见。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pysci.paths import ORCHESTRATION_ROOT, PROJECT_ROOT
from pysci.skills.orchestration.tools import (
    events,
    orch,
    queues,
    queueworker,
    watch,
)

from .conftest import envelope, fake_headless

MEMBER = "quotamember"
OK = "<result>完工，commit abc1234 已合并。</result>"

PLAYBOOK = (
    ORCHESTRATION_ROOT
    / "skills"
    / "orchestration"
    / "references"
    / "dispatch-playbook.md"
)


# ---------------------------------------------------------------------------
# 纯段：_delivery_note
# ---------------------------------------------------------------------------
def test_delivery_note_makes_relative_absolute():
    note = queueworker._delivery_note(
        {"body_file": "orchestration/state/deliveries/j1.md"}
    )
    assert "交付全文留档：" in note
    p = Path(note.split("交付全文留档：", 1)[1])
    assert p.is_absolute()
    assert p == PROJECT_ROOT / "orchestration/state/deliveries/j1.md"


def test_delivery_note_absent_is_silent():
    """留档缺失（写失败 fail-soft）→ 不加噪声。"""
    assert queueworker._delivery_note({}) == ""
    assert queueworker._delivery_note({"body_file": ""}) == ""
    assert queueworker._delivery_note(None) == ""


# ---------------------------------------------------------------------------
# 端到端：真跑一件 job → 事件细节带路径，且**文件可直接读**
# ---------------------------------------------------------------------------
def test_job_done_event_carries_readable_absolute_path(
    iso_state, iso_dispatch, monkeypatch
):
    fake_headless(monkeypatch, lambda n, cmd: envelope(OK, is_error=False, rc=0))
    queues.enqueue(MEMBER, text="出图", session="new")
    wlog = queueworker.new_run_log(queueworker.member_paths(MEMBER))

    queueworker.run_worker(MEMBER, wlog)

    evs = [e for e in events.read_all() if e["event"] == events.JOB_DONE]
    assert len(evs) == 1, [e["event"] for e in events.read_all()]
    detail = evs[0]["detail"]
    assert "交付全文留档：" in detail, detail

    p = Path(detail.split("交付全文留档：", 1)[1].strip())
    assert p.is_absolute(), detail
    assert p.is_file(), f"留档路径不可用：{p}"
    text = p.read_text(encoding="utf-8")
    assert "## 交付正文" in text and "commit abc1234" in text  # 能读到全文


def test_wake_line_and_drain_both_show_the_path(
    iso_state, iso_dispatch, monkeypatch, capsys
):
    """两个事件消费面（watch 唤醒行 / events drain 输出）都能看到该路径。"""
    fake_headless(monkeypatch, lambda n, cmd: envelope(OK, is_error=False, rc=0))
    queues.enqueue(MEMBER, text="出图", session="new")
    queueworker.run_worker(
        MEMBER, queueworker.new_run_log(queueworker.member_paths(MEMBER))
    )

    ev = next(e for e in events.read_all() if e["event"] == events.JOB_DONE)

    line = watch.format_event(ev)  # watch 命中时首行即此
    assert "交付全文留档：" in line

    assert orch.main(["events", "drain"]) == 0
    assert "交付全文留档：" in capsys.readouterr().out


def test_no_path_note_when_delivery_archive_missing(
    iso_state, iso_dispatch, monkeypatch
):
    """留档写失败（fail-soft 空串）时事件仍发，但不带路径——不制造假路径。"""
    result = {"body_file": "", "kind": "produce"}
    # 直接驱动事件发射（不经 worker）：本用例只钉「无路径即无该段」
    queueworker._notify(
        queues.queue_path(MEMBER),
        {"id": "j-x", "member": MEMBER},
        "done",
        result,
        MEMBER,
    )
    ev = next(e for e in events.read_all() if e["event"] == events.JOB_DONE)
    assert "交付全文留档" not in ev["detail"]
    assert "job 收尾：done" in ev["detail"]


# ---------------------------------------------------------------------------
# 文档面：留档规范路径有注明
# ---------------------------------------------------------------------------
def test_playbook_documents_delivery_archive_path():
    md = PLAYBOOK.read_text(encoding="utf-8")
    assert "orchestration/state/deliveries/" in md
    assert "交付全文留档" in md


@pytest.mark.parametrize("face", ["watch", "events"])
def test_detail_is_the_single_shared_source(face):
    """两消费面共用同一 `detail` —— 一处补齐两处可见（防「只补了一面」）。"""
    ev = {
        "event": events.JOB_DONE,
        "subject": "j1",
        "detail": "x｜交付全文留档：/abs/p.md",
        "action": "a",
    }
    rendered = watch.format_event(ev) if face == "watch" else events.format_event(ev)
    assert "交付全文留档：/abs/p.md" in rendered
