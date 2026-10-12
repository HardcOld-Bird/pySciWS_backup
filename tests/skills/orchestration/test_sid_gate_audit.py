"""sid 在途闸 + blocked 全文留痕（backlog 20261012-184000-leader 修复面④⑤）。

修复面①②③（第四面进程回收、世代守卫、终止/隔离阳性测试）已在 c8219ca 落 main；本模块钉
补的④⑤（#76 item4/5）：

- **④ sid 在途闸**：resume 目标 sid 若上一跳进程仍活（运行态 pid 判活 + argv 核对双闸），
  禁止双进程同 sid 并发写会话历史（#69 实证两进程共写 6f0cc8e5…）。人工派发默认 rc=2
  拒绝并点名占用 PID/任务书；``--force-new-session`` / drain 批次 ``auto=True`` 改起新
  sid，台账留 ``sid_new_reason=prev_alive``。
- **⑤ blocked 全文留痕**：needs_leader note 曾截 160 字让两案关键证据变残片（违背 §0.4）。
  blocked/run_failed 全文落 ``pods/<m>/outbox/delivery-<ts>.xml``，note 不再截断。
- 读数面：``orch ledger`` 行打「换席[原因]」——批次静默换 sid 不可读等于没标。

量真实读路：``pid``/``pid_task`` 是运行态会话池字段（``note_hop_pid`` 派生瞬间写、正常
收尾被覆盖），判活走 ``procs.pid_alive`` + ``procs.pid_is_session``——桩只给 pid 不给
argv 核对就把闸测成哑巴（同 184000 第四面教训）。
"""

from __future__ import annotations

import json
from argparse import Namespace

from pysci.skills.orchestration.tools import dispatch, drain, orch, workflow
from pysci.skills.orchestration.tools.dispatch import DispatchOutcome
from pysci.skills.orchestration.tools.ledger import LedgerEntry

from .conftest import (
    FAKE_HOP_PID,
    envelope,
    fake_headless,
    read_backlog,
    seed_backlog,
    set_models,
    sid_from_argv,
    write_runtime,
)

MEMBER = "quotamember"
OCCUPIED_SID = "deadbeef-0000-0000-0000-000000000001"
OK = "<result>完工，commit abc1234 已合并。</result>"
BUILTIN = "Qwen3.8-Flash"


def _seed_inflight_session(state, sid: str, *, pid: int, task: str) -> None:
    """往运行态会话池预置一具带 pid/pid_task 的会话（=上一跳进程仍活的证据）。"""
    write_runtime(
        state,
        {
            "version": 1,
            "members": {
                MEMBER: {
                    "sessions": [
                        {
                            "sid": sid,
                            "status": "active",
                            "last_active": "9999",
                            "hops": 3,
                            "pid": pid,
                            "pid_task": task,
                            "ctx_ratio": 0.3,
                        }
                    ]
                }
            },
        },
    )


def _alive_procs(monkeypatch) -> None:
    """桩：pid 判活为真且 argv 含该 sid（两道闸都过才算在途）。"""
    monkeypatch.setattr(dispatch.procs, "pid_alive", lambda pid: pid == FAKE_HOP_PID)
    monkeypatch.setattr(
        dispatch.procs, "pid_is_session", lambda pid, sid: sid == OCCUPIED_SID
    )


# ---------------------------------------------------------------------------
# ④ sid 在途闸：拒绝 / 放行 / 批次自动换 / fail-open 两支
# ---------------------------------------------------------------------------
def test_manual_refuses_when_prev_hop_alive(iso_dispatch, monkeypatch, capsys):
    """人工派发（默认）：上一跳进程仍活 → rc=2 拒绝并点名占用 PID 与任务书。"""
    set_models(iso_dispatch, flash=BUILTIN)
    _seed_inflight_session(
        iso_dispatch, OCCUPIED_SID, pid=FAKE_HOP_PID, task="task-20261012-185839-x.md"
    )
    _alive_procs(monkeypatch)
    calls = fake_headless(
        monkeypatch, lambda n, cmd: envelope(OK, is_error=False, rc=0)
    )
    o = dispatch.do_dispatch(
        MEMBER, text="出图", quiet=True, no_checks=True, session=OCCUPIED_SID
    )
    assert o.code == 2, "双进程同 sid 会并发写历史，人工派发必须拒绝"
    assert str(FAKE_HOP_PID) in o.error, "拒绝信息必须点名占用 PID"
    assert "task-20261012-185839-x.md" in o.error, "拒绝信息必须点名占用任务书"
    assert calls == [], "拒绝通路不得派生"
    assert "仍在途" in capsys.readouterr().out, "拒绝必须当场可见"


def test_manual_force_new_session_opens_new_sid(iso_dispatch, monkeypatch):
    """``--force-new-session``：人工放行 = 改起新 sid（不与在途进程同 sid）。"""
    set_models(iso_dispatch, flash=BUILTIN)
    _seed_inflight_session(iso_dispatch, OCCUPIED_SID, pid=FAKE_HOP_PID, task="t.md")
    _alive_procs(monkeypatch)
    calls = fake_headless(
        monkeypatch, lambda n, cmd: envelope(OK, is_error=False, rc=0)
    )
    o = dispatch.do_dispatch(
        MEMBER,
        text="出图",
        quiet=True,
        no_checks=True,
        session=OCCUPIED_SID,
        force_new_session=True,
    )
    assert o.code == 0 and o.kind == "result"
    assert len(calls) == 1
    new_sid = sid_from_argv(calls[0])
    assert new_sid and new_sid != OCCUPIED_SID, "放行必须起新 sid，不复用占用中的"
    assert o.sid_new_reason == "prev_alive", "放行须留痕换新原因（台账同源）"


def test_batch_auto_opens_new_sid_and_annotates(iso_dispatch, monkeypatch):
    """drain 批次（auto=True）：上一跳仍活 → 自动换新 sid，台账标注 prev_alive。"""
    set_models(iso_dispatch, flash=BUILTIN)
    _seed_inflight_session(iso_dispatch, OCCUPIED_SID, pid=FAKE_HOP_PID, task="t.md")
    _alive_procs(monkeypatch)
    calls = fake_headless(
        monkeypatch, lambda n, cmd: envelope(OK, is_error=False, rc=0)
    )
    o = dispatch.do_dispatch(
        MEMBER, text="出图", quiet=True, no_checks=True, session="latest", auto=True
    )
    assert o.code == 0
    assert sid_from_argv(calls[0]) != OCCUPIED_SID, "批次不得与在途进程同 sid"
    assert o.sid_new_reason == "prev_alive"
    # 台账行同名字段留痕（读数面与 outcome 一致，不靠人肉翻 run 日志）
    rows = [
        json.loads(ln)
        for ln in (iso_dispatch / "ledger.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    assert rows[-1]["sid_new_reason"] == "prev_alive"


def test_dead_pid_fails_open_and_resumes(iso_dispatch, monkeypatch):
    """pid 已死（判活不过）→ 闸 fail-open 照续，不冤枉正常跑。"""
    set_models(iso_dispatch, flash=BUILTIN)
    _seed_inflight_session(iso_dispatch, OCCUPIED_SID, pid=FAKE_HOP_PID, task="t.md")
    monkeypatch.setattr(dispatch.procs, "pid_alive", lambda pid: False)
    monkeypatch.setattr(dispatch.procs, "pid_is_session", lambda pid, sid: False)
    calls = fake_headless(
        monkeypatch, lambda n, cmd: envelope(OK, is_error=False, rc=0)
    )
    o = dispatch.do_dispatch(
        MEMBER, text="出图", quiet=True, no_checks=True, session=OCCUPIED_SID
    )
    assert o.code == 0
    assert sid_from_argv(calls[0]) == OCCUPIED_SID, "在途证据不成立必须照续旧 sid"


def test_argv_mismatch_fails_open(iso_dispatch, monkeypatch):
    """pid 活着但 argv 不含该 sid（pid 被系统复用）→ fail-open，不误判在途。"""
    set_models(iso_dispatch, flash=BUILTIN)
    _seed_inflight_session(iso_dispatch, OCCUPIED_SID, pid=FAKE_HOP_PID, task="t.md")
    monkeypatch.setattr(dispatch.procs, "pid_alive", lambda pid: True)
    monkeypatch.setattr(dispatch.procs, "pid_is_session", lambda pid, sid: False)
    calls = fake_headless(
        monkeypatch, lambda n, cmd: envelope(OK, is_error=False, rc=0)
    )
    o = dispatch.do_dispatch(
        MEMBER, text="出图", quiet=True, no_checks=True, session=OCCUPIED_SID
    )
    assert o.code == 0 and sid_from_argv(calls[0]) == OCCUPIED_SID


# ---------------------------------------------------------------------------
# ⑤ blocked/run_failed 全文留痕：outbox 审计副本
# ---------------------------------------------------------------------------
BLOCKED_BODY = "<blocked>\n关键证据行——被截断就变残片。\n" * 20 + "\n</blocked>"


def test_blocked_writes_outbox_audit_copy(iso_dispatch, monkeypatch):
    """blocked 交付全文落 pods/<m>/outbox/delivery-<ts>.xml（fail-open 不反噬交付）。"""
    set_models(iso_dispatch, flash=BUILTIN)
    fake_headless(
        monkeypatch, lambda n, cmd: envelope(BLOCKED_BODY, is_error=False, rc=0)
    )
    o = dispatch.do_dispatch(MEMBER, text="出图", quiet=True, no_checks=True)
    assert o.kind == "blocked"
    outbox = iso_dispatch.parent / "pods" / MEMBER / "outbox"
    copies = list(outbox.glob("delivery-*.xml"))
    assert copies, "blocked 交付必须落 outbox 审计副本"
    assert "关键证据行" in copies[0].read_text(encoding="utf-8"), "副本必须含全文"


def test_run_failed_writes_outbox_audit_copy(iso_dispatch, monkeypatch):
    """run_failed/系统性同样落审计副本——它们在 drain 也产 needs_leader note。"""
    set_models(iso_dispatch, flash=BUILTIN)
    fake_headless(
        monkeypatch,
        lambda n, cmd: envelope("炸在第 99 轮，栈迹关键句", is_error=True, rc=1),
    )
    o = dispatch.do_dispatch(MEMBER, text="出图", quiet=True, no_checks=True)
    assert o.code == 2
    outbox = iso_dispatch.parent / "pods" / MEMBER / "outbox"
    copies = list(outbox.glob("delivery-*.xml"))
    assert copies and "栈迹关键句" in copies[0].read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 读数面：换席痕迹必须出现在 ``orch ledger`` 行（不靠人肉翻 run 日志）
# ---------------------------------------------------------------------------
def _row(**kw) -> LedgerEntry:
    return LedgerEntry(
        ts="2026-10-13T06:00:00+08:00",
        member=MEMBER,
        session_id=OCCUPIED_SID,
        kind="result",
        num_turns=3,
        duration_ms=1000,
        **kw,
    )


def test_ledger_row_renders_sid_new_reason(monkeypatch, capsys):
    """批次自动换 sid 后，``orch ledger`` 行必须打「换席[原因]」——静默换席不可读等于没标。"""
    monkeypatch.setattr(orch, "read_all", lambda: [_row(sid_new_reason="prev_alive")])
    orch.cmd_ledger(Namespace(member=None, days=0, stats=False, tail=10))
    assert "换席[prev_alive]" in capsys.readouterr().out


def test_ledger_row_omits_sid_marker_when_not_switched(monkeypatch, capsys):
    """没换席的行不多打标记（旧台账行读法不变，同 tags 段先例）。"""
    monkeypatch.setattr(orch, "read_all", lambda: [_row()])
    orch.cmd_ledger(Namespace(member=None, days=0, stats=False, tail=10))
    assert "换席[" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# ⑤ needs_leader note 不截断（drain 侧）：截 160 字曾把两案关键证据切成残片
# ---------------------------------------------------------------------------
def test_drain_needs_leader_note_keeps_full_text(iso_state, monkeypatch):
    """种子失败 → note 留交付**全文**，并点名**被派发者**的 outbox（不是提请者）。"""
    seed_backlog(
        iso_state,
        [
            {"id": "s1", "status": "pending", "member": "leader"},
            {"id": "s2", "status": "pending", "member": "leader"},
        ],
    )
    seed, group = workflow.backlog_take_batch(4)
    long_body = "并发写入者证据：本会话零成功写入\n" * 12  # 远超旧的 160 字截断线
    monkeypatch.setattr(
        dispatch,
        "do_dispatch",
        lambda *a, **k: DispatchOutcome(
            code=2, kind="blocked", body=long_body, member="devops"
        ),
    )
    drain._drain_batch(seed, group, drain.new_run_log(), dry=False)
    note = next(i["note"] for i in read_backlog(iso_state) if i["id"] == "s1")
    assert long_body.strip() in note, "note 必须留全文，截断=证据变残片"
    assert "pods/devops/outbox" in note, (
        "审计副本路径点名被派发者（member=leader 是提请者）"
    )
