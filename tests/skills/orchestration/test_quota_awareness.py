"""额度类失败识别与 drain 全局停止（backlog 20261010-orch-quota-awareness）。

2026-10-10 事故：内置 Qwen3.8-Max 烧穿订阅 credit 后 drain 连烧 5 项全部
stop_sequence 失败，错误原文（"You've reached your credit usage limit"）只在
envelope/jsonl 里，且逐项 needs_leader 污染 backlog 语义。本组测试钉住：
do_dispatch 识别 quota_exhausted（不自动重试）、drain 遇之全局停止（当前条目复位
pending、剩余保持 pending、.done 注明系统性故障）。
"""

from __future__ import annotations

import json

import pytest

from pysci.skills.orchestration.tools import dispatch, drain, ledger, registry, workflow
from pysci.skills.orchestration.tools.dispatch import DispatchOutcome
from pysci.skills.orchestration.tools.runner import Envelope

from .conftest import read_backlog, seed_backlog

# 事故原文（会话 jsonl 实测）
QUOTA_TEXT = (
    "You've reached your credit usage limit. Please upgrade your "
    "subscription plan to get more resources. Report Issue (input /feedback)"
)


# ---------------------------------------------------------------------------
# 检测面（_detect_quota_exhaustion）
# ---------------------------------------------------------------------------
def test_detect_quota_in_envelope_result():
    env = Envelope(result=QUOTA_TEXT, is_error=True, stop_reason="stop_sequence")
    assert dispatch._detect_quota_exhaustion(env, "", "")


def test_detect_quota_in_stderr():
    env = Envelope(result="", is_error=True)
    assert dispatch._detect_quota_exhaustion(env, "", f"fatal: {QUOTA_TEXT}")


def test_detect_quota_in_raw_stdout():
    env = Envelope(result="", is_error=True)
    assert dispatch._detect_quota_exhaustion(
        env, json.dumps({"result": QUOTA_TEXT}), ""
    )


def test_detect_quota_case_insensitive():
    env = Envelope(result="CREDIT USAGE LIMIT reached")
    assert dispatch._detect_quota_exhaustion(env, "", "")


def test_detect_quota_no_false_positive():
    env = Envelope(result="任务完成，commit abc1234；速率限制 rate limit 已自愈")
    assert not dispatch._detect_quota_exhaustion(env, "", "")


# ---------------------------------------------------------------------------
# do_dispatch 集成（全隔离：tmp registry/ledger + 假 run_headless + 假 exe）
# ---------------------------------------------------------------------------
@pytest.fixture
def iso_dispatch(tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir()
    pod = tmp_path / "pods" / "quotamember"
    (pod / "inbox").mkdir(parents=True)
    reg_file = state / "registry.json"
    reg_file.write_text(
        json.dumps(
            {
                "version": 1,
                "exe": None,
                "models": {"max": "", "flash": ""},
                "members": {
                    "quotamember": {
                        "pod": str(pod),
                        "model_tier": "flash",
                        "max_turns": 3,
                        "timeout_s": 5,
                        "sessions": [],
                    }
                },
                "checks": {},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    exe = tmp_path / "fake-exe.exe"
    exe.write_bytes(b"")
    monkeypatch.setenv("PYSCI_ORCH_EXE", str(exe))
    monkeypatch.setattr(registry, "REGISTRY_PATH", reg_file)
    # Registry.save 的 path 默认值在类定义时绑定，patch 类属性改不了 __init__ 默认——
    # 直接 no-op save，防止测试写真实 registry.json
    monkeypatch.setattr(registry.Registry, "save", lambda self: None)
    monkeypatch.setattr(ledger, "LEDGER_PATH", state / "ledger.jsonl")
    monkeypatch.setattr(dispatch, "REPLIES_DIR", state / "replies")
    monkeypatch.setattr(dispatch, "SUGGESTIONS_DIR", state / "suggestions")
    monkeypatch.setattr(dispatch, "LINT_RULES_PATH", state / "taskbook-lint.json")
    return state


def _fake_headless(monkeypatch, responder):
    """monkeypatch dispatch.run_headless；responder(call_index) -> (rc, out, err)。"""
    calls = []

    def fake_run(cmd, cwd=None, env=None, timeout_s=None):
        calls.append(cmd)
        return responder(len(calls))

    monkeypatch.setattr(dispatch, "run_headless", fake_run)
    return calls


def _envelope(result, *, is_error=True, stop="stop_sequence", rc=1):
    return (
        rc,
        json.dumps(
            {
                "result": result,
                "is_error": is_error,
                "stop_reason": stop,
                "num_turns": 1,
                "duration_ms": 13000,
                "session_id": "aaaa1111",
            }
        ),
        "",
    )


def test_do_dispatch_quota_kind_and_no_retry(iso_dispatch, monkeypatch):
    calls = _fake_headless(monkeypatch, lambda n: _envelope(QUOTA_TEXT))
    o = dispatch.do_dispatch("quotamember", text="做某事", quiet=True)
    assert o.kind == "quota_exhausted" and o.code == 2
    assert len(calls) == 1, "额度类失败不得自动重试（重试只会再烧一跳）"
    assert "credit usage limit" in o.body
    line = (iso_dispatch / "ledger.jsonl").read_text(encoding="utf-8").strip()
    assert '"quota_exhausted"' in line


def test_do_dispatch_generic_failure_still_retries(iso_dispatch, monkeypatch):
    calls = _fake_headless(monkeypatch, lambda n: _envelope("internal boom"))
    o = dispatch.do_dispatch("quotamember", text="做某事", quiet=True)
    assert o.kind == "run_failed"
    assert len(calls) == 2, "普通运行失败保持自动重试一次"


def test_do_dispatch_retry_then_quota(iso_dispatch, monkeypatch):
    """首跑普通失败→重试撞上额度文案：仍归类 quota_exhausted。"""
    calls = _fake_headless(
        monkeypatch,
        lambda n: _envelope("internal boom") if n == 1 else _envelope(QUOTA_TEXT),
    )
    o = dispatch.do_dispatch("quotamember", text="做某事", quiet=True)
    assert o.kind == "quota_exhausted"
    assert len(calls) == 2


def test_do_dispatch_success_mentioning_quota_not_reclassified(
    iso_dispatch, monkeypatch
):
    """成功交付正文提及额度文案（如复盘任务）不得重分类——检测仅在失败分支。"""
    body = f"<result>事故复盘完成：原文为 {QUOTA_TEXT}，已钉住渠道策略。</result>"
    _fake_headless(
        monkeypatch, lambda n: _envelope(body, is_error=False, stop="end_turn", rc=0)
    )
    o = dispatch.do_dispatch(
        "quotamember", text="复盘额度事故", quiet=True, no_checks=True
    )
    assert o.kind == "result" and o.code == 0


def test_do_dispatch_quota_report_next(iso_dispatch, monkeypatch, capsys):
    _fake_headless(monkeypatch, lambda n: _envelope(QUOTA_TEXT))
    dispatch.do_dispatch("quotamember", text="做某事", quiet=False)
    out = capsys.readouterr().out
    assert "额度类失败（quota_exhausted）" in out
    assert "[NEXT]" in out and "registry models" in out
    assert "全局停止" in out


# ---------------------------------------------------------------------------
# workflow.backlog_requeue
# ---------------------------------------------------------------------------
def test_requeue_resets_in_progress(iso_state):
    seed_backlog(iso_state, [{"id": "r1", "status": "in_progress", "started_at": "x"}])
    workflow.backlog_requeue("r1", note="复位保现场")
    item = read_backlog(iso_state)[0]
    assert item["status"] == "pending"
    assert "started_at" not in item
    assert item["note"] == "复位保现场"


def test_requeue_ignores_non_in_progress(iso_state):
    seed_backlog(
        iso_state,
        [{"id": "d", "status": "done"}, {"id": "n", "status": "needs_leader"}],
    )
    workflow.backlog_requeue("d")
    workflow.backlog_requeue("n")
    by_id = {i["id"]: i["status"] for i in read_backlog(iso_state)}
    assert by_id == {"d": "done", "n": "needs_leader"}


def test_requeue_missing_file_noop(iso_state):
    workflow.backlog_requeue("ghost")  # 不抛异常即过


# ---------------------------------------------------------------------------
# drain 全局停止
# ---------------------------------------------------------------------------
def _quota_outcome():
    return DispatchOutcome(
        code=2, kind="quota_exhausted", body="", error="credit usage limit"
    )


def test_drain_quota_stops_globally(iso_state, monkeypatch):
    seed_backlog(
        iso_state,
        [
            {"id": "q1", "status": "pending", "summary": "s1"},
            {"id": "q2", "status": "pending", "summary": "s2"},
        ],
    )
    calls = []

    def fake_dispatch(member, **kw):
        calls.append(kw.get("slug"))
        return _quota_outcome()

    monkeypatch.setattr(dispatch, "do_dispatch", fake_dispatch)
    run_log = drain.new_run_log()
    summary = drain.drain_backlog(run_log)
    assert summary["stopped"] == "quota_exhausted"
    assert "系统性故障" in summary["error"]
    assert calls == ["q1"], "q2 不得再被派发（不再逐项烧）"
    by_id = {i["id"]: i for i in read_backlog(iso_state)}
    assert by_id["q1"]["status"] == "pending", "当前条目复位而非 needs_leader"
    assert "needs_leader_at" not in by_id["q1"]
    assert by_id["q2"]["status"] == "pending"
    assert not drain.LOCK_PATH.exists(), "锁已释放"
    done = json.loads(run_log.with_suffix(".done").read_text(encoding="utf-8"))
    assert done["stopped"] == "quota_exhausted" and "系统性故障" in done["error"]
    assert "全局停止" in run_log.read_text(encoding="utf-8")


def test_drain_quota_after_success_keeps_done(iso_state, monkeypatch):
    seed_backlog(
        iso_state,
        [
            {"id": "b1", "status": "pending", "summary": "s1"},
            {"id": "b2", "status": "pending", "summary": "s2"},
        ],
    )

    def fake_dispatch(member, **kw):
        if kw.get("slug") == "b1":
            return DispatchOutcome(code=0, kind="result", body="done commit ccc3333")
        return _quota_outcome()

    monkeypatch.setattr(dispatch, "do_dispatch", fake_dispatch)
    summary = drain.drain_backlog(drain.new_run_log())
    assert summary["stopped"] == "quota_exhausted"
    by_id = {i["id"]: i for i in read_backlog(iso_state)}
    assert by_id["b1"]["status"] == "done", "已成功项正常销账"
    assert by_id["b2"]["status"] == "pending"


def test_drain_needs_leader_path_unchanged(iso_state, monkeypatch):
    """非额度失败仍走 needs_leader 跳过（不误伤既有语义）。"""
    seed_backlog(
        iso_state,
        [
            {"id": "x1", "status": "pending", "summary": "s1"},
            {"id": "x2", "status": "pending", "summary": "s2"},
        ],
    )

    def fake_dispatch(member, **kw):
        if kw.get("slug") == "x1":
            return DispatchOutcome(code=2, kind="blocked", body="", error="工作区脏")
        return DispatchOutcome(code=0, kind="result", body="done commit ddd4444")

    monkeypatch.setattr(dispatch, "do_dispatch", fake_dispatch)
    summary = drain.drain_backlog(drain.new_run_log())
    assert "stopped" not in summary
    by_id = {i["id"]: i for i in read_backlog(iso_state)}
    assert by_id["x1"]["status"] == "needs_leader"
    assert by_id["x2"]["status"] == "done"
