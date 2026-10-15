"""mid-run 工具回合中断（stop_reason=tool_use / stop_sequence）的留档与重试。

backlog 20261016-042500-devops（组长追查 rv 二连 run_failed 定案）：
- 留档错误正文此前只剩 ``stop_reason`` 短码（stderr 空时 ``err[-400:] or stop_reason``），
  追查须翻原始会话档——本组钉住 :func:`dispatch._enriched_error` 把「停在哪」写进留档；
- 该类失败在 ``!broken`` 时本就走既有首跑重试通道——本组把它钉成回归（重试一次、二次才 failed），
  并钉住重试提示对症（分批读取）。
"""

from __future__ import annotations

import json

from pysci.skills.orchestration.tools import dispatch, transcript
from pysci.skills.orchestration.tools.runner import Envelope

from .conftest import envelope, fake_headless, set_models

A = "seat-a"
B = "seat-b"
OK_BODY = "<result>完工，commit abc1234 已合并。</result>"


def _env(stop: str = "tool_use", turns: int = 7) -> Envelope:
    return Envelope(
        result="", num_turns=turns, is_error=True, stop_reason=stop, session_id="s1"
    )


def _last_ledger_row(state) -> dict:
    lines = (state / "ledger.jsonl").read_text(encoding="utf-8").strip().splitlines()
    return json.loads(lines[-1])


# ---------------------------------------------------------------------------
# ① 判据与留档正文
# ---------------------------------------------------------------------------
def test_is_midrun_glitch_requires_turns_and_the_stop_family():
    assert dispatch._is_midrun_glitch(_env("tool_use", 7))
    assert dispatch._is_midrun_glitch(_env("stop_sequence", 131))
    assert not dispatch._is_midrun_glitch(_env("tool_use", 0)), "零回合＝席位没起跑"
    assert not dispatch._is_midrun_glitch(_env("end_turn", 7))


def test_enriched_error_names_context_not_just_the_short_code(monkeypatch):
    """留档正文含 stop_reason / num_turns / class / 重试与首停，且不再是光秃秃的短码。"""
    monkeypatch.setattr(
        dispatch, "_last_tool_calls", lambda _sid: "Read(ok)；Bash(err)"
    )
    txt = dispatch._enriched_error(
        _env("tool_use", 7), "", "", retried=True, first_stop="tool_use"
    )
    assert txt.strip() != "tool_use", "短码孤零零＝本单要消灭的形态"
    for marker in (
        "stop_reason=tool_use",
        "num_turns=7",
        "class=model_glitch",
        "retried=1(first=tool_use)",
        "末次工具：Read(ok)；Bash(err)",
    ):
        assert marker in txt, f"缺 {marker}：{txt}"


def test_enriched_error_prefers_real_text_over_fallbacks():
    """原文优先：stderr 非空就用它；空则退 envelope result；再退 stdout 杂讯尾。"""
    assert "Error executing tool unknown" in dispatch._enriched_error(
        _env(), "Error executing tool unknown\n", "", retried=False
    )
    env = Envelope(
        result="boom-from-result", num_turns=3, is_error=True, stop_reason="tool_use"
    )
    assert "boom-from-result" in dispatch._enriched_error(env, "", "", retried=False)
    assert dispatch._stdout_scene('noise line\n{"result": "x"}') == "noise line"


# ---------------------------------------------------------------------------
# ② 会话档尾部的末次工具调用形态
# ---------------------------------------------------------------------------
def _mk_session(root, sid: str, records: list[dict]) -> None:
    proj = root / "proj-key"
    proj.mkdir(parents=True, exist_ok=True)
    (proj / f"{sid}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )


def test_tail_tool_calls_pairs_results_with_calls(tmp_path):
    _mk_session(
        tmp_path,
        "sid0001",
        [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "tool_use", "id": "r1", "name": "Read", "input": {}}
                    ]
                },
            },
            {
                "type": "user",
                "message": {
                    "content": [
                        {"type": "tool_result", "tool_use_id": "r1", "content": "ok"}
                    ]
                },
            },
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "tool_use", "id": "b1", "name": "Bash", "input": {}}
                    ]
                },
            },
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "b1",
                            "content": "x",
                            "is_error": True,
                        }
                    ]
                },
            },
        ],
    )
    got = transcript.tail_tool_calls("sid0001", root=tmp_path)
    assert got == "Read(ok)；Bash(err)", got


def test_tail_tool_calls_is_fail_open(tmp_path):
    """找不到会话 / 未配对 → 空串或 ``?``，绝不抛（留档增强不得拖崩失败路径）。"""
    assert transcript.tail_tool_calls("nope", root=tmp_path) == ""
    _mk_session(
        tmp_path,
        "sid0002",
        [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "tool_use", "id": "z9", "name": "Grep", "input": {}}
                    ]
                },
            }
        ],
    )
    assert transcript.tail_tool_calls("sid0002", root=tmp_path) == "Grep(?)"


# ---------------------------------------------------------------------------
# ③ 端到端：重试一次；二次失败才落 run_failed（且带上下文）
# ---------------------------------------------------------------------------
def _glitch(_n, _cmd):
    return envelope("", is_error=True, stop="tool_use", rc=1)


def test_midrun_glitch_retries_once_then_recovers(iso_state, iso_dispatch, monkeypatch):
    set_models(iso_dispatch, flash=[A, B])

    def responder(n, cmd):
        if n == 1:
            return _glitch(n, cmd)
        return envelope(OK_BODY, is_error=False, stop="end_turn", rc=0)

    calls = fake_headless(monkeypatch, responder)
    o = dispatch.do_dispatch("quotamember", text="出图", quiet=True, no_checks=True)

    assert o.kind == "result" and o.code == 0, "重试跑通＝正常交付"
    assert len(calls) == 2, "首跑 glitch + 一次重试"
    assert all(cmd[cmd.index("-m") + 1] == A for cmd in calls if "-m" in cmd), (
        "复用首跑重试通道：停在原席"
    )


def test_midrun_glitch_double_failure_lands_run_failed_with_context(
    iso_state, iso_dispatch, monkeypatch
):
    set_models(iso_dispatch, flash=[A, B])
    calls = fake_headless(monkeypatch, _glitch)
    o = dispatch.do_dispatch("quotamember", text="出图", quiet=True)

    assert o.kind == "run_failed", "二次失败才落 failed"
    assert len(calls) == 2, "只重试一次，不无限重跳"
    row = _last_ledger_row(iso_state)
    err = row["error"]
    assert "stop_reason=tool_use" in err and "retried=1(first=tool_use)" in err, err
    assert f"num_turns={row['num_turns']}" in err, err
