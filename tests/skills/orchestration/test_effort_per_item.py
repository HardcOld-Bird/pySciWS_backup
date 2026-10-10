"""推理强度按项标注机制（backlog 20261010-orch-effort-per-item）。

用户裁决 2026-10-10：双档默认推理强度保持「中」，**不做全局 high**——组长对推理密集
项（并发/状态机/跨模块语义）按项标 high，机械项留默认；落地后台账自然 A/B。

覆盖四层来源与一条透传链：
``--effort`` 旗标 > 计划环节 ``steps[].effort`` / backlog 条目 ``effort``（approve 入队
时标，drain 消化透传）> registry 成员 ``effort`` > 不传（跟随用户级默认）；
最终由 :func:`runner.build_command` 透传原生 ``--reasoning-effort``，实际档位记入台账。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from pysci.skills.orchestration.tools import (
    dispatch,
    drain,
    ledger,
    orch,
    plans,
    runner,
    workflow,
)
from pysci.skills.orchestration.tools.dispatch import DispatchOutcome
from pysci.skills.orchestration.tools.registry import Member

from .conftest import (
    QUOTA_TEXT,
    envelope,
    fake_headless,
    read_backlog,
    seed_backlog,
    seed_suggestion,
    set_models,
)

OK = "<result>完成，commit aaa1111。</result>"


def cmd_effort(cmd: list[str]) -> str:
    """从 argv 提取 ``--reasoning-effort`` 后的档位（未透传 → ""）。"""
    return (
        cmd[cmd.index("--reasoning-effort") + 1] if "--reasoning-effort" in cmd else ""
    )


def set_member(state, **fields) -> None:
    """改写隔离 registry.json 里 quotamember 的字段（测成员级 effort 默认值）。"""
    reg_file = state / "registry.json"
    data = json.loads(reg_file.read_text(encoding="utf-8"))
    data["members"]["quotamember"].update(fields)
    reg_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# normalize_effort：exe 枚举 + 别名，非法硬失败
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,want",
    [
        ("high", "high"),
        ("  HIGH ", "high"),
        ("none", "none"),
        ("xhigh", "xhigh"),
        ("max", "max"),
        ("ultracode", "ultracode"),
        ("auto", "auto"),
        ("off", "none"),  # exe 别名
        ("disabled", "none"),
        (None, None),
        ("", None),
        ("   ", None),
    ],
)
def test_normalize_effort_ok(raw, want):
    """档位表以实测为准：逐个跑 ``qoderclicn --reasoning-effort <v> --list-models``
    （2026-10-10：上列值全 rc=0，大小写不敏感）。"""
    assert runner.normalize_effort(raw) == want


@pytest.mark.parametrize("bad", ["min", "ultra", "higer", "medium-high"])
def test_normalize_effort_rejects_unknown(bad):
    """非法值 exe 会**启动即失败**（实测 rc=1 ``Invalid reasoning effort: bogus``）；
    在本层先拦住才能给可读提示、不烧跳、不写无意义台账行——尤其 registry/条目来源的
    值不经过 argparse choices。"""
    with pytest.raises(ValueError) as exc:
        runner.normalize_effort(bad)
    assert "无效 effort" in str(exc.value)
    assert "xhigh" in str(exc.value)  # 报错列出可用档位


# ---------------------------------------------------------------------------
# build_command：透传原生旗标
# ---------------------------------------------------------------------------
def _member(tmp_path) -> Member:
    return Member(
        member_id="m",
        pod=tmp_path / "pod",
        model_tier="flash",
        max_turns=5,
        timeout_s=60,
        mcp_config=None,
    )


def test_build_command_no_effort_flag_by_default(tmp_path):
    """默认不传旗标 = 跟随用户级默认（中）——绝不主动写 medium。"""
    cmd = runner.build_command(
        _member(tmp_path),
        "p",
        session_id="s",
        resume=False,
        model="",
        exe=tmp_path / "exe",
    )
    assert "--reasoning-effort" not in cmd


def test_build_command_passes_effort(tmp_path):
    cmd = runner.build_command(
        _member(tmp_path),
        "p",
        session_id="s",
        resume=False,
        model="",
        effort="high",
        exe=tmp_path / "exe",
    )
    assert cmd_effort(cmd) == "high"


# ---------------------------------------------------------------------------
# do_dispatch：来源优先级 + 台账记录 + 非法不烧跳
# ---------------------------------------------------------------------------
def test_dispatch_no_effort_when_unmarked(iso_dispatch, monkeypatch):
    calls = fake_headless(
        monkeypatch, lambda n, cmd: envelope(OK, is_error=False, stop="end_turn", rc=0)
    )
    o = dispatch.do_dispatch("quotamember", text="t", quiet=True, no_checks=True)
    assert o.code == 0
    assert cmd_effort(calls[0]) == "" and o.effort == ""


def test_dispatch_uses_member_level_effort(iso_dispatch, monkeypatch):
    set_member(iso_dispatch, effort="high")
    calls = fake_headless(
        monkeypatch, lambda n, cmd: envelope(OK, is_error=False, stop="end_turn", rc=0)
    )
    o = dispatch.do_dispatch("quotamember", text="t", quiet=True, no_checks=True)
    assert o.code == 0 and cmd_effort(calls[0]) == "high" and o.effort == "high"
    row = json.loads((iso_dispatch / "ledger.jsonl").read_text(encoding="utf-8"))
    assert row["effort"] == "high", "台账记实际所用档位（A/B 数据源）"


def test_dispatch_caller_effort_overrides_member(iso_dispatch, monkeypatch):
    set_member(iso_dispatch, effort="medium")
    calls = fake_headless(
        monkeypatch, lambda n, cmd: envelope(OK, is_error=False, stop="end_turn", rc=0)
    )
    dispatch.do_dispatch(
        "quotamember", text="t", effort="high", quiet=True, no_checks=True
    )
    assert cmd_effort(calls[0]) == "high"


def test_dispatch_effort_alias_normalized(iso_dispatch, monkeypatch):
    set_member(iso_dispatch, effort="OFF")
    calls = fake_headless(
        monkeypatch, lambda n, cmd: envelope(OK, is_error=False, stop="end_turn", rc=0)
    )
    dispatch.do_dispatch("quotamember", text="t", quiet=True, no_checks=True)
    assert cmd_effort(calls[0]) == "none", "别名 off→none 后透传（exe 认 none）"


def test_dispatch_invalid_effort_fails_without_burning_hop(iso_dispatch, monkeypatch):
    calls = fake_headless(
        monkeypatch, lambda n, cmd: envelope(OK, is_error=False, stop="end_turn", rc=0)
    )
    o = dispatch.do_dispatch("quotamember", text="t", effort="ultra", quiet=True)
    assert o.code == 2 and o.kind == "run_failed"
    assert "无效 effort" in o.error
    assert calls == [], "非法标注不派发（不烧跳、不污染台账）"
    assert (iso_dispatch / "ledger.jsonl").exists() is False


def test_dispatch_invalid_member_effort_hard_fails(iso_dispatch, monkeypatch):
    set_member(iso_dispatch, effort="bogus")
    calls = fake_headless(
        monkeypatch, lambda n, cmd: envelope(OK, is_error=False, stop="end_turn", rc=0)
    )
    o = dispatch.do_dispatch("quotamember", text="t", quiet=True)
    assert o.code == 2 and calls == []


def test_dispatch_effort_survives_retry(iso_dispatch, monkeypatch):
    """首跑普通失败 → 重试跳沿用同档位（同会话续作，强度不该变）。"""
    set_member(iso_dispatch, effort="high")
    calls = fake_headless(
        monkeypatch,
        lambda n, cmd: (
            envelope("崩了")
            if n == 1
            else envelope(OK, is_error=False, stop="end_turn", rc=0)
        ),
    )
    o = dispatch.do_dispatch("quotamember", text="t", quiet=True, no_checks=True)
    assert o.code == 0 and len(calls) == 2
    assert [cmd_effort(c) for c in calls] == ["high", "high"]


def test_dispatch_effort_survives_quota_fallback(iso_dispatch, monkeypatch):
    """额度耗尽换备用渠道重试：换的是**渠道**，档位标注保持不变。"""
    set_models(iso_dispatch, max="", flash="builtin-flash", flash_fallback="byok-fb")
    set_member(iso_dispatch, effort="high")
    calls = fake_headless(
        monkeypatch,
        lambda n, cmd: (
            envelope(QUOTA_TEXT)
            if n == 1
            else envelope(OK, is_error=False, stop="end_turn", rc=0)
        ),
    )
    o = dispatch.do_dispatch("quotamember", text="t", quiet=True, no_checks=True)
    assert o.code == 0 and len(calls) == 2
    assert [cmd_effort(c) for c in calls] == ["high", "high"]


# ---------------------------------------------------------------------------
# 台账：新增 effort 字段向后兼容
# ---------------------------------------------------------------------------
def test_ledger_effort_field_backward_compatible(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "l.jsonl")
    ledger.append(ledger.LedgerEntry(ts="", member="m", session_id="s", effort="high"))
    with (tmp_path / "l.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(
            '{"ts": "2026-01-01T00:00:00+08:00", "member": "old", "session_id": "o"}\n'
        )
    rows = ledger.read_all()
    assert rows[0].effort == "high"
    assert rows[1].effort == "", "旧行无该字段 → 默认空（不炸读取）"


# ---------------------------------------------------------------------------
# backlog：approve 入队标注 + 组包只收同档位
# ---------------------------------------------------------------------------
def test_take_batch_groups_only_same_effort(iso_state):
    seed_backlog(
        iso_state,
        [
            {"id": "a", "status": "pending", "member": "m", "effort": "high"},
            {"id": "b", "status": "pending", "member": "m"},  # 默认档，不同组
            {"id": "c", "status": "pending", "member": "m", "effort": " HIGH "},
        ],
    )
    seed, group = workflow.backlog_take_batch(4)
    assert seed["id"] == "a"
    assert [g["id"] for g in group] == ["c"], "一跳只有一个跳级档位：混档不组包"


def test_take_batch_unmarked_items_group_as_before(iso_state):
    """全无标注 → 组包行为不变（向后兼容旧 backlog 数据）。"""
    seed_backlog(
        iso_state,
        [{"id": c, "status": "pending", "member": "m"} for c in "abc"],
    )
    seed, group = workflow.backlog_take_batch(4)
    assert seed["id"] == "a" and [g["id"] for g in group] == ["b", "c"]


def test_effort_key_fail_soft_on_invalid():
    """非法标注在组包阶段不抛（否则会拖垮整个 drain 循环）：原样作键，
    等它当种子派发时由 do_dispatch 硬失败并报出原因。"""
    assert workflow._effort_key({"effort": "bogus"}) == "bogus"


def test_approve_effort_lands_on_backlog_item(iso_state):
    sid = seed_suggestion(iso_state)
    rc = workflow.cmd_approve(
        argparse.Namespace(suggestion_id=sid, note="采纳", no_wake=True, effort="high")
    )
    assert rc == 0
    assert read_backlog(iso_state)[0]["effort"] == "high"


def test_approve_without_effort_omits_key(iso_state):
    sid = seed_suggestion(iso_state)
    workflow.cmd_approve(
        argparse.Namespace(suggestion_id=sid, note="采纳", no_wake=True, effort=None)
    )
    assert "effort" not in read_backlog(iso_state)[0]


def test_approve_partial_namespace_still_works(iso_state):
    """程序化调用方常建 partial 参数（无 effort 属性）→ 视为不标注。"""
    sid = seed_suggestion(iso_state)
    rc = workflow.cmd_approve(
        argparse.Namespace(suggestion_id=sid, note="n", no_wake=True)
    )
    assert rc == 0 and "effort" not in read_backlog(iso_state)[0]


# ---------------------------------------------------------------------------
# drain：种子条目的 effort 透传到该跳
# ---------------------------------------------------------------------------
def _capture_dispatch(monkeypatch, outcome):
    seen: dict = {}

    def fake(*a, **k):
        seen.update(k)
        seen["args"] = a
        return outcome

    monkeypatch.setattr(dispatch, "do_dispatch", fake)
    return seen


def test_drain_forwards_seed_effort(iso_state, monkeypatch):
    seed_backlog(
        iso_state,
        [{"id": "s1", "status": "pending", "member": "devops", "effort": "high"}],
    )
    seed, group = workflow.backlog_take_batch(4)
    seen = _capture_dispatch(
        monkeypatch,
        DispatchOutcome(
            code=0, kind="result", body="done commit aaa1111 backlog id=s1"
        ),
    )
    drain._drain_batch(seed, group, drain.new_run_log(), dry=False)
    assert seen["effort"] == "high"


def test_drain_unmarked_seed_passes_none(iso_state, monkeypatch):
    """无标注条目透传 None → do_dispatch 回落成员级/用户级默认（机制不变）。"""
    seed_backlog(iso_state, [{"id": "s1", "status": "pending", "member": "devops"}])
    seed, group = workflow.backlog_take_batch(4)
    seen = _capture_dispatch(
        monkeypatch,
        DispatchOutcome(code=0, kind="result", body="done commit aaa1111"),
    )
    drain._drain_batch(seed, group, drain.new_run_log(), dry=False)
    assert seen["effort"] is None


# ---------------------------------------------------------------------------
# plan：环节 effort 字段 + 旗标覆盖 + adhoc 透传
# ---------------------------------------------------------------------------
@pytest.fixture
def iso_plans(tmp_path, monkeypatch):
    """把计划目录隔离到 tmp，并捕获传给 do_dispatch 的参数。"""

    d = tmp_path / "plans"
    d.mkdir()
    monkeypatch.setattr(plans, "PLANS_DIR", d)
    seen: dict = {}
    monkeypatch.setattr(
        plans,
        "do_dispatch",
        lambda member, **k: (
            seen.update(k)
            or DispatchOutcome(code=0, kind="result", member=member, body=OK)
        ),
    )
    return d, seen


def _write_plan(d: Path, plan_id: str, step_extra: dict) -> None:

    step = {"id": "s1", "member": "sim", "brief": "粗描述", "dirs": []}
    step.update(step_extra)
    (d / f"{plan_id}.plan.yaml").write_text(
        yaml.safe_dump(
            {"id": plan_id, "goal": "g", "steps": [step]}, allow_unicode=True
        ),
        encoding="utf-8",
    )


def _run_args(plan_id: str, **over) -> SimpleNamespace:
    base = dict(
        id=plan_id,
        text="任务书",
        task=None,
        name=None,
        session=None,
        model_tier=None,
        effort=None,
        max_turns=None,
        dirs=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_plan_run_uses_step_effort(iso_plans):
    d, seen = iso_plans
    _write_plan(d, "p1", {"effort": "high"})
    assert plans.cmd_plan_run(_run_args("p1")) == 0
    assert seen["effort"] == "high"


def test_plan_run_flag_overrides_step(iso_plans):
    d, seen = iso_plans
    _write_plan(d, "p1", {"effort": "high"})
    plans.cmd_plan_run(_run_args("p1", effort="max"))
    assert seen["effort"] == "max"


def test_plan_run_without_effort_passes_none(iso_plans):
    d, seen = iso_plans
    _write_plan(d, "p1", {})
    plans.cmd_plan_run(_run_args("p1"))
    assert seen["effort"] is None


def test_plan_adhoc_carries_effort_to_run(iso_plans, monkeypatch):
    d, _seen = iso_plans
    captured: dict = {}

    def fake_run(args):
        captured["effort"] = args.effort
        return 0

    monkeypatch.setattr(plans, "cmd_plan_run", fake_run)
    args = SimpleNamespace(
        member="sim",
        text="单步任务",
        task=None,
        dirs=None,
        model_tier=None,
        max_turns=None,
        effort="high",
    )
    assert plans.cmd_plan_adhoc(args) == 0
    assert captured["effort"] == "high"


# ---------------------------------------------------------------------------
# CLI 门面：旗标接线（dispatch --effort / --from-backlog 取条目档 / 非法值入队即拒）
# ---------------------------------------------------------------------------
def _capture_orch_dispatch(monkeypatch, outcome=None):
    seen: dict = {}

    def fake(member, **k):
        seen.update(k)
        return outcome or DispatchOutcome(code=0, kind="result", member=member, body=OK)

    monkeypatch.setattr(orch, "do_dispatch", fake)
    return seen


def test_cli_dispatch_flag_reaches_dispatch(monkeypatch):
    seen = _capture_orch_dispatch(monkeypatch)
    rc = orch.main(["dispatch", "devops", "--text", "t", "--effort", "high"])
    assert rc == 0 and seen["effort"] == "high"


def test_cli_dispatch_rejects_unknown_level(monkeypatch):
    seen = _capture_orch_dispatch(monkeypatch)
    with pytest.raises(SystemExit):
        orch.main(["dispatch", "devops", "--text", "t", "--effort", "ultra"])
    assert seen == {}, "argparse choices 在派发前拦住"


def test_cli_from_backlog_uses_item_effort(iso_state, monkeypatch):
    seed_backlog(
        iso_state,
        [
            {
                "id": "a",
                "status": "pending",
                "member": "devops",
                "summary": "s",
                "effort": "high",
            }
        ],
    )
    seen = _capture_orch_dispatch(monkeypatch)
    assert orch.main(["dispatch", "devops", "--from-backlog"]) == 0
    assert seen["effort"] == "high"
    assert read_backlog(iso_state)[0]["status"] == "done"


def test_cli_from_backlog_flag_wins_over_item(iso_state, monkeypatch):
    seed_backlog(
        iso_state,
        [
            {
                "id": "a",
                "status": "pending",
                "member": "devops",
                "summary": "s",
                "effort": "high",
            }
        ],
    )
    seen = _capture_orch_dispatch(monkeypatch)
    orch.main(["dispatch", "devops", "--from-backlog", "--effort", "medium"])
    assert seen["effort"] == "medium"


def test_cli_plan_run_flag_parses(monkeypatch):
    captured: dict = {}

    def fake_run(args):
        captured["effort"] = args.effort
        return 0

    monkeypatch.setattr(plans, "cmd_plan_run", fake_run)
    assert orch.main(["plan", "run", "p1", "--text", "t", "--effort", "high"]) == 0
    assert captured["effort"] == "high"


def test_cli_approve_flag_parses(iso_state, monkeypatch):
    sid = seed_suggestion(iso_state)
    assert (
        orch.main(["approve", sid, "--note", "n", "--effort", "high", "--no-wake"]) == 0
    )
    assert read_backlog(iso_state)[0]["effort"] == "high"


def test_effort_choices_cover_exe_levels():
    """CLI choices 面 = exe 实测接受集（含别名），组长手不会撞上假阴性。"""
    assert set(orch.EFFORT_CHOICES) == {
        "auto",
        "none",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
        "ultracode",
        "off",
        "disabled",
    }
