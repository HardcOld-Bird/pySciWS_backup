"""inspector 巡检引擎的行为面回归（REFACTOR v1.0 §7/§5.2；E5a，backlog 20261012-120003-leader）。

在位性（脚手架/注册表/CLI 入口/文档）由 ``test_inspector_pod.py`` 量；本文件量**闸门**：
§7 的四类成本控制（全局开关、幂等拉起、单次激活预算到顶自停、让路）与 §5.2 的分流
（机械证据类免判决、判断类必经 J、迟滞暂停不晋升、无 override 通道）。

这些断言之所以值得单独一层：巡检引擎是**唯一会在组长不按键时自己花资源**的组件。
闸门写错的后果不是测试变红，而是某个夜里额度被自主巡逻烧光、或发现静静丢失无人察觉。

夹具：``iso_engine`` = ``iso_inspect``（state 全面隔离 + 让路判定清空）之上再把
「参数」与「bencher 队列长度」两个外部读数钉成确定值——否则用例会随开发者机器上的真
registry 与真队列而变绿/变红。
"""

from __future__ import annotations

import json
import os
import random
from argparse import Namespace
from datetime import datetime, timedelta

import pytest

from pysci.skills.orchestration.tools import (
    benchd,
    drain,
    inspectd,
    ledger,
    orch,
    registry,
    verdict,
    workflow,
    workq,
)
from pysci.skills.orchestration.tools.jury import Run as JuryRun

from .conftest import read_backlog, seed_backlog, seed_suggestion, set_inspect_busy

DEFAULT_INSPECTOR = registry.DEFAULT_INSPECTOR


@pytest.fixture
def iso_engine(iso_inspect, monkeypatch):
    """隔离面 + 参数出厂化 + bencher 队列指向 tmp（迟滞读数确定化）+ 派生全桩。

    ``_spawn_detached`` 两处都桩：inspector 自己的拉起，以及晋升尾部对 devops 的唤醒
    ——后者不桩就是每个此类用例真起一个后台 drain 进程。
    """
    monkeypatch.setattr(
        inspectd, "params", lambda: {**DEFAULT_INSPECTOR, "hysteresis_valid": True}
    )
    bench_q = iso_inspect / "bencher" / "queue.json"
    bench_q.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(benchd, "QUEUE_PATH", bench_q)
    monkeypatch.setattr(inspectd, "_spawn_detached", lambda argv, **kw: None)
    monkeypatch.setattr(drain, "_spawn_detached", lambda argv, **kw: None)
    return iso_inspect


def use_params(monkeypatch, **over):
    """把引擎生效参数钉成出厂值 + 覆盖项。"""
    par = {**DEFAULT_INSPECTOR, "hysteresis_valid": True, **over}
    monkeypatch.setattr(inspectd, "params", lambda: par)
    return par


def seed_mechanical(**over):
    """登记一条合规机械发现（证据三件套 + 判据齐备）。"""
    kw = dict(
        cls="mechanical",
        title="doctor 报 pod 部署副本漂移",
        symptom="bench pod 副本哈希与真本不符",
        repro="pysci-dev sync --check",
        evidence="outbox: drift 1",
        expect="drift 0",
        kind="defect",
        criteria="- `pysci-dev sync --check` 输出 drift 0",
        stamp="20261012-170000",
    )
    kw.update(over)
    return inspectd.append_finding(**kw)


def seed_judgment(title="同一词表两处实现（反冗余）", stamp="20261012-170100", **over):
    kw = dict(
        cls="judgment",
        title=title,
        criteria="- 词表只剩一处定义，其余读同一源",
        kind="usability",
        stamp=stamp,
    )
    kw.update(over)
    return inspectd.append_finding(**kw)


def journal_rows(state):
    p = state / "preference-journal.jsonl"
    if not p.exists():
        return []
    return [
        json.loads(ln)
        for ln in p.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]


def backlog_rows(state):
    """未发生过晋升时台账文件本就不存在——「没立案」与「立了空案」在断言里等价。"""
    p = state / "backlog.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))["items"]


def _write_done(state, *, completed: datetime) -> None:
    """落一份 ``.done`` 摘要（冷却窗判据的唯一输入）。"""
    runs = state / "inspector-runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "20261012-100000.log").write_text("", encoding="utf-8")
    (runs / "20261012-100000.done").write_text(
        json.dumps(
            {"started": "", "completed": completed.isoformat(), "hops": 1, "items": []}
        ),
        encoding="utf-8",
    )


def _spawns(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(
        inspectd, "_spawn_detached", lambda argv, **kw: calls.append(argv)
    )
    return calls


def activate(*, dry=True, rng_seed=7):
    log = inspectd.new_run_log(inspectd.INSPECT_PATHS)
    return inspectd.inspect_drain(log, dry=dry, rng=random.Random(rng_seed)), log


def idle_lock(monkeypatch):
    monkeypatch.setattr(inspectd, "lock_status", lambda *a, **k: {"state": "idle"})


# ---------------------------------------------------------------------------
# ① 全局开关（§7：默认开启，关闭即拉起 no-op）
# ---------------------------------------------------------------------------
def test_switch_absent_means_on_and_off_blocks_wake(iso_engine):
    assert inspectd.enabled() is True, "§7 默认开启：文件缺席必须判 on"
    assert inspectd.set_enabled(False) == "false"
    assert inspectd.enabled() is False
    spawned, msg = inspectd.wake_inspector()
    assert spawned is False and "开关" in msg
    inspectd.set_enabled(True)
    assert inspectd.enabled() is True


def test_cmd_off_also_requests_cooperative_stop(iso_engine, capsys):
    """关闭不只「不再激活」：在跑的 worker 要在条目间隙自停，否则关一次等一轮。

    用真锁而不是桩 lock_status：``request_stop`` 自己会再读一次锁（空闲则不写信号，
    免残留信号危及下一个 worker），桩住 inspectd 那一名只会让两处读数分叉——测出来的
    是这个分叉，不是关闭语义。
    """
    assert inspectd.acquire_lock(
        "state/inspector-runs/fixture.log", inspectd.INSPECT_PATHS
    )
    try:
        assert inspectd.cmd_inspector_off(Namespace()) == 0
        out = capsys.readouterr().out
        assert inspectd.enabled() is False and "协作停止" in out
        rec = inspectd.cancel_status(inspectd.INSPECT_PATHS)
        assert rec is not None and rec["pid"] == os.getpid()
    finally:
        inspectd.release_lock(paths=inspectd.INSPECT_PATHS)


def test_cmd_on_restores_and_status_shows_switch(iso_engine, capsys):
    inspectd.set_enabled(False)
    assert any("开关=关闭" in ln for ln in inspectd.status_lines()), (
        "关闭态必须在场（缺席=没配的误读）"
    )
    assert inspectd.cmd_inspector_on(Namespace()) == 0
    lines = inspectd.status_lines()
    assert not any("开关=关闭" in ln for ln in lines)
    capsys.readouterr()


# ---------------------------------------------------------------------------
# ② 幂等拉起三态与 --force 的边界
# ---------------------------------------------------------------------------
def test_wake_idempotent_across_three_blocking_states(monkeypatch, iso_engine):
    calls = _spawns(monkeypatch)

    idle_lock(monkeypatch)
    monkeypatch.setattr(inspectd, "lock_status", lambda *a, **k: {"state": "running"})
    ok, msg = inspectd.wake_inspector()
    assert not ok and "正在运行" in msg and not calls, (
        "在跑时派生第二个 worker = 互斥击穿"
    )

    idle_lock(monkeypatch)
    _write_done(iso_engine, completed=datetime.now().astimezone())
    ok, msg = inspectd.wake_inspector()
    assert not ok and "冷却" in msg and not calls, (
        "冷却窗内每敲一次 status 就重激活一轮 = 成本按敲击放大"
    )

    _write_done(iso_engine, completed=datetime.now().astimezone() - timedelta(hours=2))
    ok, msg = inspectd.wake_inspector()
    assert ok and len(calls) == 1 and "_inspect-worker" in calls[0]


def test_force_skips_cooldown_but_not_switch_nor_lock(monkeypatch, iso_engine, capsys):
    """``--force`` 越冷却、**不越开关也不越锁**；越不过时 rc=2（静默 rc=0 即假成功）。"""
    calls = _spawns(monkeypatch)
    idle_lock(monkeypatch)
    _write_done(iso_engine, completed=datetime.now().astimezone())

    assert inspectd.cmd_inspector_up(Namespace(dry=True, force=True)) == 0
    assert len(calls) == 1 and "--force" in capsys.readouterr().out

    inspectd.set_enabled(False)
    assert inspectd.cmd_inspector_up(Namespace(dry=True, force=True)) == 2
    assert len(calls) == 1, "开关是用户的成本裁决，--force 不许替用户打开"

    inspectd.set_enabled(True)
    monkeypatch.setattr(inspectd, "lock_status", lambda *a, **k: {"state": "running"})
    assert inspectd.cmd_inspector_up(Namespace(dry=True, force=True)) == 2
    assert len(calls) == 1, "锁在跑时 --force 不起第二个 worker"


def test_embedded_wake_on_status_and_approve(monkeypatch, iso_engine):
    """拉起内嵌在组长常用命令里（§7「生命周期由代码保证」，不是「组长记得开」）。"""
    hits: list[int] = []
    monkeypatch.setattr(
        inspectd, "ensure_inspector", lambda **kw: hits.append(1) or (False, "stub")
    )
    orch.cmd_status(Namespace())
    assert hits, "orch status 未内嵌拉起"

    sid = seed_suggestion(iso_engine)
    monkeypatch.setattr(verdict, "validate", lambda *a, **k: ("override", ""))
    workflow.cmd_approve(
        Namespace(
            suggestion_id=sid,
            note="采纳",
            no_wake=True,
            criteria="",
            bench="",
            jury=False,
            override="测试",
        )
    )
    assert len(hits) >= 2, "orch approve 未内嵌拉起"


def test_embedded_wake_is_fail_soft(monkeypatch, iso_engine):
    """内嵌拉起是组长命令的副作用：设施抖动不许让 ``orch status`` 失败，但要留痕。"""

    def boom(**kw):
        raise RuntimeError("锁文件不可读")

    monkeypatch.setattr(inspectd, "wake_inspector", boom)
    spawned, msg = inspectd.ensure_inspector()
    assert spawned is False and "不影响本命令" in msg and "RuntimeError" in msg


# ---------------------------------------------------------------------------
# ③ 单次激活的预算与自停原因（§7 预算硬上限）
# ---------------------------------------------------------------------------
def test_hops_budget_stops_activation(monkeypatch, iso_engine):
    use_params(monkeypatch, max_hops=2)
    summary, log = activate()
    assert summary["stopped"] == "budget_hops" and summary["hops"] == 2, (
        "到顶即停，不许超预算"
    )
    done = json.loads(log.with_suffix(".done").read_text(encoding="utf-8"))
    assert done["stopped"] == "budget_hops" and done["completed"]
    assert inspectd.lock_status(inspectd.INSPECT_PATHS)["state"] != "running", (
        "收尾必须放锁"
    )


def test_wallclock_budget_stops_activation(monkeypatch, iso_engine):
    use_params(monkeypatch, wallclock_s=0)
    summary, _ = activate()
    assert summary["stopped"] == "budget_wallclock"


def test_switch_off_mid_activation_stops_at_next_gap(monkeypatch, iso_engine):
    """条目间隙复查开关：关闭后不再派第二跳（不等预算耗尽）。"""
    use_params(monkeypatch, max_hops=5)
    state = {"n": 0}
    real = inspectd.enabled

    def flip():
        state["n"] += 1
        return real() if state["n"] < 2 else False

    monkeypatch.setattr(inspectd, "enabled", flip)
    summary, _ = activate()
    assert summary["stopped"] == "switch_off" and summary["hops"] == 1


def test_yield_does_not_dispatch_or_consume(monkeypatch, iso_engine):
    """让路：生产侧忙 → 不派跳、不消费队列项，本轮以 ``yield`` 收束。"""
    use_params(monkeypatch, max_hops=3)
    set_inspect_busy(monkeypatch, ["devops-drain"])
    before = workq.load(inspectd.QUEUE_PATH)
    summary, _ = activate()
    assert summary["stopped"] == "yield" and summary["hops"] == 0
    assert summary["items"] == [] and workq.load(inspectd.QUEUE_PATH) == before


def test_plan_run_and_bencher_busy_also_yield(monkeypatch, iso_engine):
    """让路判据三面之一（bencher worker 锁）成立时同样只读。"""
    use_params(monkeypatch, max_hops=2)
    set_inspect_busy(monkeypatch, ["bencher-worker"])
    summary, _ = activate()
    assert summary["stopped"] == "yield"


def test_queue_empty_does_not_kill_the_worker(monkeypatch, iso_engine):
    """队列空**不自毁**：抽一项菜单继续做（与 bencher worker 的关键差别）。"""
    use_params(monkeypatch, max_hops=2)
    summary, _ = activate()
    assert {it["kind"] for it in summary["items"]} == {"patrol"} and summary[
        "hops"
    ] == 2
    queued = workq.load(inspectd.QUEUE_PATH)
    assert queued and all(i["target"] in inspectd.PATROL_BY_ID for i in queued), (
        "抽出的菜单项必须落到队列（空 target 项=组员拿到无素材任务书的死信）"
    )


def test_quota_exhaustion_ends_the_activation(monkeypatch, iso_engine):
    """系统性故障（额度耗尽）立即收束，不在同渠道连烧第二跳。"""
    use_params(monkeypatch, max_hops=3)

    def fake_run(item, run_log, *, dry=False):
        return {"status": "failed", "hop": True, "systemic": "quota_exhausted"}

    monkeypatch.setattr(inspectd, "run_item", fake_run)
    summary, _ = activate(dry=False)
    assert summary["stopped"] == "quota_exhausted" and "quota" in summary["error"]


def test_channel_error_ends_the_activation(monkeypatch, iso_engine):
    """渠道级故障与额度同族处置：巡检激活同样全局停止，不在坏渠道上连烧第二跳。

    ``systemic`` 是 kind 串而非布尔（backlog 20261012-194500-leader）——三个 worker 面
    共用一支清单，停止动作相同、``.done`` 里必须说得出是哪一类故障。
    """
    use_params(monkeypatch, max_hops=3)

    def fake_run(item, run_log, *, dry=False):
        return {"status": "failed", "hop": True, "systemic": "channel_error"}

    monkeypatch.setattr(inspectd, "run_item", fake_run)
    summary, _ = activate(dry=False)
    assert summary["stopped"] == "channel_error"
    assert "渠道级故障" in summary["error"]
    assert "quota" not in summary["error"], "两类故障的读数不得混成一副面孔"


def test_worker_command_prints_nothing_to_stdout(monkeypatch, iso_engine, capsys):
    """派生时 stdout 被重定向进本 run 的 ``.log``——打印会从偏移 0 覆写整段运行日志。

    2026-10-12 实测：worker 收尾 ``print(summary)`` 让 run 日志只剩摘要 JSON，
    让路/闸门/派跳过程全灭（mock 测不出：``_spawn_detached`` 被桩时 stdout 不参与）。
    """
    use_params(monkeypatch, max_hops=1)
    log = inspectd.new_run_log(inspectd.INSPECT_PATHS)
    assert inspectd.cmd_inspect_worker(Namespace(run_log=str(log), dry=True)) == 0
    assert capsys.readouterr().out == "", "stdout 即运行日志：一个字都不许打印"
    text = log.read_text(encoding="utf-8")
    assert "worker 启动" in text and "worker 结束" in text


# ---------------------------------------------------------------------------
# ④ 双队列迟滞（§5.2：超过才停，回落至 resume 才恢复）
# ---------------------------------------------------------------------------
def test_hysteresis_holds_between_thresholds(iso_engine):
    st = inspectd.throttle_state()
    assert st["produce"] == "on"

    seed_backlog(iso_engine, [{"id": f"s{i}", "status": "pending"} for i in range(13)])
    assert inspectd.throttle_state()["produce"] == "paused"

    seed_backlog(iso_engine, [{"id": f"s{i}", "status": "pending"} for i in range(10)])
    assert inspectd.throttle_state()["produce"] == "paused", (
        "回落到 pause 与 resume 之间仍停：立即恢复就是每轮翻转的开关，不叫迟滞"
    )

    seed_backlog(iso_engine, [{"id": f"s{i}", "status": "pending"} for i in range(8)])
    st = inspectd.throttle_state()
    assert st["produce"] == "on" and st["depths"]["backlog_pending"] == 8


def test_bench_depth_alone_pauses_production(iso_engine):
    workq.save(
        benchd.QUEUE_PATH, [{"id": f"b{i}", "status": "pending"} for i in range(7)]
    )
    st = inspectd.throttle_state()
    assert st["produce"] == "paused" and "bench" in st["reason"]


def test_paused_gate_holds_promotion_without_losing_finding(monkeypatch, iso_engine):
    """暂停期间不晋升，条目留 pending（不是 rejected）——恢复后要投得出去。"""
    use_params(monkeypatch, pause_backlog=12, resume_backlog=8)
    monkeypatch.setattr(
        inspectd, "depths", lambda **kw: {"backlog_pending": 20, "bench_pending": 0}
    )
    rec = seed_mechanical()
    out = inspectd.promote_finding(rec, channel="mechanical")
    assert out["action"] == "held" and "积压" in out["reason"]
    assert inspectd.finding_by_id(rec["id"])["status"] == "pending"
    assert backlog_rows(iso_engine) == []


def test_illegal_thresholds_fall_back_wholesale_and_are_flagged(
    iso_engine, iso_registry, monkeypatch
):
    """``resume >= pause`` 的半套配置整体回落出厂值，并在 status 面标注被忽略。"""
    reg_file, _runtime, _backup = iso_registry
    reg_file.write_text(
        json.dumps(
            {
                "version": 1,
                "members": {},
                "inspector": {
                    "max_hops": 9,
                    "pause_backlog": 12,
                    "resume_backlog": 12,
                    "pause_bench": 6,
                    "resume_bench": 4,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(inspectd, "params", lambda: registry.Registry.load().inspector)
    par = inspectd.params()
    assert par["hysteresis_valid"] is False
    assert par["pause_backlog"] == DEFAULT_INSPECTOR["pause_backlog"]
    assert par["resume_backlog"] == DEFAULT_INSPECTOR["resume_backlog"]
    assert par["max_hops"] == 9, "非迟滞键不受牵连（逐键覆盖是设计）"
    assert any("非法" in ln for ln in inspectd.status_lines()), "被忽略的配置必须可见"


# ---------------------------------------------------------------------------
# ⑤ 发现分流（§5.2——巡检产出与管道之间唯一的接口）
# ---------------------------------------------------------------------------
def test_mechanical_promotes_without_a_ruling(iso_engine):
    rec = seed_mechanical()
    out = inspectd.promote_finding(rec, channel="mechanical", wake=False)
    assert out["action"] == "promoted"
    entry = next(it for it in read_backlog(iso_engine) if it["id"] == rec["id"])
    assert entry["source"] == "inspector" and entry["actor"] == "inspector"
    assert entry["member"] == "inspector" and entry["class"] == "mechanical"
    assert (
        entry["kind"] == "defect"
        and entry["success_criteria"] == rec["success_criteria"]
    )

    path = inspectd.EVIDENCE_DIR / f"{rec['id']}.md"
    assert path.is_file() and out["evidence_file"].endswith(f"{rec['id']}.md")
    body = path.read_text(encoding="utf-8")
    for token in (rec["repro"], rec["evidence"], rec["success_criteria"]):
        assert token in body, "证据正文要能独立复核（复现+实际输出+判据）"

    rows = journal_rows(iso_engine)
    assert rows[-1]["channel"] == "mechanical" and rows[-1]["source"] == "inspector"
    assert all(r.get("channel") != "override" for r in rows), (
        "免判决不等于自裁，不占 override 率"
    )
    assert inspectd.finding_by_id(rec["id"])["status"] == "promoted"


def test_promotion_is_idempotent_per_finding(iso_engine):
    rec = seed_mechanical()
    assert (
        inspectd.promote_finding(rec, channel="mechanical", wake=False)["action"]
        == "promoted"
    )
    again = inspectd.promote_finding(
        inspectd.finding_by_id(rec["id"]), channel="mechanical", wake=False
    )
    assert again["action"] == "duplicate", "同一发现二次立案会淹掉 FIFO"
    assert len([it for it in read_backlog(iso_engine) if it["id"] == rec["id"]]) == 1


def test_finding_without_criteria_is_rejected_not_retried(iso_engine):
    """§6 总闸：缺判据判 rejected——留在 pending 就是每跳重跑同一个拒绝。"""
    rec = seed_mechanical(criteria="   ")
    out = inspectd.promote_finding(rec, channel="mechanical", wake=False)
    assert out["action"] == "rejected_no_criteria"
    stale = inspectd.finding_by_id(rec["id"])
    assert stale["status"] == "rejected" and "缺可证伪成功判据" in stale["note"]
    assert backlog_rows(iso_engine) == []


def test_mechanical_class_requires_rerunnable_evidence(iso_engine):
    with pytest.raises(ValueError, match="复现"):
        inspectd.append_finding(cls="mechanical", title="x", criteria="- 判据")
    with pytest.raises(ValueError, match="类别"):
        inspectd.append_finding(cls="aesthetic", title="x", criteria="- 判据")
    with pytest.raises(ValueError, match="kind"):
        inspectd.append_finding(
            cls="mechanical",
            title="x",
            repro="r",
            evidence="e",
            kind="optional",
            criteria="- c",
        )
    with pytest.raises(ValueError, match="title"):
        inspectd.append_finding(cls="judgment", title="  ", criteria="- c")


def test_judgment_cannot_use_the_no_ruling_channel(iso_engine):
    """判断类**没有**免判决出口：CLI 的 ``promote`` 面拒收，机械类反向也无须开庭。"""
    rec = seed_judgment()
    from pysci.skills.inspector.tools import inspect as insp_cli

    assert insp_cli.cmd_promote(Namespace(finding=rec["id"], no_wake=True)) == 2
    assert backlog_rows(iso_engine) == [], "判断类绕过 J 直接进队即 §5.2 失效"
    assert inspectd.finding_by_id(rec["id"])["status"] == "pending"

    wrong = inspectd.submit_judgment({"class": "mechanical", "id": "insp-x"})
    assert wrong["outcome"] == "wrong_class", "机械类免判决，不该去占仲裁额度"

    mrec = seed_mechanical(stamp="20261012-170500")
    assert (
        insp_cli.cmd_submit(Namespace(finding=mrec["id"], dry=True, timeout=None)) == 2
    )


def _jury(status="recorded", call="A", tally=None, message=""):
    return JuryRun(
        status=status,
        message=message,
        record={"verdict": call, "tally": tally or {"A": 3, "B": 0}},
    )


@pytest.fixture
def stub_jury(monkeypatch, iso_engine):
    """把仲裁两级（组卷/开庭）桩掉：案卷形状由 test_verdict/jury 专测，这里只验分流。"""
    monkeypatch.setattr(verdict, "assemble_case", lambda **kw: iso_engine / "case.md")
    box = {"run": _jury()}
    monkeypatch.setattr(verdict, "run_jury", lambda *a, **k: box["run"])
    return box


def test_jury_a_promotes_with_channel_recorded(stub_jury, iso_engine):
    rec = seed_judgment()
    out = inspectd.submit_judgment(inspectd.finding_by_id(rec["id"]))
    assert out["action"] == "promoted"
    entry = next(it for it in read_backlog(iso_engine) if it["id"] == rec["id"])
    assert entry["channel"] == "jury" and entry["source"] == "inspector"
    assert "A" in entry["note"], "裁决摘要（计票）要进 backlog 条目，复核者不必再翻案卷"


def test_jury_b_rejects_without_billing_an_override(stub_jury, iso_engine):
    rec = seed_judgment(stamp="20261012-170101")
    stub_jury["run"] = _jury(call="B", tally={"A": 1, "B": 2})
    out = inspectd.submit_judgment(inspectd.finding_by_id(rec["id"]))
    assert out["outcome"] == "rejected"
    assert inspectd.finding_by_id(rec["id"])["status"] == "rejected"
    rows = journal_rows(iso_engine)
    assert rows[-1]["choice"] == "reject" and rows[-1]["channel"] == "jury"
    assert all(r.get("channel") != "override" for r in rows), (
        "§5.1：判 B 不占 override 率——inspector 没有自裁额度，账面也不许记它有"
    )


def test_jury_inconclusive_escalates_to_twin(stub_jury, iso_engine):
    rec = seed_judgment(stamp="20261012-170102")
    stub_jury["run"] = _jury(call=None, message="双序矛盾")
    out = inspectd.submit_judgment(inspectd.finding_by_id(rec["id"]))
    assert out["outcome"] == "escalated"
    note = inspectd.finding_by_id(rec["id"])["note"]
    assert "twin" in note, "模棱两可交终裁，不许引擎硬判（§5.5）"


def test_jury_unreachable_queues_for_next_round(stub_jury, iso_engine):
    rec = seed_judgment(stamp="20261012-170103")
    stub_jury["run"] = _jury(status="queued", message="渠道不可达")
    out = inspectd.submit_judgment(inspectd.finding_by_id(rec["id"]))
    assert out["outcome"] == "queued"
    assert inspectd.finding_by_id(rec["id"])["status"] == "queued", (
        "降级不是失败，下轮重投"
    )


def test_activation_auto_prompts_mechanical_backlog(monkeypatch, iso_engine):
    """worker 每轮自动扫 pending 机械类并晋升——这一段是「组长介入=0」的实体。"""
    use_params(monkeypatch, max_hops=1)
    rec = seed_mechanical()
    summary, _ = activate()
    assert summary["stopped"] == "budget_hops"
    assert rec["id"] in {it["id"] for it in read_backlog(iso_engine)}


def test_readonly_checks_register_and_promote_engine_findings(monkeypatch, iso_engine):
    """引擎侧读侧检查（成本曲线）发现的机械证据类，同样走自动晋升——且让路时也照跑。"""
    use_params(monkeypatch, max_hops=1)

    def fake_run(item, run_log, *, dry=False):
        return {"status": "done", "hop": True}

    monkeypatch.setattr(inspectd, "run_item", fake_run)
    rows = [
        {
            "ts": f"2026-10-12T{h:02d}:00:00+08:00",
            "member": "figure",
            "session_id": "s",
            "est_tokens": 5000 if i >= 7 else 100,
        }
        for i, h in enumerate(range(10))
    ]
    ledger.LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    ledger.LEDGER_PATH.write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    assert inspectd.check_cost_curve(), "读数异常应成案（尾部超均值 3 倍）"
    summary, _ = activate(dry=False)
    found = inspectd.findings()
    assert found and found[0]["source"] == "inspectd"
    assert any(f["id"] in summary.get("findings", []) for f in found)
    assert found[0]["status"] == "promoted", "引擎侧机械证据类免判决"

    set_inspect_busy(monkeypatch, ["devops-drain"])
    n_before = len(inspectd.findings())
    inspectd.run_readonly_checks()
    assert len(inspectd.findings()) == n_before, (
        "重复成案会每跳堆一条——同指纹不重登（已 promoted 的条目不再是 pending）"
    )


def test_assignment_items_use_given_brief_without_menu(monkeypatch, iso_engine):
    """组长指派项吃自带任务书；``orch inspector-patrol`` 不给 target 时随机补一项。"""
    item = inspectd.enqueue(
        "assignment", brief="核查：bencher 队列是否 FIFO", summary="核查"
    )
    assert item["kind"] == "assignment" and item["target"] == ""
    log = inspectd.new_run_log(inspectd.INSPECT_PATHS)
    result = inspectd.run_item(item, log, dry=True)
    assert result["status"] == "done" and result["hop"] is True

    with pytest.raises(ValueError, match="队列项类型"):
        inspectd.enqueue("patrol-self", brief="x")

    rc = inspectd.cmd_inspector_patrol(
        Namespace(target="no-such-item", summary="", start=False)
    )
    assert rc == 2
    rc = inspectd.cmd_inspector_patrol(Namespace(target="", summary="", start=False))
    assert rc == 0
    latest = workq.load(inspectd.QUEUE_PATH)[-1]
    assert latest["target"] in inspectd.PATROL_BY_ID, (
        "空 target 的巡检项是死信（组员无素材可做）"
    )
