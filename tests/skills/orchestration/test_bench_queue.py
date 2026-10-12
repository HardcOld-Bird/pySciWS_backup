"""E4 bencher 测量队列与非常驻 worker 的行为回归（backlog 20261012-120002-leader）。

三块机制与 ``test_drain_*.py`` 同构（锁/墓碑/``.done``/run 名同秒顺延都是被三次事故
打磨出来的），故本文件**不重测原语**，只测 bencher 这套落点上的**队列语义与生命周期**：

- 入队校验：非法 kind / ab 少一端 / capture 无案，一律当场拒绝（不入库就是证据）；
- FIFO 与孤儿回收：单 worker 互斥由锁保证，故获锁一刻仍在 ``in_progress`` 必是死 worker 遗物；
- 「择机查看」通知面（§5.1-B）：完成项进 status 待查看面直到 ``bench-seen``，且
  **唤醒器不挂 watch**——这是它与 devops drain 的唯一行为差别（§8 明写不即时打断）；
- 干跑全链：注入 → 拉起 → 消化 → 自毁 → 结果待查看（判据 3 的机械面）。

真派发与真派生一律被 ``iso_bench`` 夹具桩掉；副本 A/B 在 ``test_bench_ab.py``。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pytest

from pysci.skills.orchestration.tools import benchd, drain, orch
from pysci.skills.orchestration.tools import eval as ev


def item_ids(items):
    return [str(i.get("id")) for i in items]


# ===========================================================================
#  1. 入队校验（非法分类不许直达队列）
# ===========================================================================


def test_enqueue_rejects_unknown_kind(iso_bench):
    with pytest.raises(ValueError, match="未知队列项类型"):
        benchd.enqueue("bench", cases=["fig-basic"])
    assert benchd.queue_load() == [], "被拒的条目不得留下半截文件"


def test_enqueue_requires_both_refs_for_ab(iso_bench):
    with pytest.raises(ValueError, match="两端"):
        benchd.enqueue("ab", ref_a="HEAD~1")
    with pytest.raises(ValueError, match="两端"):
        benchd.enqueue("ab", ref_b="HEAD")


def test_enqueue_requires_case_for_capture(iso_bench):
    with pytest.raises(ValueError, match="按案归档"):
        benchd.enqueue("capture")


def test_enqueue_appends_and_summarizes(iso_bench):
    first = benchd.enqueue("run", cases=["fig-basic", "lit-arxiv"], stamp="T1")
    second = benchd.enqueue("check", brief="查一下这轮读数", stamp="T1")
    assert item_ids(benchd.queue_load()) == [first["id"], second["id"]]
    assert first["id"] == "bench-T1-run" and second["id"] == "bench-T1-check"
    assert first["summary"] == "run: fig-basic,lit-arxiv"
    assert first["status"] == "pending" and first["requested_by"] == "leader"


def test_same_second_ids_do_not_collide(iso_bench):
    """同秒顺延：id 唯一是销账依据（与 devops run 名同秒顺延同一套认知）。"""
    a = benchd.enqueue("run", stamp="T1")
    b = benchd.enqueue("run", stamp="T1")
    c = benchd.enqueue("run", stamp="T1")
    assert [a["id"], b["id"], c["id"]] == [
        "bench-T1-run",
        "bench-T1-run-002",
        "bench-T1-run-003",
    ]


def test_queue_load_survives_corruption(iso_bench):
    benchd.enqueue("run", stamp="T1")
    benchd.QUEUE_PATH.write_text("{ 坏 JSON", encoding="utf-8")
    assert benchd.queue_load() == []
    items = benchd.queue_load() + [{"id": "keep-me"}]
    benchd.queue_save(items)
    assert item_ids(benchd.queue_load()) == ["keep-me"], "整体重写即净化坏行"


# ===========================================================================
#  2. FIFO / 孤儿 / 终态面
# ===========================================================================


def test_take_next_is_fifo_and_marks_in_progress(iso_bench):
    benchd.enqueue("run", stamp="T1")
    benchd.enqueue("capture", cases=["lit-arxiv"], stamp="T2")
    got = benchd.take_next()
    assert got["id"] == "bench-T1-run" and got["status"] == "in_progress"
    assert benchd.take_next()["id"] == "bench-T2-capture"
    assert benchd.take_next() is None, "没有 pending 就不该再发活"


def test_reset_orphans_reclaims_inflight(iso_bench):
    """刚获锁时 ``in_progress``＝已死 worker 的孤儿：复位为 pending 并留痕。"""
    benchd.enqueue("run", stamp="T1")
    benchd.take_next()
    reclaimed = benchd.reset_orphans()
    assert reclaimed == ["bench-T1-run"]
    it = benchd.queue_load()[0]
    assert it["status"] == "pending" and "孤儿回收" in it["note"]
    assert it["reclaimed_at"], "回收时刻要留证（谁在什么时候被抢）"
    assert benchd.reset_orphans() == [], "无孤儿即空手，不改写队列"


def test_review_face_lifecycle(iso_bench):
    benchd.enqueue("run", stamp="T1")
    item = benchd.take_next()
    assert benchd.pending_review() == [], "在途项不进待查看面"
    benchd.finish(item["id"], "done", {"headline": "通过 2/2"})
    assert [i["id"] for i in benchd.pending_review()] == [item["id"]]
    assert benchd.mark_seen(item["id"])["reviewed_at"]
    assert benchd.pending_review() == [], "看完即摘面"
    assert item_ids(benchd.queue_load()) == [item["id"]], "销面不等于删审计"


def test_prune_seen_bounds_the_queue_file(iso_bench):
    for n in range(25):
        it = benchd.enqueue("run", stamp=f"T{n:02d}")
        benchd.finish(it["id"], "done", {})
        benchd.mark_seen(it["id"])
    assert len(benchd.queue_load()) == 25
    assert benchd.prune_seen(keep=20) == 5
    assert len(benchd.queue_load()) == 20
    assert benchd.queue_load()[-1]["id"] == "bench-T24-run", "裁头留尾"


def test_queue_counts_by_status(iso_bench):
    benchd.enqueue("run", stamp="T1")
    benchd.enqueue("run", stamp="T2")
    benchd.take_next()
    assert benchd.queue_counts() == {"pending": 1, "in_progress": 1}


# ===========================================================================
#  3. 条目执行（异常即终态，一项坏不许拖垮整轮）
# ===========================================================================


def fake_case(case_id: str):
    return argparse.Namespace(case_id=case_id, member="figure", tier="minimum")


def row(case: str, verdict: str = "pass", *, turns: int = 5, reasons=None):
    return ev.Row(
        ts="2026-10-12T00:00:00+08:00",
        case=case,
        member="figure",
        tier="minimum",
        model="Qwen3.8-Flash",
        verdict=verdict,
        reasons=reasons or [],
        num_turns=turns,
        duration_ms=12000,
        max_turns=12,
        runner=ev.RUNNER_BENCHER,
    )


def test_headline_per_kind():
    ok = {"cases": {"a": {"verdict": "pass"}, "b": {"verdict": "fail"}}}
    assert benchd._headline({"kind": "run"}, ok) == "通过 1/2"
    ab = {"verdict": "未改善", "ref_a": "X", "ref_b": "Y", "escalate": "twin"}
    assert benchd._headline({"kind": "ab"}, ab) == "未改善（X→Y），escalate=twin"
    cap = {"case": "fig-basic", "ref": "HEAD"}
    assert benchd._headline({"kind": "capture"}, cap) == "基线已捕获 fig-basic @HEAD"
    chk = {"delivery": "result"}
    assert benchd._headline({"kind": "check"}, chk) == "组员交付=<result>"
    assert benchd._headline({"kind": "run"}, {"error": "案卷不存在"}) == "案卷不存在", (
        "错误优先于类型读数"
    )


def test_run_item_writes_terminal_state_and_headline(iso_bench, monkeypatch, tmp_path):
    monkeypatch.setattr(ev, "load_cases", lambda case_id="": [fake_case("fig-basic")])
    monkeypatch.setattr(ev, "run_case", lambda case, **kw: row(case.case_id, turns=7))
    it = benchd.enqueue("run", cases=["fig-basic"], stamp="T1")
    res = benchd.run_item(it, tmp_path / "run.log")
    assert res["status"] == "done" and res["headline"] == "通过 1/1"
    assert res["cases"]["fig-basic"]["num_turns"] == 7
    assert benchd.queue_load()[0]["result"]["headline"] == "通过 1/1"


def test_run_item_passes_bencher_runner(iso_bench, monkeypatch, tmp_path):
    """台账行的发起方必须是 bencher（与组长抽检通道分账——判据 6 的机械面）。"""
    seen: dict = {}

    def spy(case, **kw):
        seen.update(kw)
        return row(case.case_id)

    monkeypatch.setattr(ev, "load_cases", lambda case_id="": [fake_case("fig-basic")])
    monkeypatch.setattr(ev, "run_case", spy)
    it = benchd.enqueue("run", cases=["fig-basic"], label="e4", tier="minimum")
    benchd.run_item(it, tmp_path / "run.log")
    assert seen["runner"] == ev.RUNNER_BENCHER
    assert seen["label"] == "e4" and seen["tier"] == "minimum"


def test_missing_case_marks_failed_but_keeps_queue(iso_bench, monkeypatch, tmp_path):
    monkeypatch.setattr(ev, "load_cases", lambda case_id="": [fake_case("fig-basic")])
    it = benchd.enqueue("run", cases=["不存在案"], stamp="T1")
    res = benchd.run_item(it, tmp_path / "run.log")
    assert res["status"] == "failed" and "案卷不存在" in res["error"]
    assert benchd.queue_load()[0]["status"] == "failed"


def test_unknown_kind_fails_without_raising(iso_bench, tmp_path):
    """后台无人接盘：队列文件被人手改成未知 kind 也不能炸 worker，只能变终态。"""
    benchd.queue_save([{"id": "bench-x-ghost", "kind": "ghost"}])
    res = benchd.run_item(benchd.queue_load()[0], tmp_path / "run.log")
    assert res["status"] == "failed" and "未知类型" in res["error"]


# ===========================================================================
#  4. worker 生命周期（非常驻：非空拉起、消化完销毁）
# ===========================================================================


def _run_log(iso_bench) -> Path:
    return benchd.new_run_log(benchd.BENCH_PATHS)


def test_dry_drain_end_to_end(iso_bench, monkeypatch):
    """判据 3 的机械面：注入 → 拉起 → 消化 → 自毁 → 结果进待查看面。"""
    monkeypatch.setattr(ev, "load_cases", lambda case_id="": [fake_case("fig-basic")])
    a = benchd.enqueue("run", cases=["fig-basic"], stamp="T1")
    b = benchd.enqueue("capture", cases=["lit-arxiv"], stamp="T2")
    log = _run_log(iso_bench)
    summary = benchd.bench_drain(log, dry=True)
    assert summary["count"] == 2 and summary["completed"]
    assert [i["id"] for i in summary["items"]] == [a["id"], b["id"]]
    assert benchd.lock_status(benchd.BENCH_PATHS)["state"] == "idle", "自毁即放锁"
    done = log.with_suffix(".done")
    assert done.exists() and json.loads(done.read_text(encoding="utf-8"))["count"] == 2
    assert item_ids(benchd.pending_review()) == [a["id"], b["id"]]
    text = log.read_text(encoding="utf-8")
    assert "bencher worker 启动" in text and "bencher worker 结束" in text


def test_dry_drain_reclaims_orphans(iso_bench):
    benchd.enqueue("run", stamp="T1")
    benchd.take_next()  # 模拟上一任被杀死时留下的在途项
    summary = benchd.bench_drain(_run_log(iso_bench), dry=True)
    assert summary["reclaimed"] == ["bench-T1-run"]
    assert summary["count"] == 1 and benchd.queue_load()[0]["status"] == "done"


def test_empty_queue_drain_still_self_destructs(iso_bench):
    summary = benchd.bench_drain(_run_log(iso_bench), dry=True)
    assert summary["count"] == 0
    assert benchd.lock_status(benchd.BENCH_PATHS)["state"] == "idle"


def test_drain_skips_when_lock_busy(iso_bench, monkeypatch):
    """另一 worker 持活锁 → 优雅退出，**不消费条目**（单 worker 互斥）。"""
    (iso_bench / "bencher.lock").write_text(
        json.dumps({"pid": 4242, "started": drain._now(), "run_log": "r.log"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: True)
    benchd.enqueue("run", stamp="T1")
    summary = benchd.bench_drain(_run_log(iso_bench), dry=True)
    assert summary["skipped"] == "lock_busy"
    assert benchd.queue_load()[0]["status"] == "pending", "被跳过不得消费条目"


def test_cooperative_stop_exits_between_items(iso_bench):
    """停止信号在**条目间隙**生效：不撕正在跑的测量，只不再取下一项。"""
    benchd.enqueue("run", stamp="T1")
    (iso_bench / "bencher.cancel").write_text(
        json.dumps({"pid": os.getpid(), "requested_at": drain._now()}),
        encoding="utf-8",
    )
    summary = benchd.bench_drain(_run_log(iso_bench), dry=True)
    assert summary["stopped"] == "cooperative" and summary["count"] == 0
    assert not (iso_bench / "bencher.cancel").exists(), "信号消费即自删"
    assert benchd.queue_load()[0]["status"] == "pending"


def test_quota_exhausted_stops_and_leaves_pending(iso_bench, monkeypatch):
    """额度是渠道级系统性故障：全局停止，剩余条目保持 pending 待恢复（不标 failed）。"""
    monkeypatch.setattr(ev, "load_cases", lambda case_id="": [fake_case("fig-basic")])
    monkeypatch.setattr(
        ev,
        "run_case",
        lambda case, **kw: row(case.case_id, "fail", reasons=["quota_exhausted: 渠道"]),
    )
    benchd.enqueue("run", cases=["fig-basic"], stamp="T1")
    benchd.enqueue("run", cases=["fig-basic"], stamp="T2")
    summary = benchd.bench_drain(_run_log(iso_bench))
    assert summary["stopped"] == "quota_exhausted"
    assert "额度耗尽" in summary["error"]
    assert benchd.queue_load()[1]["status"] == "pending"
    assert benchd.lock_status(benchd.BENCH_PATHS)["state"] == "idle"


# ===========================================================================
#  5. 唤醒器：拉起但不打断（§5.1-B / §8）
# ===========================================================================


def _spawns(monkeypatch):
    calls = []
    monkeypatch.setattr(
        benchd, "_spawn_detached", lambda argv, **kw: calls.append(argv)
    )
    return calls


def test_wake_bencher_spawns_internal_worker(iso_bench, monkeypatch):
    calls = _spawns(monkeypatch)
    spawned, msg = benchd.wake_bencher()
    assert spawned is True and len(calls) == 1
    argv = calls[0]
    assert argv[1] == "-m" and argv[2] == "pysci.skills.orchestration.tools.orch"
    assert "_bench-worker" in argv and "--run-log" in argv


def test_wake_bencher_does_not_arm_watch(iso_bench, monkeypatch):
    """与 devops drain 的唯一行为差别：**不挂 watch**（测量完成属择机查看型通知）。"""
    from pysci.skills.orchestration.tools import watch

    calls = _spawns(monkeypatch)
    assert calls == []
    _spawned, msg = benchd.wake_bencher()
    assert "待查看" in msg and "orch watch" not in msg
    assert not watch.WATCH_PATH.exists(), "唤醒器不得覆写在飞的心跳面"


def test_wake_bencher_skips_when_running(iso_bench, monkeypatch):
    calls = _spawns(monkeypatch)
    (iso_bench / "bencher.lock").write_text(
        json.dumps({"pid": 4242, "started": drain._now(), "run_log": "r.log"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: True)
    spawned, msg = benchd.wake_bencher()
    assert spawned is False and calls == []
    assert "不重复唤醒" in msg


def test_wake_respects_dry_env_and_force(iso_bench, monkeypatch):
    calls = _spawns(monkeypatch)
    monkeypatch.setenv(drain.DRY_ENV, "1")
    benchd.wake_bencher()
    assert "--dry" in calls[0]
    benchd.wake_bencher(force_dry=False)
    assert "--dry" not in calls[1], "force_dry 覆盖环境变量（演练与真跑分得开）"


# ===========================================================================
#  6. CLI 面与 status 待查看面
# ===========================================================================


def test_bench_add_and_up_via_orch_main(iso_bench, monkeypatch, capsys):
    calls = _spawns(monkeypatch)
    rc = orch.main(
        ["bench-add", "run", "--cases", "fig-basic,lit-arxiv", "--label", "e4"]
    )
    assert rc == 0
    assert "[NEXT]" in capsys.readouterr().out and calls == []
    rc = orch.main(["bench-up", "--dry"])
    assert rc == 0 and len(calls) == 1 and "--dry" in calls[0]


def test_bench_add_rejects_bad_kind_with_rc2(iso_bench, capsys):
    """ab 少一端：parser 放行（kind 合法），**入队闸**拦住并回 rc=2（不入库）。"""
    assert orch.main(["bench-add", "ab", "--ref-a", "X"]) == 2
    assert "入队被拒" in capsys.readouterr().out
    assert benchd.queue_load() == [], "被拒的项不落队列"
    assert orch.main(["bench-add", "run", "--cases", "fig-basic"]) == 0
    assert len(benchd.queue_load()) == 1


def test_bench_add_unknown_kind_is_parser_error(iso_bench):
    """kind 词表在 parser 面即执法（choices），非法值到不了业务层。"""
    with pytest.raises(SystemExit) as exc:
        orch.main(["bench-add", "bench"])
    assert exc.value.code == 2
    assert benchd.queue_load() == []


def test_bench_queue_seen_and_stop(iso_bench, monkeypatch, capsys):
    monkeypatch.setattr(ev, "load_cases", lambda case_id="": [fake_case("fig-basic")])
    it = benchd.enqueue("run", cases=["fig-basic"], stamp="T1")
    benchd.bench_drain(_run_log(iso_bench), dry=True)
    assert orch.main(["bench-queue"]) == 0
    out = capsys.readouterr().out
    assert it["id"] in out and "run: fig-basic" in out
    assert orch.main(["bench-seen", it["id"]]) == 0
    assert orch.main(["bench-seen", "不存在id"]) == 2
    assert orch.main(["bench-queue", "--all"]) == 0
    assert "已查看" in capsys.readouterr().out
    assert orch.main(["bench-stop"]) == 1, "锁空闲时无处可停（残留信号会危及下一任）"


def test_status_shows_pending_review(iso_bench, iso_registry, monkeypatch, capsys):
    """待查看面出现在 ``orch status``，含各 kind 的一句话判决（择机查看，不即时唤醒）。"""
    monkeypatch.setattr(ev, "load_cases", lambda case_id="": [fake_case("fig-basic")])
    monkeypatch.setattr(
        ev,
        "run_case",
        lambda case, **kw: row(case.case_id, "fail", reasons=["断言未命中"]),
    )
    it = benchd.enqueue("run", cases=["fig-basic"], stamp="T1")
    benchd.bench_drain(_run_log(iso_bench))
    orch.cmd_status(argparse.Namespace())
    out = capsys.readouterr().out
    assert "bencher worker" in out and "idle" in out
    assert "[待查看结果]" in out and it["id"] in out
    assert "通过 0/1" in out, "status 要直接给出判决，不只有 id"
    benchd.mark_seen(it["id"])
    orch.cmd_status(argparse.Namespace())
    assert "待查看结果" not in capsys.readouterr().out


def test_bench_worker_command_resolves_relative_log(iso_bench, monkeypatch):
    """内部命令 ``_bench-worker``：相对 run 日志按项目根解析（worker 的 cwd 不可假设）。"""
    monkeypatch.setattr(ev, "load_cases", lambda case_id="": [fake_case("fig-basic")])
    monkeypatch.setattr(benchd, "PROJECT_ROOT", iso_bench.parent)
    benchd.enqueue("run", cases=["fig-basic"], stamp="T1")
    rel = _run_log(iso_bench).relative_to(iso_bench.parent)
    assert not rel.is_absolute()
    assert orch.main(["_bench-worker", "--run-log", str(rel), "--dry"]) == 0
    assert benchd.queue_load()[0]["status"] == "done", "相对路径必须落在同一个日志面上"
