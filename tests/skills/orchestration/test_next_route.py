"""PSEP_3 pb：跨任务路由（§3.2/§3.3）——next 命令卡片 / 排序 / 四推送点 / 优先级贯通。"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from pysci import paths as _paths
from pysci.skills.orchestration.tools import (
    events,
    nextroute,
    orch,
    plans,
    queues,
    queueworker,
    taskbooks,
    watch,
    workq,
)


@pytest.fixture
def rface(iso_state, tmp_path, monkeypatch):
    plans_dir = tmp_path / "plans"
    plans_dir.mkdir()
    monkeypatch.setattr(plans, "PLANS_DIR", plans_dir)
    data = tmp_path / "data"
    monkeypatch.setattr(taskbooks, "TASKBOOKS_DIR", data / "taskbooks")
    monkeypatch.setattr(_paths, "DATA_ROOT", data)
    monkeypatch.setattr(
        queueworker, "wake_member", lambda member, **k: (True, "stub（不消化）")
    )
    return tmp_path


def _args(plan_id: str, **over) -> Namespace:
    base = dict(
        id=plan_id,
        text=None,
        task=None,
        name=None,
        session=None,
        model_tier=None,
        effort=None,
        max_turns=None,
        dirs=None,
        file="",
        force=True,
        evidence=None,
        dry=False,
        mode="full",
        goal=None,
        plan_id=None,
        limit=5,
    )
    base.update(over)
    return Namespace(**base)


def _mk_taskbook(
    title: str, *, priority="normal", mode="", rework_of="", round_no=0
) -> str:
    return taskbooks.create(
        title,
        f"{title} 的一行简介",
        priority=priority,
        mode=mode,
        rework_of=rework_of,
        rework_round=round_no,
    )["id"]


def _mk_plan(tmp: Path, plan_id: str, *, tb_id="", steps=None, mode="") -> str:
    p = tmp / "plans" / f"{plan_id}.plan.yaml"
    doc = {
        "id": plan_id,
        "goal": f"目标 {plan_id}",
        "steps": steps or [{"id": "s1", "member": "sim", "brief": "环节一"}],
    }
    if tb_id:
        doc["taskbook"] = tb_id
    if mode:
        doc["template"] = "rework"
        doc["rework_mode"] = mode
    p.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    plans.cmd_plan_new(_args(plan_id, file=str(p)))
    return plan_id


def _fake_worker(member: str, **kw):
    qp = queues.queue_path(member)
    for it in workq.load(qp):
        if str(it.get("status")) == workq.STATUS_DONE:
            continue
        workq.mutate(
            qp,
            it["id"],
            status=workq.STATUS_DONE,
            result={
                "code": 0,
                "kind": "result",
                "body": "done",
                "checks": [],
                "member": member,
            },
        )
    return (True, "stub worker")


# ===========================================================================
#  ① 排序：优先级 → 动作类别 → 序
# ===========================================================================
def test_priority_high_before_normal(rface):
    tb_hi = _mk_taskbook("高优大任务", priority="high")
    tb_no = _mk_taskbook("普通大任务", priority="normal")
    _mk_plan(rface, "p-hi", tb_id=tb_hi)
    _mk_plan(rface, "p-no", tb_id=tb_no)
    acts = nextroute.sort_actions(nextroute.collect_actions())
    assert [a["tb_id"] for a in acts] == [tb_hi, tb_no]
    assert acts[0]["priority"] == "high" and acts[1]["priority"] == "normal"


def test_category_blocked_before_dispatch(rface):
    tb = _mk_taskbook("大任务", priority="normal")
    # 一个 plan：s1 blocked（need_leader），另一 plan 可派发
    p1 = _mk_plan(
        rface, "p-blk", tb_id=tb, steps=[{"id": "s1", "member": "sim", "brief": "甲"}]
    )
    _mk_plan(rface, "p-disp", steps=[{"id": "s1", "member": "sim", "brief": "乙"}])
    st = plans.load_state(p1)
    st["steps"][0]["status"] = "blocked"
    plans._save_state(p1, st)
    acts = nextroute.sort_actions(nextroute.collect_actions())
    assert [a["category"] for a in acts] == ["blocked", "dispatch"]


def test_unlinked_plan_is_normal(rface):
    _mk_plan(rface, "p-solo")  # 无 taskbook 关联
    acts = nextroute.collect_actions()
    assert acts and acts[0]["priority"] == "normal"
    assert acts[0]["tb_id"] == ""


# ===========================================================================
#  ② 任务卡片形状
# ===========================================================================
def test_card_has_required_fields(rface, capsys):
    tb = _mk_taskbook("改善交付质量", priority="high")
    _mk_plan(rface, "p-card", tb_id=tb)
    out = "\n".join(nextroute.next_lines())
    assert f"【大任务 {tb}：改善交付质量】" in out
    assert "优先级=high" in out
    assert "简介：改善交付质量 的一行简介" in out
    assert "位置：plan p-card @ s1 环节一" in out
    assert "叶子进度 0/1" in out
    assert "动作：书写本环节任务书并派发" in out
    assert "命令模板：pysci-orch plan advance p-card" in out
    assert orch.main(["next"]) == 0
    assert "orch next" in capsys.readouterr().out


def test_next_simple_table_lists_rest(rface):
    tb1 = _mk_taskbook("甲任务", priority="high")
    tb2 = _mk_taskbook("乙任务", priority="normal")
    _mk_plan(rface, "p1", tb_id=tb1)
    _mk_plan(rface, "p2", tb_id=tb2)
    lines = nextroute.next_lines()
    assert any("其余就绪项" in ln for ln in lines)
    assert any("乙任务" in ln for ln in lines)


# ===========================================================================
#  ③ 四个机械推送点
# ===========================================================================
def test_push_advance_tail(rface, capsys, monkeypatch):
    tb = _mk_taskbook("大任务", priority="high")
    _mk_plan(
        rface,
        "p-adv",
        tb_id=tb,
        steps=[
            {
                "id": "G",
                "mode": "parallel",
                "children": [
                    {"id": "s1", "member": "sim", "brief": "甲"},
                    {"id": "s2", "member": "sim", "brief": "乙"},
                ],
            },
        ],
    )
    monkeypatch.setattr(queueworker, "wake_member", lambda m, **k: (True, "stub"))
    capsys.readouterr()
    assert plans.cmd_plan_advance(_args("p-adv", text="t")) == 0
    out = capsys.readouterr().out
    assert "[NEXT] 工作清单" in out, "三栏工作清单并存"
    assert "路由卡片" in out and f"【大任务 {tb}：" in out


def test_push_drain_tail(rface, capsys):
    tb = _mk_taskbook("大任务", priority="high")
    _mk_plan(rface, "p-drn", tb_id=tb)
    events.emit(events.JOB_DONE, subject="x", detail="d", action="a")
    capsys.readouterr()
    assert orch.main(["events", "drain"]) == 0
    out = capsys.readouterr().out
    assert "orch next" in out and f"【大任务 {tb}：" in out


def test_push_watch_detail(rface, monkeypatch, owner_side):
    tb = _mk_taskbook("大任务", priority="high")
    _mk_plan(rface, "p-w", tb_id=tb)
    echoed: list[str] = []
    fired = watch._emit(
        {"armed_by": owner_side},
        [
            {
                "event": "JOB-DONE",
                "subject": "job-1",
                "detail": "worker 收尾",
                "action": "推进",
            }
        ],
        echo=echoed.append,
    )
    assert fired is True
    joined = "\n".join(echoed)
    assert "JOB-DONE" in joined and f"【大任务 {tb}：" in joined


def test_push_run_handoff(rface, capsys, monkeypatch):
    tb = _mk_taskbook("大任务", priority="high")
    _mk_plan(
        rface,
        "p-run",
        tb_id=tb,
        steps=[
            {
                "id": "G",
                "mode": "parallel",
                "children": [
                    {"id": "s1", "member": "sim", "brief": "甲"},
                    {"id": "s2", "member": "sim", "brief": "乙"},
                ],
            },
        ],
    )
    monkeypatch.setattr(queueworker, "wake_member", lambda m, **k: _fake_worker(m))
    capsys.readouterr()
    plans.cmd_plan_run(_args("p-run", text="t"))
    out = capsys.readouterr().out
    assert f"【大任务 {tb}：" in out


# ===========================================================================
#  ④ 返工任务书的卡片（模式/轮次）
# ===========================================================================
def test_card_shows_rework_mode_and_round(rface):
    tb0 = _mk_taskbook("大任务")
    tb1 = _mk_taskbook(
        "大任务", mode="full", rework_of=tb0, round_no=2, priority="high"
    )
    _mk_plan(
        rface,
        "p-rw",
        tb_id=tb1,
        mode="full",
        steps=[
            {
                "id": "A",
                "mode": "serial",
                "brief": "综合项目改进",
                "children": [{"id": "a1", "kind": "self", "brief": "成因分析报告"}],
            }
        ],
    )
    out = "\n".join(nextroute.next_lines())
    assert "模式=改进：基础设施+监察／第 2 轮返工" in out
    assert "位置：plan p-rw @ A 综合项目改进 > a1 成因分析报告" in out


# ===========================================================================
#  ⑤ serial 门闸 + 要求与历史总览附卡（用户 2026-10-17）
# ===========================================================================
def _serial(tmp_path, monkeypatch, mode: str = "serial"):
    from pysci.skills.orchestration.tools import settings

    path = tmp_path / "pysci_settings.toml"
    path.write_text(f'[taskbooks]\nexecution_mode = "{mode}"\n', encoding="utf-8")
    monkeypatch.setattr(settings, "SETTINGS_PATH", path)


def test_serial_hides_queued_taskbook_cards(rface, tmp_path, monkeypatch):
    """serial：非队首大任务的卡片（plan 面+待办卡）一律不推；队列状态块随卡播报。"""
    _serial(tmp_path, monkeypatch)
    head = _mk_taskbook("甲")
    queued = _mk_taskbook("乙")
    _mk_plan(rface, "p-b", tb_id=queued)
    acts = nextroute.collect_actions()
    assert any(a["tb_id"] == head for a in acts), "队首待办照常推"
    assert all(a["tb_id"] != queued for a in acts), "排队件不推待办"
    out = "\n".join(nextroute.next_lines())
    assert "串行执行模式：队首" in out and f"排队 1 件：{queued}" in out
    _serial(tmp_path, monkeypatch, mode="parallel")
    assert any(a["tb_id"] == queued for a in nextroute.collect_actions()), (
        "热切换即生效"
    )


def test_serial_keeps_user_review_cards(rface, tmp_path, monkeypatch):
    """delivered「用户验收」是用户侧动作，不受 serial 门闸约束。"""
    _serial(tmp_path, monkeypatch)
    _mk_taskbook("甲")
    done = _mk_taskbook("乙")
    taskbooks.transition(done, "active")
    taskbooks.transition(done, "delivered")
    acts = nextroute.collect_actions()
    assert any(a["tb_id"] == done and a["category"] == "review" for a in acts)


def test_card_attaches_requirements_digest_only_with_history(rface):
    root = _mk_taskbook("甲")
    taskbooks.create_rework(root, improve={}, note="这版不行")
    out = "\n".join(nextroute.next_lines())
    assert "要求与历史总览（自动附送）" in out and "这版不行" in out

    card = "\n".join(
        nextroute.render_card(
            {"tb_id": _mk_taskbook("丙"), "tb_title": "丙", "category": "dispatch"}
        )
    )
    assert "要求与历史总览" not in card, "无历史任务书不附总览块（卡片保持紧凑）"


def test_intro_skips_rework_metadata_prefix(rface):
    """卡简介回归用户内容——旧前缀「（返工第 n 轮…）」不再占位。"""
    root = taskbooks.create("甲", "真正的要求内容")["id"]
    rw = taskbooks.create_rework(root, improve={}, note="不行")["id"]
    assert nextroute._intro(rw) == "真正的要求内容"


# ===========================================================================
#  ⑥ 卡片取工作清单真实状态（tb-5 L.l1 摩擦 #1）
# ===========================================================================
def test_inflight_step_card_waits_instead_of_dispatch(rface, monkeypatch):
    """已入队在跑的环节只报「等待完成事件」，绝不再写「书写本环节任务书并派发」。"""
    tb = _mk_taskbook("在途大任务", priority="high")
    _mk_plan(rface, "p-inflight", tb_id=tb)
    monkeypatch.setattr(queueworker, "wake_member", lambda m, **k: (True, "stub"))
    assert plans.cmd_plan_advance(_args("p-inflight", text="t")) == 0

    acts = [a for a in nextroute.collect_actions() if a["plan_id"] == "p-inflight"]
    assert [a["category"] for a in acts] == ["inflight"], acts
    assert "等待完成事件" in acts[0]["action"]

    out = "\n".join(nextroute.next_lines())
    assert f"【大任务 {tb}：" in out and "动作：等待完成事件" in out
    assert "动作：书写本环节任务书并派发" not in out, "在途环节被再次引导派发"


def test_state_less_duplicate_plan_does_not_hijack_the_card(rface, monkeypatch):
    """遗留重复 YAML（无 state 档）不进路由；link + 在途的正式 plan 正确回显大任务。

    实案（tb-5）：`<id>.plan.yaml` 与 `<id>.plan.plan.yaml` 并存，前者无 state，全部
    叶子按 pending 兜底 → 卡片长期停在它的 w1 并显示「（无关联任务书）」。
    """
    tb = _mk_taskbook("正式大任务", priority="high")
    taskbooks.link_plan(tb, "p-real")
    _mk_plan(rface, "p-real", tb_id=tb)
    monkeypatch.setattr(queueworker, "wake_member", lambda m, **k: (True, "stub"))
    assert plans.cmd_plan_advance(_args("p-real", text="t")) == 0

    stale = rface / "plans" / "p-legacy.plan.yaml"
    stale.write_text(
        yaml.safe_dump(
            {
                "id": "p-legacy",
                "goal": "旧目标",
                "steps": [{"id": "w1", "member": "sim", "brief": "旧环节"}],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    acts = nextroute.collect_actions()
    assert all(a["plan_id"] != "p-legacy" for a in acts), (
        "无 state 的遗留件被当成可派发"
    )
    assert any(a["plan_id"] == "p-real" and a["category"] == "inflight" for a in acts)

    out = "\n".join(nextroute.next_lines())
    assert f"【大任务 {tb}：" in out
    assert "（无关联任务书）" not in out
    assert "旧环节" not in out and "动作：等待完成事件" in out


# ===========================================================================
#  ⑦ 命令尾部卡片与 orch next 同源（tb-5 路由卡片残迹）
# ===========================================================================
def test_routing_card_fresh_reloads_router_module(rface, monkeypatch):
    """长驻同步命令尾部强制重载 nextroute——进程启动早于一次升级时不得打印旧判据。"""
    sentinel = lambda *a, **k: ["STALE 旧判据卡片"]  # noqa: E731
    monkeypatch.setattr(nextroute, "card_lines", sentinel)
    assert plans._routing_card() == ["STALE 旧判据卡片"]
    assert plans._routing_card(fresh=True) != ["STALE 旧判据卡片"]


def test_plan_run_tail_card_has_no_dispatch_or_unlinked_header(
    rface, monkeypatch, capsys
):
    """全部完成的计划 + 遗留无 state YAML：plan run/adhoc 尾部不得再推派发卡。"""
    tb = _mk_taskbook("已完成的正式任务", priority="high")
    _mk_plan(rface, "p-done", tb_id=tb)
    taskbooks.link_plan(tb, "p-done")
    stale = rface / "plans" / "p-done-legacy.plan.yaml"
    stale.write_text(
        yaml.safe_dump(
            {
                "id": "p-done-legacy",
                "goal": "旧目标",
                "steps": [{"id": "G1.P1.w1", "member": "sim", "brief": "旧环节"}],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(queueworker, "wake_member", lambda m, **k: _fake_worker(m))
    capsys.readouterr()
    assert plans.cmd_plan_run(_args("p-done", text="t")) == 0
    out = capsys.readouterr().out
    assert "书写本环节任务书并派发" not in out, out
    assert "（无关联任务书）" not in out, out


def test_advance_tail_card_waits_for_inflight(rface, monkeypatch, capsys):
    """在途计划的 advance 尾部卡片＝「等待完成事件」，不写成可派发。"""
    tb = _mk_taskbook("在途正式任务", priority="high")
    _mk_plan(rface, "p-wait", tb_id=tb)
    monkeypatch.setattr(queueworker, "wake_member", lambda m, **k: (True, "stub"))
    capsys.readouterr()
    assert plans.cmd_plan_advance(_args("p-wait", text="t")) == 0
    out = capsys.readouterr().out
    assert "动作：等待完成事件" in out, out
    assert "书写本环节任务书并派发" not in out


# ===========================================================================
#  ⑧ plan advance 输出分层 + adhoc id 唯一性（tb-5 返工 L 摩擦）
# ===========================================================================
def test_advance_digest_is_tail_on_first_guide_only(rface, monkeypatch, capsys):
    """首次引导：行动面在前、总览在后；第二次 advance 不再打总览。"""
    root = _mk_taskbook("原始大任务")
    rw = taskbooks.create_rework(root, improve={}, note="上轮批语")["id"]
    _mk_plan(rface, "p-layer", tb_id=rw)
    monkeypatch.setattr(queueworker, "wake_member", lambda m, **k: (True, "stub"))
    capsys.readouterr()

    assert plans.cmd_plan_advance(_args("p-layer", text="t")) == 0
    first = capsys.readouterr().out
    assert "要求与历史总览" in first, "首次引导仍须提供总览"
    assert first.index("已入队") < first.index("要求与历史总览"), (
        "入队确认/工作清单必须前置，总览在后"
    )
    assert first.index("[NEXT] 工作清单") < first.index("要求与历史总览")

    assert plans.cmd_plan_advance(_args("p-layer")) == 0
    second = capsys.readouterr().out
    assert "要求与历史总览" not in second, "后续 advance 不得再淹没行动面"


def test_advance_full_flag_reprints_digest(rface, monkeypatch, capsys):
    root = _mk_taskbook("原始大任务")
    rw = taskbooks.create_rework(root, improve={}, note="上轮批语")["id"]
    _mk_plan(rface, "p-full", tb_id=rw)
    monkeypatch.setattr(queueworker, "wake_member", lambda m, **k: (True, "stub"))
    capsys.readouterr()

    assert plans.cmd_plan_advance(_args("p-full", text="t")) == 0
    capsys.readouterr()
    assert plans.cmd_plan_advance(_args("p-full", full=True)) == 0
    out = capsys.readouterr().out
    assert "要求与历史总览" in out
    assert out.index("[NEXT] 工作清单") < out.index("要求与历史总览")


def test_adhoc_ids_are_unique_within_the_same_second(rface, monkeypatch, capsys):
    """同一秒两单：后缀递增，各一 plan/state（实测 050159 曾吞两单）。"""
    from datetime import datetime as _dt

    class _Fixed(_dt):
        @classmethod
        def now(cls, tz=None):
            return _dt(2026, 10, 19, 12, 0, 0)

    monkeypatch.setattr(plans, "datetime", _Fixed)
    monkeypatch.setattr(queueworker, "wake_member", lambda m, **k: _fake_worker(m))

    def args() -> Namespace:
        return Namespace(
            member="sim",
            text="单步任务",
            task=None,
            dirs=None,
            model_tier=None,
            effort=None,
            max_turns=None,
            name=None,
            session=None,
            full=False,
            dry=False,
        )

    capsys.readouterr()
    assert plans.cmd_plan_adhoc(args()) == 0
    assert plans.cmd_plan_adhoc(args()) == 0
    ids = [
        p.name[: -len(".state.json")]
        for p in sorted((rface / "plans").glob("adhoc-*.state.json"))
    ]
    assert set(ids) == {"adhoc-20261019-120000", "adhoc-20261019-120000-02"}, ids
    for pid in ids:
        st = plans._load_state(pid)
        assert st["plan_id"] == pid and len(st["steps"]) == 1
        assert st["steps"][0]["status"] == "done", "每单各自完成一跳"
