"""背压原语：活跃会话注册表 + 延期纯函数库（backlog 20261013-193000-leader）。

这是「带宽治理」的**安全半**（原 #115 拆单 1/2）：纯配置面 + 纯函数库，**不接 drain 主
循环、不改变任何现有行为**。本组只测原语本身；「零行为变更」的机械证明是全量套件里
既有 `channel_error` 全局停止用例照绿（见交付证据），不是本文件的断言。

矩阵覆盖 success_criteria 的两条：注册表生命周期（写/读/stale 清理/心跳刷新）
与延期纯函数（退避序列/预算触顶/边界）。
"""

from __future__ import annotations

import json

import pytest

from pysci.skills.orchestration.tools import backpressure as bp
from pysci.skills.orchestration.tools import settings

NOW = 1_800_000_000.0


def _reg(tmp_path):
    return tmp_path / "active-sessions.json"


# ---------------------------------------------------------------------------
# ① 注册表生命周期
# ---------------------------------------------------------------------------
def test_note_read_and_drop(tmp_path):
    p = _reg(tmp_path)
    bp.note_session(
        "devops", sid="s1", tier="devops", purpose="任务书 A", now=NOW, path=p
    )
    entries = bp.read_registry(p)
    assert [(e["worker"], e["sid"], e["tier"]) for e in entries] == [
        ("devops", "s1", "devops")
    ]
    assert entries[0]["purpose"] == "任务书 A"

    assert bp.drop_session("devops", "s1", path=p) is True
    assert bp.read_registry(p) == []
    assert bp.drop_session("devops", "s1", path=p) is False, "摘不存在的会话返回 False"


def test_note_is_upsert_and_keeps_started(tmp_path):
    """同一 worker+sid 再次登记 = 刷新，**started 保留首次值**（别把「跑了多久」抹掉）。"""
    p = _reg(tmp_path)
    first = bp.note_session(
        "devops", sid="s1", tier="devops", purpose="A", now=NOW, path=p
    )
    again = bp.note_session(
        "devops", sid="s1", tier="devops", purpose="B", now=NOW + 300, path=p
    )
    assert len(bp.read_registry(p)) == 1, "同 worker+sid 不许出现两条"
    assert again["started"] == first["started"]
    assert again["updated"] != first["updated"] and again["purpose"] == "B"


def test_heartbeat_refreshes_only_updated(tmp_path):
    p = _reg(tmp_path)
    bp.note_session("devops", sid="s1", tier="devops", purpose="A", now=NOW, path=p)
    assert bp.heartbeat("devops", "s1", now=NOW + 120, path=p) is True
    entry = bp.read_registry(p)[0]
    assert bp._epoch(entry["updated"]) == pytest.approx(NOW + 120)
    assert bp._epoch(entry["started"]) == pytest.approx(NOW), "心跳不动 started"
    assert bp.heartbeat("devops", "nope", now=NOW, path=p) is False


def test_live_entries_drops_stale(tmp_path):
    """>10min 无更新 = 死条目（worker 被杀/进程没了），读侧不上报它。"""
    p = _reg(tmp_path)
    bp.note_session("devops", sid="live", tier="devops", purpose="A", now=NOW, path=p)
    bp.note_session("twin", sid="dead", tier="twin", purpose="B", now=NOW, path=p)

    fresh = bp.live_entries(now=NOW + 60, path=p)
    assert [e["sid"] for e in fresh] == ["live", "dead"], "1 分钟内的都算活的"

    later = bp.live_entries(now=NOW + bp.STALE_S + 1, path=p)
    assert later == [], "两边都静默超过阈值 → 全判死"
    assert len(bp.read_registry(p)) == 2, "读侧不负责写：死条目仍在盘上，由 prune 清"


def test_prune_stale_removes_only_dead(tmp_path):
    p = _reg(tmp_path)
    bp.note_session("devops", sid="old", tier="devops", purpose="A", now=NOW, path=p)
    bp.note_session("twin", sid="new", tier="twin", purpose="B", now=NOW, path=p)
    bp.heartbeat("twin", "new", now=NOW + 900, path=p)  # 只让 twin 心跳

    dropped = bp.prune_stale(now=NOW + 901, path=p)
    assert dropped == 1
    assert [e["sid"] for e in bp.read_registry(p)] == ["new"]

    assert bp.prune_stale(now=NOW + 902, path=p) == 0, "没有死的就一条不删"


def test_registry_read_is_fail_open(tmp_path):
    """缺失/坏 JSON/形状不对 → 空表（观察面不堵派发链）。"""
    p = _reg(tmp_path)
    assert bp.read_registry(p) == []
    p.write_text("{not json", encoding="utf-8")
    assert bp.read_registry(p) == []
    p.write_text(json.dumps({"sessions": "nope"}), encoding="utf-8")
    assert bp.read_registry(p) == []
    p.write_text(json.dumps({"sessions": [1, {"worker": "x"}]}), encoding="utf-8")
    assert bp.read_registry(p) == [{"worker": "x"}], "非 dict 条目被滤掉"


# ---------------------------------------------------------------------------
# ② 延期纯函数：退避序列 / 双闸 / 边界
# ---------------------------------------------------------------------------
def test_defer_backoff_sequence():
    """退避 2→4→8→16→32→64 分钟（基数逐次翻倍）。"""
    assert [bp.defer_delay_s(i) for i in range(6)] == [
        120,
        240,
        480,
        960,
        1920,
        3840,
    ]


def test_plan_defer_happy_path():
    plan = bp.plan_defer(2, 360, now=NOW)
    assert (plan.delay_s, plan.attempt, plan.exhausted, plan.reason) == (
        480,
        2,
        False,
        "",
    )
    assert plan.until == pytest.approx(NOW + 480)


def test_plan_defer_stops_on_attempt_limit():
    plan = bp.plan_defer(bp.DEFER_MAX_ATTEMPTS, 0, now=NOW)
    assert plan.exhausted and plan.delay_s == 0
    assert "次数触顶" in plan.reason


def test_plan_defer_stops_on_wall_clock_budget():
    """次数还没用完，但累计耗时已经吃掉预算 → 立即停（次数闸挡不住「真下线」）。"""
    plan = bp.plan_defer(2, bp.DEFER_BUDGET_S, now=NOW)
    assert plan.exhausted and "耗时触顶" in plan.reason


def test_plan_defer_refuses_when_next_delay_would_overshoot():
    """**序列与预算的交互**：退避指数增长，越往后一次就吃掉越多预算——提前拒比先睡再拒好。"""
    plan = bp.plan_defer(3, 1000, now=NOW)
    assert plan.exhausted and "再延一次会超预算" in plan.reason
    assert plan.until == NOW, "触顶时不许再往前排一个到期时刻"


def test_plan_defer_boundaries_allow_disabling_each_gate():
    """``0`` = 该闸不设限（与 ``skip_window_s=0`` 同口径，便于调试与压测）。

    注意两闸是**独立**的：关掉一道不等于全放行——次数闸关掉后，超长退避照样会被
    预算闸拦下（这正是双闸并存的意义）。
    """
    only_attempts = bp.plan_defer(3, 10_000, now=NOW, budget_s=0)
    assert not only_attempts.exhausted and only_attempts.delay_s == 960

    relaxed = bp.plan_defer(7, 0, now=NOW, max_attempts=0)
    assert "次数触顶" not in relaxed.reason, "次数闸已关：不该再以次数为由拒"
    assert relaxed.exhausted and "超预算" in relaxed.reason, (
        "但预算闸仍在——2**7 分钟的退避本来就该被拦"
    )


def test_plan_defer_negative_inputs_are_clamped():
    """负的 deferred/elapsed 不该算出奇怪结果（防御式：调用方读数可能来自坏台账）。"""
    plan = bp.plan_defer(-3, -1.0, now=NOW)
    assert plan.attempt == 0 and plan.delay_s == bp.DEFER_BASE_S and not plan.exhausted


# ---------------------------------------------------------------------------
# ③ 优先级表（B1）
# ---------------------------------------------------------------------------
def test_priority_order_follows_the_user_table():
    """出厂表＝用户裁决：leader > research > devops > bench-pressure > bencher/twin > inspector。"""
    order = settings.DEFAULT_PRIORITY_ORDER
    assert order[0] == ("leader",)
    assert order[4] == ("bencher", "twin"), "bencher 与 twin 同档（用户表里并列）"
    ranks = [
        settings.priority_rank(t)
        for t in (
            "leader",
            "research",
            "devops",
            "bench-pressure",
            "bencher",
            "inspector",
        )
    ]
    assert ranks == sorted(ranks) and len(set(ranks)) == len(ranks)


def test_priority_rank_same_group_and_unknown():
    data = {"priority": {"order": [["leader"], ["bencher", "twin"], ["inspector"]]}}
    assert settings.priority_rank("bencher", data) == settings.priority_rank(
        "twin", data
    )
    assert settings.priority_rank("leader", data) == 0
    assert settings.priority_rank("nobody", data) == 3, "未登记取末位——不插队"
    assert settings.priority_rank("", data) == 3


def test_priority_order_reads_shipped_config():
    """随附 toml 里的 [priority] 表就是出厂表（用户改的缺省不该与代码里的兜底漂开）。"""
    shipped = settings.priority_order()
    assert shipped == settings.DEFAULT_PRIORITY_ORDER


def test_priority_order_falls_back_on_bad_shape():
    assert settings.priority_order({"priority": {"order": "nope"}}) == (
        settings.DEFAULT_PRIORITY_ORDER
    )
    assert settings.priority_order({"priority": {"order": []}}) == (
        settings.DEFAULT_PRIORITY_ORDER
    )
    assert settings.priority_order({}) == settings.DEFAULT_PRIORITY_ORDER
    # 单项写法也收（["leader"] 与 "leader" 等价），且空项被剔除
    assert settings.priority_order(
        {"priority": {"order": ["leader", ["", "twin"]]}}
    ) == (
        ("leader",),
        ("twin",),
    )
