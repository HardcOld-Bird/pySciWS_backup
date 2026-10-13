"""大单 solo 旗标与组包过载提醒（backlog 20261012-170301-leader）。

实锤形状（run 122624 / 233336）：drain 组包只按「同提请者 + 同 effort」聚类，不估任务
体量——E4+E5a+E5b 三张大单同批 → 一跳消化 → 触顶 150 腰斩。三面修复各有阳性面：

- 组包：``solo: true`` 条目不并入组包、也不作组包种子（batch_size 恒 1）；
- 入队：approve --solo → 条目 JSON 可见 ``solo`` 字段，status [backlog] 行显示 [solo]；
- 提醒：pending > 组包上限 或连续 ≥2 批 run_failed → status「疑似组包过载」提醒行。
"""

from __future__ import annotations

import argparse
import json

from pysci.skills.orchestration.tools import dispatch, drain, orch, workflow
from pysci.skills.orchestration.tools.dispatch import DispatchOutcome

from .conftest import read_backlog, seed_backlog, seed_suggestion


# ---------------------------------------------------------------------------
# 组包：solo 双向不并入
# ---------------------------------------------------------------------------
def test_take_batch_solo_seed_groups_nothing(iso_state):
    """两张 solo 大单 + 一张小单：首包只吃种子（solo 作种子 → batch_size 恒 1）。"""
    seed_backlog(
        iso_state,
        [
            {"id": "big1", "status": "pending", "member": "devops", "solo": True},
            {"id": "big2", "status": "pending", "member": "devops", "solo": True},
            {"id": "small", "status": "pending", "member": "devops"},
        ],
    )
    seed, group = workflow.backlog_take_batch(4)
    assert seed["id"] == "big1"
    assert group == [], "solo 种子不收组包（含同 member 同 effort 的小单）"
    on_disk = {i["id"]: i for i in read_backlog(iso_state)}
    assert on_disk["big2"]["status"] == "pending"
    assert on_disk["small"]["status"] == "pending", "组包不承诺 in_progress（机制不变）"


def test_take_batch_small_seed_skips_solo_items(iso_state):
    """小单作种子：队列里的 solo 大单一律跳过，只并入其余小单。"""
    seed_backlog(
        iso_state,
        [
            {"id": "small1", "status": "pending", "member": "devops"},
            {"id": "big1", "status": "pending", "member": "devops", "solo": True},
            {"id": "big2", "status": "pending", "member": "devops", "solo": True},
            {"id": "small2", "status": "pending", "member": "devops"},
        ],
    )
    seed, group = workflow.backlog_take_batch(4)
    assert seed["id"] == "small1"
    assert [g["id"] for g in group] == ["small2"], "solo 项互不并入"
    seed2, group2 = workflow.backlog_take_batch(4)
    assert seed2["id"] == "big1" and group2 == [], "solo 大单轮到时独占一跳"


def test_drain_solo_items_never_share_a_batch(iso_state, monkeypatch):
    """集成面：2 张 solo 大单 + 1 张小单 → 三次派发、每批 batch_size=1。"""
    seed_backlog(
        iso_state,
        [
            {"id": "big1", "status": "pending", "member": "devops", "solo": True},
            {"id": "big2", "status": "pending", "member": "devops", "solo": True},
            {"id": "small", "status": "pending", "member": "devops"},
        ],
    )
    calls: list[str] = []

    def fake(member, **kw):
        slug = kw.get("slug")
        calls.append(slug)
        return DispatchOutcome(
            code=0, kind="result", body=f"done commit aaa1111\nbacklog id={slug}"
        )

    monkeypatch.setattr(dispatch, "do_dispatch", fake)
    summary = drain.drain_backlog(drain.new_run_log())
    assert calls == ["big1", "big2", "small"], "solo 项各占一跳，互不并批"
    assert summary["count"] == 3
    by_id = {i["id"]: i for i in read_backlog(iso_state)}
    assert all(by_id[x]["batch_size"] == 1 for x in ("big1", "big2", "small"))


# ---------------------------------------------------------------------------
# 入队：--solo → 条目 JSON 字段可见
# ---------------------------------------------------------------------------
def test_approve_solo_lands_on_backlog_item(iso_state):
    sid = seed_suggestion(iso_state)
    rc = workflow.cmd_approve(
        argparse.Namespace(
            suggestion_id=sid,
            note="大单采纳",
            no_wake=True,
            solo=True,
            override="测试自裁",
        )
    )
    assert rc == 0
    assert read_backlog(iso_state)[0]["solo"] is True


def test_approve_without_solo_omits_key(iso_state):
    sid = seed_suggestion(iso_state)
    workflow.cmd_approve(
        argparse.Namespace(suggestion_id=sid, note="n", no_wake=True, override="自裁")
    )
    assert "solo" not in read_backlog(iso_state)[0], "不标不写字段（向后兼容旧数据）"


def test_cli_approve_solo_flag_parses(iso_state, monkeypatch):
    sid = seed_suggestion(iso_state)
    rc = orch.main(
        ["approve", sid, "--note", "n", "--solo", "--no-wake", "--override", "自裁"]
    )
    assert rc == 0
    assert read_backlog(iso_state)[0]["solo"] is True


def test_status_shows_solo_marker(iso_state, iso_registry, monkeypatch, capsys):
    seed_backlog(
        iso_state,
        [
            {
                "id": "big1",
                "status": "pending",
                "member": "devops",
                "summary": "大单",
                "solo": True,
            }
        ],
    )
    orch.cmd_status(argparse.Namespace())
    out = capsys.readouterr().out
    assert "[backlog] big1" in out and "[solo]" in out


# ---------------------------------------------------------------------------
# 过载提醒：pending 超组包上限 / 连续两批 run_failed
# ---------------------------------------------------------------------------
def _write_done(iso_state, name: str, status: str) -> None:
    runs = iso_state / "devops-runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / f"{name}.done").write_text(
        json.dumps(
            {"items": [{"id": f"bk-{name}", "status": status}], "count": 1},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_run_failed_streak_counts_consecutive_batches(iso_state):
    assert drain.run_failed_streak() == 0, "无 run 记录 → 0"
    _write_done(iso_state, "20261011-1", "run_failed")
    _write_done(iso_state, "20261012-2", "run_failed")
    assert drain.run_failed_streak() == 2
    _write_done(iso_state, "20261013-3", "done")
    assert drain.run_failed_streak() == 0, "干净批断链（最新批没失败就不提醒）"


def test_run_failed_streak_ignores_systemic_and_bad_files(iso_state):
    """系统性故障条目不计入（渠道坏了条目无罪）；坏 .done 断链不计也不续。"""
    _write_done(iso_state, "20261011-1", "run_failed")
    runs = iso_state / "devops-runs"
    (runs / "20261012-2.done").write_text("{ 坏 JSON", encoding="utf-8")
    _write_done(iso_state, "20261013-3", "quota_exhausted")
    assert drain.run_failed_streak() == 0, "系统性/坏文件都撑不起「组包过载」结论"


def test_status_overload_hint_on_consecutive_run_failed(
    iso_state, iso_registry, monkeypatch, capsys
):
    _write_done(iso_state, "20261011-1", "run_failed")
    _write_done(iso_state, "20261012-2", "run_failed")
    orch.cmd_status(argparse.Namespace())
    out = capsys.readouterr().out
    assert "疑似组包过载" in out and "--solo" in out


def test_status_overload_hint_on_pending_overflow(
    iso_state, iso_registry, monkeypatch, capsys
):
    seed_backlog(
        iso_state,
        [
            {"id": f"p{i}", "status": "pending", "member": "devops", "summary": "s"}
            for i in range(drain.BATCH_MAX + 1)
        ],
    )
    orch.cmd_status(argparse.Namespace())
    out = capsys.readouterr().out
    assert "疑似组包过载" in out and f"组包上限 {drain.BATCH_MAX}" in out


def test_status_no_hint_when_quiet(iso_state, iso_registry, monkeypatch, capsys):
    _write_done(iso_state, "20261012-2", "run_failed")  # 只一批失败 → 不提醒
    orch.cmd_status(argparse.Namespace())
    assert "疑似组包过载" not in capsys.readouterr().out
