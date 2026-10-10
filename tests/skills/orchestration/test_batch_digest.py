"""drain 批量消化（backlog 20261010-batch-digest）。

用户裁决 2026-10-10：drain 取队首为种子 → 收集同 member（提请者）pending 为组包（≤4）→
任务书呈现种子+菜单 → devops 自主选取 ≥1 项（必含种子）合并实施 → 交付列
`backlog id=<ids>` → orch 逐 id 销账；未选留 pending 下轮；失败仅种子 needs_leader（不连坐）。
"""

from __future__ import annotations

import argparse

from pysci.skills.orchestration.tools import dispatch, drain, ledger, workflow
from pysci.skills.orchestration.tools.dispatch import DispatchOutcome

from .conftest import read_backlog, seed_backlog


# ---------------------------------------------------------------------------
# backlog_take_batch：种子 + 同 member 组包（≤ max_batch），组包保持 pending
# ---------------------------------------------------------------------------
def test_take_batch_groups_same_member_up_to_cap(iso_state):
    seed_backlog(
        iso_state,
        [{"id": c, "status": "pending", "member": "m"} for c in "abcde"],
    )
    seed, group = workflow.backlog_take_batch(4)
    assert seed["id"] == "a"
    assert seed["status"] == "in_progress" and seed["worktree"] == "a"
    assert [g["id"] for g in group] == ["b", "c", "d"]  # ≤ max_batch-1
    on_disk = {i["id"]: i for i in read_backlog(iso_state)}
    assert on_disk["a"]["status"] == "in_progress"
    assert all(on_disk[x]["status"] == "pending" for x in "bcde"), (
        "组包不承诺 in_progress"
    )


def test_take_batch_excludes_other_members(iso_state):
    seed_backlog(
        iso_state,
        [
            {"id": "a", "status": "pending", "member": "m1"},
            {"id": "b", "status": "pending", "member": "m2"},
            {"id": "c", "status": "pending", "member": "m1"},
        ],
    )
    seed, group = workflow.backlog_take_batch(4)
    assert seed["id"] == "a"
    assert [g["id"] for g in group] == ["c"]  # 仅同提请者 m1


def test_take_batch_no_member_solo(iso_state):
    seed_backlog(
        iso_state,
        [{"id": "a", "status": "pending"}, {"id": "b", "status": "pending"}],
    )
    seed, group = workflow.backlog_take_batch(4)
    assert seed["id"] == "a" and group == []  # member 空不组包（兼容旧数据）


def test_take_batch_empty(iso_state):
    seed_backlog(iso_state, [])
    assert workflow.backlog_take_batch(4) == (None, [])


def test_take_batch_skips_non_pending(iso_state):
    seed_backlog(
        iso_state,
        [
            {"id": "done1", "status": "done", "member": "m"},
            {"id": "a", "status": "pending", "member": "m"},
            {"id": "nl", "status": "needs_leader", "member": "m"},
            {"id": "b", "status": "pending", "member": "m"},
        ],
    )
    seed, group = workflow.backlog_take_batch(4)
    assert seed["id"] == "a" and [g["id"] for g in group] == ["b"]


# ---------------------------------------------------------------------------
# backlog_batch_text：种子任务书 + 菜单 + 选取/销账指令
# ---------------------------------------------------------------------------
def test_batch_text_solo_equals_task_text():
    seed = {"id": "a", "summary": "s", "member": "m"}
    assert workflow.backlog_batch_text(seed, []) == workflow.backlog_task_text(seed)


def test_batch_text_group_menu_and_instructions():
    seed = {"id": "a", "summary": "种子任务", "member": "figure"}
    group = [{"id": "b", "summary": "组包b"}, {"id": "c", "summary": "组包c"}]
    text = workflow.backlog_batch_text(seed, group, max_batch=4)
    assert "批量菜单" in text and "figure" in text
    assert "`b`" in text and "`c`" in text  # 菜单列出组包 id
    assert "必含种子" in text and "backlog id=" in text
    assert "≤ 4" in text  # 护栏批大小
    assert text.startswith(workflow.backlog_task_text(seed).split("\n")[0])


# ---------------------------------------------------------------------------
# _extract_backlog_ids：解析交付的 backlog id 清单
# ---------------------------------------------------------------------------
def test_extract_backlog_ids_single():
    body = "改动完成。backlog id=20261010-abc 以便销账。"
    assert drain._extract_backlog_ids(body) == ["20261010-abc"]


def test_extract_backlog_ids_multi_separators():
    assert drain._extract_backlog_ids("backlog id=a-1,b-2、c-3") == [
        "a-1",
        "b-2",
        "c-3",
    ]


def test_extract_backlog_ids_dedup_and_none():
    assert drain._extract_backlog_ids("backlog id=x,y … backlog id=x") == ["x", "y"]
    assert drain._extract_backlog_ids("无清单") == []


# ---------------------------------------------------------------------------
# _drain_batch：逐 id 销账 / 未选留 pending / 失败不连坐 / quota / dry
# ---------------------------------------------------------------------------
def test_batch_result_completes_listed_only(iso_state, monkeypatch):
    seed_backlog(
        iso_state,
        [
            {"id": "s1", "status": "pending", "member": "figure", "summary": "a"},
            {"id": "s2", "status": "pending", "member": "figure", "summary": "b"},
            {"id": "s3", "status": "pending", "member": "figure", "summary": "c"},
        ],
    )
    seed, group = workflow.backlog_take_batch(4)
    monkeypatch.setattr(
        dispatch,
        "do_dispatch",
        lambda *a, **k: DispatchOutcome(
            code=0, kind="result", body="完成 commit aaa1111\nbacklog id=s1,s2"
        ),
    )
    results = drain._drain_batch(seed, group, drain.new_run_log(), dry=False)
    by_id = {i["id"]: i for i in read_backlog(iso_state)}
    assert by_id["s1"]["status"] == "done" and by_id["s1"]["batch_size"] == 2
    assert by_id["s2"]["status"] == "done" and by_id["s2"]["batch_seed"] == "s1"
    assert by_id["s3"]["status"] == "pending", "未选项留 pending 下轮"
    assert {r["id"] for r in results} == {"s1", "s2"}


def test_batch_result_no_ids_defensive_seed_done(iso_state, monkeypatch):
    """交付未列 backlog id（漏报）→ 种子仍防御性 done，组包留 pending。"""
    seed_backlog(
        iso_state,
        [
            {"id": "s1", "status": "pending", "member": "figure"},
            {"id": "s2", "status": "pending", "member": "figure"},
        ],
    )
    seed, group = workflow.backlog_take_batch(4)
    monkeypatch.setattr(
        dispatch,
        "do_dispatch",
        lambda *a, **k: DispatchOutcome(
            code=0, kind="result", body="done commit bbb2222"
        ),
    )
    drain._drain_batch(seed, group, drain.new_run_log(), dry=False)
    by_id = {i["id"]: i for i in read_backlog(iso_state)}
    assert by_id["s1"]["status"] == "done" and by_id["s1"]["batch_size"] == 1
    assert by_id["s2"]["status"] == "pending"


def test_batch_ignores_ids_outside_batch(iso_state, monkeypatch):
    """交付列了不在本批的 id → 忽略（防误销账他项）。"""
    seed_backlog(
        iso_state,
        [
            {"id": "s1", "status": "pending", "member": "figure"},
            {"id": "s2", "status": "pending", "member": "figure"},
        ],
    )
    seed, group = workflow.backlog_take_batch(4)
    monkeypatch.setattr(
        dispatch,
        "do_dispatch",
        lambda *a, **k: DispatchOutcome(
            code=0, kind="result", body="backlog id=s1,ghost-999"
        ),
    )
    drain._drain_batch(seed, group, drain.new_run_log(), dry=False)
    by_id = {i["id"]: i for i in read_backlog(iso_state)}
    assert by_id["s1"]["status"] == "done"
    assert by_id["s2"]["status"] == "pending"  # ghost-999 不在批内，s2 未被误销


def test_batch_blocked_only_seed_needs_leader(iso_state, monkeypatch):
    seed_backlog(
        iso_state,
        [
            {"id": "s1", "status": "pending", "member": "figure"},
            {"id": "s2", "status": "pending", "member": "figure"},
        ],
    )
    seed, group = workflow.backlog_take_batch(4)
    monkeypatch.setattr(
        dispatch,
        "do_dispatch",
        lambda *a, **k: DispatchOutcome(
            code=2, kind="blocked", body="", error="脏工作区"
        ),
    )
    results = drain._drain_batch(seed, group, drain.new_run_log(), dry=False)
    by_id = {i["id"]: i for i in read_backlog(iso_state)}
    assert by_id["s1"]["status"] == "needs_leader"
    assert by_id["s2"]["status"] == "pending", "组包不连坐"
    assert results[0]["status"] == "needs_leader"


def test_batch_quota_requeues_seed_group_pending(iso_state, monkeypatch):
    seed_backlog(
        iso_state,
        [
            {"id": "s1", "status": "pending", "member": "figure"},
            {"id": "s2", "status": "pending", "member": "figure"},
        ],
    )
    seed, group = workflow.backlog_take_batch(4)
    monkeypatch.setattr(
        dispatch,
        "do_dispatch",
        lambda *a, **k: DispatchOutcome(
            code=2, kind="quota_exhausted", body="", error="credit usage limit"
        ),
    )
    results = drain._drain_batch(seed, group, drain.new_run_log(), dry=False)
    by_id = {i["id"]: i for i in read_backlog(iso_state)}
    assert by_id["s1"]["status"] == "pending"  # 种子复位
    assert by_id["s2"]["status"] == "pending"
    assert results[0]["status"] == "quota_exhausted"


def test_batch_dry_completes_all(iso_state):
    seed_backlog(
        iso_state,
        [
            {"id": "s1", "status": "pending", "member": "figure"},
            {"id": "s2", "status": "pending", "member": "figure"},
            {"id": "s3", "status": "pending", "member": "figure"},
        ],
    )
    seed, group = workflow.backlog_take_batch(4)
    results = drain._drain_batch(seed, group, drain.new_run_log(), dry=True)
    by_id = {i["id"]: i for i in read_backlog(iso_state)}
    assert all(by_id[x]["status"] == "done" for x in ("s1", "s2", "s3"))
    assert all(by_id[x]["batch_size"] == 3 for x in ("s1", "s2", "s3"))
    assert all(r["status"] == "dry" for r in results)


# ---------------------------------------------------------------------------
# drain_backlog 集成：一批一派发；按 member 分组；未选下轮续吃
# ---------------------------------------------------------------------------
def test_drain_one_batch_for_same_member(iso_state, monkeypatch):
    seed_backlog(
        iso_state,
        [
            {"id": "s1", "status": "pending", "member": "figure"},
            {"id": "s2", "status": "pending", "member": "figure"},
            {"id": "s3", "status": "pending", "member": "figure"},
        ],
    )
    calls = []

    def fake(member, **kw):
        calls.append(kw.get("slug"))
        return DispatchOutcome(
            code=0, kind="result", body="done commit aaa1111\nbacklog id=s1,s2,s3"
        )

    monkeypatch.setattr(dispatch, "do_dispatch", fake)
    summary = drain.drain_backlog(drain.new_run_log())
    assert calls == ["s1"], "同提请者一批一次派发"
    assert summary["count"] == 3
    by_id = {i["id"]: i for i in read_backlog(iso_state)}
    assert all(by_id[x]["status"] == "done" for x in ("s1", "s2", "s3"))
    assert all(by_id[x]["batch_size"] == 3 for x in ("s1", "s2", "s3"))


def test_drain_groups_by_member_and_requeues_unselected(iso_state, monkeypatch):
    seed_backlog(
        iso_state,
        [
            {"id": "f1", "status": "pending", "member": "figure"},
            {"id": "l1", "status": "pending", "member": "leader"},
            {"id": "f2", "status": "pending", "member": "figure"},
        ],
    )
    calls = []

    def fake(member, **kw):
        slug = kw.get("slug")
        calls.append(slug)
        # devops 只吃种子（不选组包）
        return DispatchOutcome(
            code=0, kind="result", body=f"done commit aaa1111\nbacklog id={slug}"
        )

    monkeypatch.setattr(dispatch, "do_dispatch", fake)
    summary = drain.drain_backlog(drain.new_run_log())
    # 批1 seed=f1 group=[f2]（同 figure）→ 只吃 f1，f2 留 pending
    # 批2 seed=l1（leader，无同组）→ 吃 l1；批3 seed=f2 → 吃 f2
    assert calls == ["f1", "l1", "f2"]
    assert summary["count"] == 3
    assert all(i["status"] == "done" for i in read_backlog(iso_state))


# ---------------------------------------------------------------------------
# backlog_stats + cmd_stats：等待时长 + 批量分布
# ---------------------------------------------------------------------------
def test_backlog_stats_wait_and_batch(iso_state):
    seed_backlog(
        iso_state,
        [
            {
                "id": "a",
                "status": "done",
                "member": "m",
                "proposed": "2026-10-10T10:00:00+08:00",
                "done_at": "2026-10-10T11:00:00+08:00",
                "batch_size": 2,
            },
            {
                "id": "b",
                "status": "done",
                "member": "m",
                "proposed": "2026-10-10T10:00:00+08:00",
                "done_at": "2026-10-10T10:30:00+08:00",
                "batch_size": 2,
            },
            {"id": "c", "status": "pending", "member": "m"},
        ],
    )
    st = workflow.backlog_stats()
    assert st["done"] == 2 and st["pending"] == 1
    assert st["batch_dist"] == {2: 2}
    assert sorted(st["waits_s"]) == [1800.0, 3600.0]


def test_cmd_stats_prints_backlog_section(iso_state, monkeypatch, capsys):
    seed_backlog(
        iso_state,
        [
            {
                "id": "a",
                "status": "done",
                "member": "m",
                "proposed": "2026-10-10T10:00:00+08:00",
                "done_at": "2026-10-10T11:00:00+08:00",
                "batch_size": 3,
            }
        ],
    )
    monkeypatch.setattr(ledger, "LEDGER_PATH", iso_state / "empty-ledger.jsonl")
    workflow.cmd_stats(argparse.Namespace(plan=None, member=None, days=None))
    out = capsys.readouterr().out
    assert "backlog 消化" in out and "批量大小分布" in out and "等待时长" in out
