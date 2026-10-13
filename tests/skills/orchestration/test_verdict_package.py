"""E5b 组长判决包流（backlog 20261012-120004-leader，README §5.3）。

钉住四件事，每件都是「不写代码就会靠人记」的口径：

1. **四态硬闸**（§5.1）：approve 必须给 J / M / J+M / override 之一，否则非零退出；
   唯一例外是「缺陷类且当条交付含 blocked」——它的判据来自落盘时就固化的
   ``delivery:`` 标记行，不是事后反查台账。
2. **仲裁直接生效**：裁决 A 入队、B **自动驳回且不计 override 率**、模棱两可挂起，
   组长不再手动翻译判决结果（这正是「基建判断与组长职责事实分离」的落点）。
3. **核验闭环**（§5.6）：审批时定死核验方式（基线 commit 进条目 ``verify`` 字段），
   销账时机械排出 A/B，判决三态自动落地——改善销账 / 未改善自动立案纠正提案 /
   不可判挂起。防重复：``stage`` 判据保证「实施前至多一次仲裁、实施后只实测」。
4. **账目可读**：通道用量与 override 率从决策日志回读，进 ``orch status``。

真仲裁与真 A/B 在别处测（test_jury.py / replica 单元）；本文件用 ``verdict.run_jury``
与 ``replica.run_ab`` 两个既有门面试判决落地，组卷格式则用真 :func:`jury.parse_case`
断言——共享库的案卷契约漂移必须在这里就红。
"""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import pytest

from pysci.skills.orchestration.tools import (
    benchd,
    dispatch,
    journal,
    jury,
    orch,
    replica,
    verdict,
    workflow,
)
from pysci.skills.orchestration.tools.delivery import parse_suggestion_fields

from .conftest import (
    read_backlog,
    seed_backlog,
    seed_raw_suggestion,
    seed_suggestion,
)

REASON = "测试自裁（判决包 override 通道）"


def _args(sid: str, **kw) -> Namespace:
    """approve 的测试参数：默认不选任何通道（测闸门的用例自己补）。"""
    base = dict(
        suggestion_id=sid, note="", no_wake=True, jury=False, bench="", override=""
    )
    base.update(kw)
    return Namespace(**base)


def _seed_blocked_defect(state, sid=None) -> str:
    """落一条「缺陷类 + 当条交付 blocked」的建议（例外通道的唯一入口）。"""
    return seed_raw_suggestion(
        state,
        "# 改进建议（devops，待审批）\n\nkind: defect\ndelivery: blocked\n\n"
        "锁判活把收尾中的 worker 误判为死，导致重复派生。\n\n"
        "success_criteria：\n- 构造收尾中 worker → lock_status 判活\n",
        sid=sid,
    )


def _stub_jury(monkeypatch, *, status="recorded", letter="A", **extra):
    """桩掉仲裁门面并记下组卷路径（真开庭由 test_jury.py 覆盖）。"""
    calls: list[str] = []

    def fake(path, *, dry=False, timeout=None):
        calls.append(str(path))
        record = {"verdict": letter, "tally": {"A": 3, "B": 1}, **extra}
        return jury.Run(status, f"桩：裁决={letter}", record=record, path=path)

    monkeypatch.setattr(verdict, "run_jury", fake)
    return calls


@pytest.fixture
def iso_head(monkeypatch):
    """钉死基线锚点（读数不随运行仓库的真实 commit 漂移）。"""
    monkeypatch.setattr(verdict, "head_commit", lambda *, short=False: "deadbeef")
    return "deadbeef"


# ===========================================================================
#  1. 四态硬闸与例外通道
# ===========================================================================


def test_approve_without_package_is_rejected(iso_state, capsys):
    """无判决包 → 非零退出、不入队、建议原地不动。"""
    sid = seed_suggestion(iso_state)  # kind=usability，无 delivery 标记

    assert workflow.cmd_approve(_args(sid, note="采纳")) == 2
    assert not (iso_state / "backlog.json").exists()
    assert (iso_state / "suggestions" / f"{sid}.md").exists()
    out = capsys.readouterr().out
    assert "缺判决包" in out and "--jury" in out and "--override" in out


def test_defect_with_blocked_delivery_takes_exception_channel(iso_state, capsys):
    """例外通道实测保留：缺陷类且当条交付含 blocked → 裸 approve 仍可行。"""
    sid = _seed_blocked_defect(iso_state)

    assert workflow.cmd_approve(_args(sid, note="采纳")) == 0
    out = capsys.readouterr().out
    assert "缺陷例外" in out and "--bench 复跑原失败案" in out
    assert read_backlog(iso_state)[0]["kind"] == "defect"


def test_defect_without_blocked_delivery_still_needs_package(iso_state):
    """缺陷但交付是 result（提案出于成功跑）→ 不在例外通道，仍要判决包。"""
    sid = seed_raw_suggestion(
        iso_state,
        "# 改进建议（devops，待审批）\n\nkind: defect\ndelivery: result\n\n"
        "统计口径与文档描述不一致，附复现命令若干。\n\n"
        "success_criteria：\n- 跑复现命令应得文档所述输出\n",
    )

    assert workflow.cmd_approve(_args(sid)) == 2


def test_legacy_suggestion_without_delivery_marker_needs_package(iso_state):
    """旧建议（无 delivery 标记）不享例外通道——缺失=背后没有失败实据。"""
    sid = seed_raw_suggestion(
        iso_state,
        "# 改进建议（devops，待审批）\n\nkind: defect\n\n"
        "缺陷描述若干，够长够具体以满足空壳门槛，判据见下。\n\n"
        "success_criteria：\n- 复现命令由失败转通过\n",
    )

    assert workflow.cmd_approve(_args(sid)) == 2


def test_override_conflicts_with_jury(iso_state, capsys):
    """override 是「不选 J/M」的分支，混给即拒绝（通道读数不容污染）。"""
    sid = seed_suggestion(iso_state)

    assert workflow.cmd_approve(_args(sid, override=REASON, jury=True)) == 2
    assert "冲突" in capsys.readouterr().out


def test_summary_line_skips_head_markers(iso_state):
    """delivery/kind 标记行不得被当成建议摘要（backlog 摘要面给人读的是内容）。"""
    sid = _seed_blocked_defect(iso_state)

    assert workflow.cmd_approve(_args(sid)) == 0
    assert read_backlog(iso_state)[0]["summary"].startswith("锁判活")


# ===========================================================================
#  2. J 路：仲裁直接生效
# ===========================================================================


def test_jury_A_enqueues_and_dossier_is_parseable(iso_state, monkeypatch):
    """裁决 A → 入队；自动组的案卷须是共享库能解析的真案卷（契约不漂移）。"""
    sid = seed_suggestion(iso_state)
    paths = _stub_jury(monkeypatch, letter="A")

    assert workflow.cmd_approve(_args(sid, jury=True, note="采纳")) == 0
    assert read_backlog(iso_state)[0]["id"] == sid
    case = jury.parse_case(paths[0])
    assert case.question and case.option_a and case.option_b
    assert "默认判据" in case.evidence


def test_jury_B_auto_rejects_and_is_not_override(iso_state, monkeypatch, capsys):
    """裁决 B → 自动落驳回（组长只写回复），且这笔不计入 override 率。"""
    sid = seed_suggestion(iso_state)
    _stub_jury(monkeypatch, letter="B")

    assert workflow.cmd_approve(_args(sid, jury=True)) == 0
    assert not (iso_state / "suggestions" / f"{sid}.md").exists()
    assert (iso_state / "suggestions" / "rejected" / f"{sid}.md").exists()
    assert (iso_state / "replies" / "devops" / f"{sid}.md").exists()
    assert not (iso_state / "backlog.json").exists()
    rows = journal.read_records()
    assert rows[-1]["choice"] == "reject" and rows[-1]["channel"] == "jury"
    assert "不计 override" in capsys.readouterr().out
    assert verdict.health()["channels"]["override"] == 0


def test_jury_inconclusive_keeps_suggestion_pending(iso_state, monkeypatch, capsys):
    """模棱两可（escalate=twin）→ 不入队也不驳回，挂起呈组长并留痕。"""
    sid = seed_suggestion(iso_state)
    _stub_jury(monkeypatch, letter=None, escalate="twin")

    assert workflow.cmd_approve(_args(sid, jury=True)) == 1
    assert (iso_state / "suggestions" / f"{sid}.md").exists()
    assert not (iso_state / "backlog.json").exists()
    assert journal.read_records()[-1]["choice"] == "pending"
    assert "escalate=twin" in capsys.readouterr().out


def test_jury_degraded_does_not_enqueue(iso_state, monkeypatch, capsys):
    """渠道降级（queued）时不得入队——否则「过了仲裁」这个读数就是假的。"""
    sid = seed_suggestion(iso_state)
    monkeypatch.setattr(
        verdict,
        "run_jury",
        lambda *a, **k: jury.Run("queued", "降级入队：缺 SILICONFLOW_API_KEY"),
    )

    assert workflow.cmd_approve(_args(sid, jury=True)) == 1
    assert not (iso_state / "backlog.json").exists()
    assert "仲裁未出判决" in capsys.readouterr().out


# ===========================================================================
#  3. M 路与 §5.6 核验闭环
# ===========================================================================


def test_bench_channel_captures_baseline_and_marks_entry(
    iso_state, iso_bench, iso_head, capsys
):
    sid = seed_suggestion(iso_state)

    assert workflow.cmd_approve(_args(sid, bench="fig-basic", note="采纳")) == 0
    q = benchd.queue_load()
    assert [i["kind"] for i in q] == ["capture"]
    assert q[0]["ref"] == "deadbeef" and q[0]["cases"] == ["fig-basic"]
    assert read_backlog(iso_state)[0]["verify"] == {
        "stage": "approved",
        "cases": ["fig-basic"],
        "baseline_commit": "deadbeef",
        "baseline_item": q[0]["id"],
        "label": f"post-verify-{sid}",
    }
    assert "判决通道=bench" in capsys.readouterr().out


def test_completion_auto_queues_ab_exactly_once(iso_state, iso_bench, iso_head):
    """销账时机械排出 A/B；重复销账（批量消化/孤儿重跑）不得排出第二项。"""
    sid = seed_suggestion(iso_state)
    workflow.cmd_approve(_args(sid, bench="fig-basic"))
    before = len(benchd.queue_load())

    workflow.backlog_complete(sid, "devops 交付")
    workflow.backlog_complete(sid, "重复销账")

    ab = [i for i in benchd.queue_load()[before:] if i["kind"] == "ab"]
    assert len(ab) == 1
    assert ab[0]["ref_a"] == "deadbeef" and ab[0]["ref_b"] == "HEAD"
    assert ab[0]["backlog_id"] == sid and ab[0]["cases"] == ["fig-basic"]
    assert read_backlog(iso_state)[0]["verify"]["stage"] == "verifying"


def test_non_bench_entry_queues_nothing(iso_state, iso_bench, iso_head):
    sid = seed_suggestion(iso_state)
    workflow.cmd_approve(_args(sid, override=REASON))

    workflow.backlog_complete(sid)
    assert benchd.queue_load() == []


def test_improvement_clears_the_ledger(iso_state, iso_bench, iso_head):
    sid = seed_suggestion(iso_state)
    workflow.cmd_approve(_args(sid, bench="fig-basic"))
    workflow.backlog_complete(sid)
    ab = benchd.queue_load()[-1]

    out = verdict.apply_ab_verdict(ab, {"verdict": "改善", "label": ab["label"]})

    assert out["action"] == "cleared"
    entry = read_backlog(iso_state)[0]
    assert entry["status"] == "done" and entry["verified_at"]
    assert entry["verify"]["stage"] == "cleared"
    assert "核验通过" in entry["note"]


def test_regression_files_correction_case(iso_state, iso_bench, iso_head):
    """未改善 → 自动生成纠正提案（新案卷进正常判决流，分类固定 defect）。"""
    sid = seed_suggestion(iso_state)
    workflow.cmd_approve(_args(sid, bench="fig-basic"))
    workflow.backlog_complete(sid)
    ab = benchd.queue_load()[-1]

    doc = {
        "verdict": "未改善",
        "label": "post-verify-x",
        "ref_a": "deadbeef",
        "ref_b": "HEAD",
        "commit_a": "aaaa",
        "commit_b": "bbbb",
        "compared": ["fig-basic"],
    }
    out = verdict.apply_ab_verdict(ab, doc)
    assert out["action"] == "correction" and out["suggestion_id"]

    body = (iso_state / "suggestions" / f"{out['suggestion_id']}.md").read_text(
        encoding="utf-8"
    )
    kind, criteria = parse_suggestion_fields(body)
    assert kind == "defect" and "fig-basic" in criteria
    assert "source: post-verify" in body and "delivery:" not in body
    assert read_backlog(iso_state)[0]["verify"]["correction"] == out["suggestion_id"]


def test_inconclusive_escalates_to_leader(iso_state, iso_bench, iso_head):
    sid = seed_suggestion(iso_state)
    workflow.cmd_approve(_args(sid, bench="fig-basic"))
    workflow.backlog_complete(sid)
    ab = benchd.queue_load()[-1]

    out = verdict.apply_ab_verdict(
        ab,
        {"verdict": "inconclusive", "label": ab["label"], "error": "窗口期污染"},
    )

    assert out["action"] == "escalated"
    entry = read_backlog(iso_state)[0]
    assert entry["verify"]["stage"] == "escalated"
    assert entry["verify"]["escalate"] == "twin"
    assert sid in verdict.health()["escalate"]


def test_ab_worker_runs_post_verify_hook(iso_bench, monkeypatch, tmp_path):
    """benchd 消化核验项时自动落地判决（不靠人记得去销账）。"""
    seed_backlog(
        iso_bench, [{"id": "bk-1", "status": "done", "verify": {"stage": "verifying"}}]
    )
    item = benchd.enqueue(
        "ab", ref_a="a1", ref_b="b2", cases=["fig-basic"], backlog_id="bk-1"
    )
    monkeypatch.setattr(
        replica, "run_ab", lambda *a, **k: {"verdict": "改善", "label": k["label"]}
    )

    doc = benchd._run_kind_ab(item, tmp_path / "run.log", dry=False)

    assert doc["post_verify"]["action"] == "cleared"
    assert read_backlog(iso_bench)[0]["verify"]["stage"] == "cleared"


def test_plain_ab_item_does_not_touch_backlog(iso_bench, monkeypatch, tmp_path):
    """普通测量项（无 backlog_id）判决不外溢——M 通道之外的读数不该改提案账。"""
    item = benchd.enqueue("ab", ref_a="a1", ref_b="b2", cases=["fig-basic"])
    seen: list = []
    monkeypatch.setattr(
        verdict,
        "apply_ab_verdict",
        lambda *a, **k: seen.append(a) or {"action": "x"},
    )
    monkeypatch.setattr(
        replica, "run_ab", lambda *a, **k: {"verdict": "改善", "label": k["label"]}
    )

    benchd._run_kind_ab(item, tmp_path / "run.log", dry=False)
    assert seen == []


# ===========================================================================
#  4. 读数面与健康度
# ===========================================================================


def test_health_counts_channels_and_override_rate(iso_state, iso_bench, iso_head):
    seed_suggestion(iso_state, sid="o-1")
    seed_suggestion(iso_state, sid="m-2")
    seed_suggestion(iso_state, sid="r-3")
    workflow.cmd_approve(_args("o-1", override=REASON))
    workflow.cmd_approve(_args("m-2", bench="fig-basic"))
    workflow.cmd_reject(Namespace(suggestion_id="r-3", override="无价值", note=""))

    h = verdict.health()
    assert h["channels"]["override"] == 2 and h["channels"]["bench"] == 1
    assert h["decided"] == 3
    assert h["override_rate"] == pytest.approx(2 / 3)


def test_orch_status_carries_verdict_package_section(iso_state, iso_registry, capsys):
    journal.append({"choice": "approve", "channel": "jury", "source": "jury"})

    orch.cmd_status(None)
    out = capsys.readouterr().out
    assert "判决包健康" in out and "J=1" in out and "override 率=0%" in out


def test_orch_status_without_records_says_so(iso_state, iso_registry, capsys):
    orch.cmd_status(None)
    assert "尚无判决包记录" in capsys.readouterr().out


# ===========================================================================
#  5. 咨询环节与落盘标记
# ===========================================================================


def test_consult_twin_refused_when_disabled(iso_state, iso_registry, capsys, tmp_path):
    case = tmp_path / "case.md"
    case.write_text(
        "## 问题\n要不要改\n\n## 选项 A\n改\n\n## 选项 B\n不改\n", encoding="utf-8"
    )

    rc = orch.main(["consult-twin", str(case)])

    assert rc == 2
    out = capsys.readouterr().out
    assert "twin 偏好面未启用" in out and "orch consult" in out


def test_consult_twin_dispatches_when_enabled(iso_state, monkeypatch, tmp_path):
    case = tmp_path / "case.md"
    case.write_text("案卷正文", encoding="utf-8")
    seen: list[tuple] = []
    monkeypatch.setattr(verdict, "taste_enabled", lambda: True)
    monkeypatch.setattr(
        dispatch,
        "do_dispatch",
        lambda member, **kw: (
            seen.append((member, kw)) or SimpleNamespace(code=0, error="")
        ),
    )

    assert orch.main(["consult-twin", str(case)]) == 0
    assert seen[0][0] == "twin" and "案卷正文" in seen[0][1]["text"]


def test_persist_suggestion_records_delivery_marker(iso_state):
    """落盘面写 delivery: 标记——例外通道的判据必须在那一跳就固化。"""
    sid = dispatch._persist_suggestion("figure", "正文若干", "defect", "blocked")

    body = (iso_state / "suggestions" / f"{sid}.md").read_text(encoding="utf-8")
    lines = body.splitlines()
    assert lines[2] == "kind: defect" and lines[3] == "delivery: blocked"
    assert verdict.delivery_of(body) == "blocked"


def test_reapproval_keeps_both_archived_originals(iso_state):
    """归档重名不得覆盖：上一笔判决的原文正是审计对象（实测崩溃回归钉）。"""
    seed_suggestion(iso_state, sid="dup-1")
    assert workflow.cmd_approve(_args("dup-1", override=REASON)) == 0
    seed_suggestion(iso_state, sid="dup-1")

    assert workflow.cmd_approve(_args("dup-1", override=REASON)) == 0
    assert len(list((iso_state / "suggestions" / "approved").glob("dup-1*.md"))) == 2
