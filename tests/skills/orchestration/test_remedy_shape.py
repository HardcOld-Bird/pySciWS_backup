"""修复形状分流（用户 2026-10-13 05:5x 审查裁决，backlog 20261013-060000-leader）。

ctx 水位闸教训：mechanical 发现 → 自动晋升 → devops 自行发明**策略变更**类修复形状 →
销账，J/M/twin 零介入。分流规则把免判决收窄到「复原既有声明行为」（restore）；凡改变
声明行为/策略/权衡面（change）且无判决记录 → drain 任务书必须要求 devops 交付 blocked
上报，不得实施。三个落点各有阳性面：append/promote 带字段、任务书注入约束、旧条目不追溯。
"""

from __future__ import annotations

import pytest

from pysci.skills.inspector.tools import inspect as insp_cli
from pysci.skills.orchestration.tools import benchd, drain, inspectd, registry, workflow

from .conftest import seed_backlog

DEFAULT_INSPECTOR = registry.DEFAULT_INSPECTOR


@pytest.fixture
def iso_remedy(iso_inspect, monkeypatch):
    """晋升闸读数确定化（同 test_inspect_engine.iso_engine 的钉法）+ 派生全桩。"""
    monkeypatch.setattr(
        inspectd, "params", lambda: {**DEFAULT_INSPECTOR, "hysteresis_valid": True}
    )
    bench_q = iso_inspect / "bencher" / "queue.json"
    bench_q.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(benchd, "QUEUE_PATH", bench_q)
    monkeypatch.setattr(inspectd, "_spawn_detached", lambda argv, **kw: None)
    monkeypatch.setattr(drain, "_spawn_detached", lambda argv, **kw: None)
    return iso_inspect


def seed_mechanical(**over):
    kw = dict(
        cls="mechanical",
        title="断掉的正则不再匹配",
        symptom="规格行存在但匹配恒空",
        repro="pytest -k broken_regex",
        evidence="0 matched",
        expect="1 matched",
        kind="defect",
        criteria="- 正则命中规格行",
        stamp="20261013-060001",
    )
    kw.update(over)
    return inspectd.append_finding(**kw)


# ===========================================================================
#  1. 字段：登记缺省 change、词表硬闸、restore 显式可选
# ===========================================================================
def test_append_finding_defaults_to_change(iso_state):
    rec = seed_mechanical()
    assert rec["remedy_shape"] == "change", "缺省 change=保守走判决"


def test_append_finding_accepts_restore(iso_state):
    rec = seed_mechanical(remedy_shape="restore")
    assert rec["remedy_shape"] == "restore"


def test_append_finding_rejects_unknown_shape(iso_state):
    with pytest.raises(ValueError, match="修复形状"):
        seed_mechanical(remedy_shape="polish")
    assert inspectd.findings() == [], "被拒的登记不落台账"


# ===========================================================================
#  2. 晋升：backlog 条目强制带 remedy_shape
# ===========================================================================
def test_promote_entry_carries_default_change(iso_remedy):
    rec = seed_mechanical()
    out = inspectd.promote_finding(rec, channel="mechanical", wake=False)
    assert out["action"] == "promoted"
    entry = workflow.backlog_find(rec["id"])
    assert entry["remedy_shape"] == "change", "晋升条目必须带字段（缺省 change）"


def test_promote_entry_carries_restore(iso_remedy):
    rec = seed_mechanical(remedy_shape="restore", stamp="20261013-060002")
    out = inspectd.promote_finding(rec, channel="mechanical", wake=False)
    assert out["action"] == "promoted"
    assert workflow.backlog_find(rec["id"])["remedy_shape"] == "restore"


# ===========================================================================
#  3. 任务书注入：change+免判决 → 要求 blocked；restore/有判决 → 照常
# ===========================================================================
def _item(**over):
    base = {
        "id": "insp-x-mechanical",
        "member": "inspector",
        "source": "inspector",
        "summary": "某发现",
        "channel": "mechanical",
        "remedy_shape": "change",
        "status": "pending",
    }
    base.update(over)
    return base


def test_taskbook_change_without_verdict_requires_blocked(iso_state):
    text = workflow.backlog_task_text(_item())
    assert "remedy_shape=change" in text
    assert "未经判决不得实施" in text and "<blocked>" in text
    assert "remedy_shape=change 缺判决记录" in text, "blocked 正文要给出固定短语"


def test_taskbook_batch_text_carries_change_constraint(iso_state):
    """批量任务书同样注入（组包种子路径走 backlog_batch_text → backlog_task_text）。"""
    seed_backlog(iso_state, [_item()])
    seed, group = workflow.backlog_take_batch(4)
    text = workflow.backlog_batch_text(seed, group, max_batch=4)
    assert "未经判决不得实施" in text


def test_taskbook_restore_proceeds_without_blocked(iso_state):
    text = workflow.backlog_task_text(_item(remedy_shape="restore"))
    assert "修复形状=restore" in text
    # 通用尾部本就写「合并前 git status 检查…交付 <blocked>」，故只否形状闸那一段
    assert "修复形状约束" not in text and "未经判决不得实施" not in text


def test_taskbook_change_with_jury_verdict_proceeds(iso_state):
    text = workflow.backlog_task_text(_item(channel="jury"))
    assert "判决记录已在案" in text and "不得实施" not in text


def test_taskbook_old_entries_untouched(iso_state):
    """旧条目（无 remedy_shape 字段）不注入——分流只执法于新体系，不追溯。"""
    text = workflow.backlog_task_text(_item(remedy_shape=None, channel=None))
    assert "修复形状约束" not in text


def test_taskbook_non_inspector_source_untouched(iso_state):
    """leader 审批面（approve 判决包闸已管它）不注入 inspector 分流约束。"""
    text = workflow.backlog_task_text(_item(source="leader", remedy_shape=None))
    assert "修复形状约束" not in text


# ===========================================================================
#  4. CLI 面：--remedy-shape 接线
# ===========================================================================
def test_cli_finding_add_wires_remedy_shape(iso_state, capsys):
    rc = insp_cli.main(
        [
            "finding",
            "add",
            "--class",
            "mechanical",
            "--title",
            "闸形缺陷",
            "--repro",
            "cmd",
            "--evidence",
            "out",
            "--criteria",
            "- 判据",
            "--remedy-shape",
            "restore",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "remedy_shape=restore" in out
    assert inspectd.findings()[0]["remedy_shape"] == "restore"


def test_cli_finding_add_rejects_bad_shape(iso_state, capsys):
    with pytest.raises(SystemExit) as ei:
        insp_cli.main(
            [
                "finding",
                "add",
                "--class",
                "mechanical",
                "--title",
                "x",
                "--repro",
                "c",
                "--evidence",
                "e",
                "--criteria",
                "- c",
                "--remedy-shape",
                "polish",
            ]
        )
    assert ei.value.code == 2, "argparse choices 在登记前拦住"
    assert inspectd.findings() == [], "被拒的形状不得留下半条发现"


# ===========================================================================
#  4. 两源一致：形状以 findings 行为准（backlog 20261013-184040-devops）
# ===========================================================================
def test_taskbook_follows_findings_row_over_backlog_item(iso_state, iso_remedy):
    """组长在 findings 行裁定 restore → 重派任务书**不再**注入矛盾的 blocked 指令。

    复现路径（实测 `insp-20261013-045910-mechanical`）：backlog 条目上的形状是晋升瞬间
    从 findings 行拷来的，组长事后改 findings 行不会回流；执法只读 backlog，于是任务书
    照旧要求「change 缺判决 → blocked」，组员照做就白烧一跳。
    """
    rec = seed_mechanical(remedy_shape="restore", stamp="20261013-184041")
    item = _item(id=rec["id"], remedy_shape="change")  # backlog 侧是陈旧拷贝

    text = workflow.backlog_task_text(item)
    assert "直接交付 <blocked>" not in text, (
        "findings 行已裁定 restore，不该再要求 blocked"
    )
    assert "修复形状=restore" in text
    assert "两源不一致" in text, "两源不一致必须出声，不许默默按某一侧走"


def test_taskbook_still_blocks_change_without_judgment(iso_state, iso_remedy):
    """两源一致为 change 且无判决 → 仍要求 blocked（不放松原闸）。"""
    rec = seed_mechanical(stamp="20261013-184042")  # 缺省 change
    text = workflow.backlog_task_text(_item(id=rec["id"]))
    assert "直接交付 <blocked>" in text
    assert "两源不一致" not in text


def test_taskbook_unblocks_when_findings_has_judgment(iso_state, iso_remedy):
    """findings 行有 `judgment` 留痕 → change 也照常实施（判决记录已在案）。"""
    rec = seed_mechanical(stamp="20261013-184043")
    inspectd._mutate_finding(rec["id"], judgment="leader-ruling@2026-10-13：已裁定")

    text = workflow.backlog_task_text(_item(id=rec["id"]))
    assert "直接交付 <blocked>" not in text
    assert "判决记录已在案" in text


def test_reverse_mismatch_blocks_and_surfaces(iso_state, iso_remedy):
    """**反向**不一致（findings=change、backlog=restore）也要出声并回到 change 支。

    这一向更危险：只读 backlog 会**静默放行**一段本该过判决的实施，直接绕过免判决通道的
    边界（缺陷报告里的后果②）。
    """
    rec = seed_mechanical(stamp="20261013-184044")  # findings 侧是 change（缺省）
    item = _item(id=rec["id"], remedy_shape="restore")  # backlog 侧陈旧为 restore

    text = workflow.backlog_task_text(item)
    assert "直接交付 <blocked>" in text, "以 findings 行为准 → 仍是 change 无判决"
    assert "两源不一致" in text


def test_shape_falls_back_to_backlog_when_findings_row_absent(iso_state, iso_remedy):
    """findings 行不在（旧条目/已清理）→ 回落 backlog 字段，行为与修前一致。"""
    text = workflow.backlog_task_text(_item(id="insp-20260101-000000-mechanical"))
    assert "直接交付 <blocked>" in text
    assert "两源不一致" not in text
