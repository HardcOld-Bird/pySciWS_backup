"""E0-B：提案 kind 结构化 + backlog success_criteria 字段 + approve 硬校验。

覆盖 backlog 20261011-210317-leader（README §6 判据纪律）三条：
② 新 approve 的 backlog 条目含**非空** kind 与 success_criteria；
③ 缺 kind 或缺判据的建议 → approve **拒绝入队**（非零退出 + 提示），
   判据缺失可经 ``--criteria`` 补录放行。
另含 delivery 解析端（parse_delivery 抓 infra_suggestion 的 kind 属性、
parse_suggestion_fields 从正文抽 kind/success_criteria）的单元断言。

workflow/dispatch 改动须阳性覆盖（20261010-193601 教训）——拒绝路径真跑
cmd_approve，断言它**没**把条目写进 backlog.json。
"""

from __future__ import annotations

import argparse

from pysci.skills.orchestration.tools import workflow
from pysci.skills.orchestration.tools.delivery import (
    parse_delivery,
    parse_suggestion_fields,
)

from .conftest import read_backlog, seed_raw_suggestion, seed_suggestion

# E0-A 之前 seed_suggestion 只写标题+正文；本批后它自带 kind+success_criteria。
# 这些「裸正文」用 seed_raw_suggestion 精确构造缺失场景，不依赖夹具默认。


def _args(sid, **kw):
    # E5b 后 approve 必填判决包（§5.1 四态）；本文件测的是 kind/判据两道闸，
    # 判决通道一律走 override（组长自裁），不与被测闸门口径混在一起。
    base = dict(jury=False, bench="", override="测试自裁")
    base.update(kw)
    return argparse.Namespace(suggestion_id=sid, note="", no_wake=True, **base)


# ---------------------------------------------------------------------------
# ② 合规建议 approve → backlog 条目带非空 kind/success_criteria
# ---------------------------------------------------------------------------
def test_approve_enqueues_kind_and_criteria(iso_state):
    sid = seed_suggestion(iso_state)  # 自带 kind: usability + success_criteria

    rc = workflow.cmd_approve(_args(sid))

    assert rc == 0
    item = read_backlog(iso_state)[0]
    assert item["kind"] == "usability"
    assert item["success_criteria"].strip()  # 非空
    assert "默认判据" in item["success_criteria"]


# ---------------------------------------------------------------------------
# ③ 缺 kind → 拒绝入队（非零 + 建议仍留在 pending，未改名）
# ---------------------------------------------------------------------------
def test_approve_missing_kind_rejects_enqueue(iso_state):
    sid = seed_raw_suggestion(
        iso_state,
        "# 改进建议（devops，待审批）\n\n只描述问题，没有 kind 行。\n\n"
        "success_criteria：\n- 有判据但缺 kind\n",
    )

    rc = workflow.cmd_approve(_args(sid))

    assert rc != 0
    # 拒绝入队：backlog.json 从未被创建（approve 成功才会写它）
    assert not (iso_state / "backlog.json").exists()
    # 建议未被移走（拒绝路径不改状态、不写回复）
    assert (iso_state / "suggestions" / f"{sid}.md").exists()


# ---------------------------------------------------------------------------
# ③ 缺 success_criteria → 拒绝；--criteria 补录后放行
# ---------------------------------------------------------------------------
def test_approve_missing_criteria_rejects(iso_state):
    sid = seed_raw_suggestion(
        iso_state,
        "# 改进建议（devops，待审批）\n\nkind: defect\n附复现证据。\n",
    )

    rc = workflow.cmd_approve(_args(sid))

    assert rc != 0
    assert not (iso_state / "backlog.json").exists()


def test_approve_criteria_supplied_via_flag_passes(iso_state):
    sid = seed_raw_suggestion(
        iso_state,
        "# 改进建议（devops，待审批）\n\nkind: defect\n附复现证据。\n",
    )

    rc = workflow.cmd_approve(_args(sid, criteria="跑 X 命令应得 Y"))

    assert rc == 0
    item = read_backlog(iso_state)[0]
    assert item["kind"] == "defect"
    assert item["success_criteria"] == "跑 X 命令应得 Y"


# ---------------------------------------------------------------------------
# 解析端单元断言
# ---------------------------------------------------------------------------
def test_parse_delivery_captures_infra_kind():
    text = (
        'ok <result>r</result><infra_suggestion kind="defect">body</infra_suggestion>'
    )

    d = parse_delivery(text)

    assert d.infra_suggestion == "body"
    assert d.infra_kind == "defect"


def test_parse_delivery_infra_without_kind_attr():
    """旧交付（属性早于 E0-B）：body 仍解析、kind 为空（不抛）。"""
    d = parse_delivery("<result>r</result><infra_suggestion>body</infra_suggestion>")
    assert d.infra_suggestion == "body"
    assert d.infra_kind == ""


def test_parse_suggestion_fields_reads_markers():
    body = (
        "# t\n\nkind: usability\n背景若干\n\n"
        "success_criteria（判据）：\n- 甲\n- 乙\n\n注意\n"
    )
    kind, criteria = parse_suggestion_fields(body)
    assert kind == "usability"
    assert "甲" in criteria and "乙" in criteria
    assert "注意" not in criteria  # 空行截断，后续正文不误入判据


def test_parse_suggestion_fields_absent_returns_empty():
    assert parse_suggestion_fields("纯文本，无标记行。") == ("", "")
