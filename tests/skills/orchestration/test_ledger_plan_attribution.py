"""台账 ``plan``/``step`` **归属**回归（2026-10-15 TUI 详情页实施中发现）。

病灶：``dispatch.do_dispatch`` 的入参 ``plan: str``（计划 id）在「档位退役改预设」
（c4b25ca/0876ecf，2026-10-13）后，被同名局部量 ``plan = _channels.plan_seats(...)`` 遮蔽 →
台账 ``LedgerEntry.plan`` 落的是**席表对象**（``{'tier': …, 'usable': […]}``），而全部消费面
（计划完成统计 / ``orch stats --plan`` / stats 行尾「计划=…」/ TUI 详情页环节耗时）都按 id
比对 → **静默读空**（不报错、只是永远查不到）。

本文件钉住修复后的形状：一跳派发后台账的 ``plan`` 就是传入的计划 id（str），``step`` 同理。
"""

from __future__ import annotations

from pysci.skills.orchestration.tools import dispatch, ledger

from .conftest import envelope, fake_headless

OK = "<result>完工，commit abc1234 已合并。</result>"
MEMBER = "quotamember"


def test_ledger_plan_is_the_plan_id_not_the_seat_table(iso_dispatch, monkeypatch):
    """success_criteria（连带面）：台账 plan 字段=计划 id——计划完成统计与环节耗时读得到。"""
    fake_headless(monkeypatch, lambda n, cmd: envelope(OK, is_error=False, rc=0))
    o = dispatch.do_dispatch(
        MEMBER,
        text="出图",
        quiet=True,
        no_checks=True,
        session="new",
        plan="plan-42",
        step="s3",
    )
    assert o.code == 0 and o.kind == "result"
    row = ledger.read_all()[-1]
    assert isinstance(row.plan, str), (
        f"plan 落成了 {type(row.plan).__name__}：{row.plan!r}"
    )
    assert row.plan == "plan-42"
    assert row.step == "s3"


def test_ledger_plan_stays_empty_without_a_plan(iso_dispatch, monkeypatch):
    """无计划的 adhoc 跳：plan 为空串（不得回填席表对象）。"""
    fake_headless(monkeypatch, lambda n, cmd: envelope(OK, is_error=False, rc=0))
    dispatch.do_dispatch(MEMBER, text="出图", quiet=True, no_checks=True, session="new")
    row = ledger.read_all()[-1]
    assert row.plan == "" and row.step == ""
