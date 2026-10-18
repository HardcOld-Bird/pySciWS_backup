"""建议队列：**超龄报警**（seed 20261017-050500）＋ **第三态销账**（菜单 20261017-050501）。

现象（leader 2026-10-17 实测）：`state/suggestions/` 累积 19 条 pending 超 24h **零报警**
（PENDING-WORK 巡更 5 项判据不含建议队列），「待审批建议=N」虚高误导组长逐条重判已办结项；
且建议只有 approve/reject 两态——**决策型**（本单不做改动）与**已被他单覆盖型**无家可归，
只能永远 pending（成员因原单不闭环而重报）。
"""

from __future__ import annotations

from argparse import Namespace
from datetime import datetime, timedelta

import pytest

from pysci.skills.orchestration.tools import journal, orch, watch, workflow

from .conftest import read_backlog, seed_raw_suggestion

BODY = (
    "# 改进建议（devops，待审批）\n\nkind: defect\n现象：某读数缺一项，复现见下。\n\n"
    "success_criteria:\n- 跑 `orch suggestions` 能看到龄期列\n"
)


def _sugg(state, *, at: datetime, member: str = "devops") -> str:
    """落一条 pending 建议，**后缀时刻即 id 前缀**（龄期由 id 决定，便于注入 now）。"""
    sid = at.strftime("%Y%m%d-%H%M%S") + f"-{member}"
    return seed_raw_suggestion(state, BODY, member=member, sid=sid)


def _args(**over) -> Namespace:
    base = dict(jury=False, bench="", override="测试自裁", note="", no_wake=True)
    base.update(over)
    return Namespace(**base)


# ---------------------------------------------------------------------------
# ① 积压读数（单一取数口）
# ---------------------------------------------------------------------------
def test_backlog_report_counts_aged_and_oldest(iso_state):
    now = datetime(2026, 10, 17, 12, 0, 0)
    _sugg(iso_state, at=now - timedelta(hours=31))
    _sugg(iso_state, at=now - timedelta(hours=1))

    back = workflow.suggestion_backlog(now=now)

    assert back["total"] == 2
    assert back["aged"] == 1
    assert back["oldest_hours"] == 31


def test_fresh_suggestions_are_not_aged(iso_state):
    now = datetime(2026, 10, 17, 12, 0, 0)
    _sugg(iso_state, at=now - timedelta(hours=3))
    assert workflow.suggestion_backlog(now=now) == {
        "total": 1,
        "aged": 0,
        "oldest_hours": 0,
        "oldest_id": "",
    }


def test_cross_day_counts_as_aged_even_under_24h(iso_state):
    """**跨日**即该处置（用户口径），不必等满 24h。"""
    now = datetime(2026, 10, 17, 1, 0, 0)
    sid = _sugg(iso_state, at=datetime(2026, 10, 16, 23, 0, 0))  # 仅 2h，但昨日
    back = workflow.suggestion_backlog(now=now)
    assert back["aged"] == 1 and back["oldest_id"] == sid


def test_cmd_suggestions_shows_age_column_and_stale_mark(iso_state, capsys):
    now = datetime.now()
    _sugg(iso_state, at=now - timedelta(hours=31))
    capsys.readouterr()
    assert workflow.cmd_suggestions(Namespace()) == 0
    out = capsys.readouterr().out
    assert "龄期=31h" in out and "⚠ 超 24h 未判决" in out, out
    assert "建议队列 1 条超 24h 未判决（最老 31h）" in out, out


# ---------------------------------------------------------------------------
# ② PENDING-WORK 巡更：超龄报警 / 未超龄静默
# ---------------------------------------------------------------------------
def test_patrol_reports_aged_suggestions(iso_state):
    now = datetime.now()
    _sugg(iso_state, at=now - timedelta(hours=25))

    press = watch.queue_pressure()

    assert any(
        "建议队列 1 条超 24h 未判决（最老 25h）" == r for r in press["reasons"]
    ), press["reasons"]
    assert press["pending"] is True


def test_patrol_silent_when_suggestions_fresh(iso_state):
    # 「新鲜」按建议面的真实判据＝**同一天**且未超 24h；不能用 now-2h（凌晨 0–2 点会
    # 跨日 → 按「跨日即该处置」报警，测试随运行时刻翻红）。
    _sugg(iso_state, at=datetime.now())
    press = watch.queue_pressure()
    assert not any("建议队列" in r for r in press["reasons"]), press["reasons"]


# ---------------------------------------------------------------------------
# ③ 第三态销账：superseded / decision
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("as_", ["superseded", "decision"])
def test_ack_closes_without_backlog_or_rejection(iso_state, as_, capsys):
    sid = _sugg(iso_state, at=datetime.now() - timedelta(hours=30))
    capsys.readouterr()

    rc = workflow.cmd_suggestions_ack(
        _args(
            suggestion_id=sid,
            as_=as_,
            by="20261017-999999-devops" if as_ == "superseded" else "组长裁定 #7",
            why="内容已由他单落地" if as_ == "superseded" else "仅留档供决策",
        )
    )

    assert rc == 0
    assert not (workflow.SUGGESTIONS_DIR / f"{sid}.md").exists(), "仍留在 pending"
    closed = workflow.SUGGESTIONS_DIR / workflow.CLOSED_DIR_NAME / f"{sid}.md"
    assert closed.is_file(), "未移入 closed/"
    reply = workflow.REPLIES_DIR / "devops" / f"{sid}.md"
    assert reply.is_file() and "非否决" in reply.read_text(encoding="utf-8")
    # 决策日志留痕（choice=ack，两态可辨）
    recs = [
        r
        for r in (journal.read_records() if hasattr(journal, "read_records") else [])
        if r.get("suggestion_id") == sid
    ]
    assert recs and recs[-1]["choice"] == "ack" and recs[-1]["ack_as"] == as_, recs
    assert not (iso_state / "backlog.json").exists(), "销账不该进 backlog"
    assert "已销账" in capsys.readouterr().out


def test_ack_rejects_unknown_id(iso_state, capsys):
    assert (
        workflow.cmd_suggestions_ack(
            _args(suggestion_id="nope", as_="decision", by="x", why="y")
        )
        == 2
    )
    assert "未找到待审批建议" in capsys.readouterr().out


def test_closed_suggestions_leave_the_pending_surface(iso_state):
    """销账后：pending 计数与积压读数都不再含它。"""
    sid = _sugg(iso_state, at=datetime.now() - timedelta(hours=30))
    workflow.cmd_suggestions_ack(
        _args(suggestion_id=sid, as_="decision", by="裁定 #1", why="留档")
    )
    assert workflow.pending_suggestions() == []
    assert workflow.suggestion_backlog()["aged"] == 0


# ---------------------------------------------------------------------------
# ④ 立单面 `supersedes:` 联动销账
# ---------------------------------------------------------------------------
def test_approve_with_supersedes_closes_covered_suggestions(iso_state, capsys):
    old = _sugg(iso_state, at=datetime.now() - timedelta(hours=50))
    new_id = "20261017-120000-devops"
    seed_raw_suggestion(
        iso_state,
        "# 改进建议（devops，待审批）\n\nkind: defect\n现象正文（补单）。\n\n"
        f"supersedes: {old}\n\nsuccess_criteria:\n- 旧单已闭环\n",
        sid=new_id,
    )
    capsys.readouterr()

    assert workflow.cmd_approve(_args(suggestion_id=new_id)) == 0

    # 新单进 backlog；旧单被联动销账（closed/，非否决），不进 backlog
    assert [i["id"] for i in read_backlog(iso_state)] == [new_id]
    assert (workflow.SUGGESTIONS_DIR / workflow.CLOSED_DIR_NAME / f"{old}.md").is_file()
    reply = workflow.REPLIES_DIR / "devops" / f"{old}.md"
    assert reply.is_file() and "销账" in reply.read_text(encoding="utf-8")
    out = capsys.readouterr().out
    assert "联动销账" in out, out


def test_supersedes_marker_forms(iso_state):
    """标记行前缀形态（行首/列表项/粗体）皆认；列表标记须跟空白（免吃 `**` 的头一个星）。"""
    assert workflow._supersedes_ids("supersedes: a,b") == ["a", "b"]
    assert workflow._supersedes_ids("- supersedes: a，b") == ["a", "b"]
    assert workflow._supersedes_ids("**supersedes**: a,b") == ["a", "b"]
    assert workflow._supersedes_ids("  Supersedes: a") == ["a"]
    assert workflow._supersedes_ids("supersedes：a") == ["a"]
    assert workflow._supersedes_ids("无关行") == []
    assert workflow._supersedes_ids("supersedes:") == []


# ---------------------------------------------------------------------------
# ⑦ `suggestions show <id>`：正文/证据/判据可查，归档件同样可查（tb-5 摩擦 #2）
# ---------------------------------------------------------------------------
def test_suggestions_show_pending_body(iso_state, capsys):
    sid = _sugg(iso_state, at=datetime(2026, 10, 18, 12, 0, 0))
    capsys.readouterr()
    assert orch.main(["suggestions", "show", sid[:8]]) == 0
    out = capsys.readouterr().out
    assert f"== 建议 {sid}（待审批）==" in out
    assert "kind: defect" in out and "success_criteria" in out
    assert "跑 `orch suggestions` 能看到龄期列" in out, "正文/判据必须原样输出"


def test_suggestions_show_archived_after_approval(iso_state, capsys):
    sid = _sugg(iso_state, at=datetime(2026, 10, 18, 12, 0, 0))
    workflow._archive(
        workflow.SUGGESTIONS_DIR / f"{sid}.md",
        workflow.SUGGESTIONS_DIR / "approved",
    )
    capsys.readouterr()
    assert orch.main(["suggestions", "show", sid]) == 0
    out = capsys.readouterr().out
    assert "（已采纳）" in out and "success_criteria" in out


def test_suggestions_show_missing_and_ambiguous(iso_state, capsys):
    sid = _sugg(iso_state, at=datetime(2026, 10, 18, 12, 0, 0))
    # 无命中：明确报不存在并给入口
    assert orch.main(["suggestions", "show", "20990101-000000-devops"]) == 2
    assert "未找到建议" in capsys.readouterr().out
    # 前缀歧义：列出候选、要求更长前缀
    other = seed_raw_suggestion(
        iso_state, BODY, member="devops", sid=sid[:12] + "59-devops"
    )
    assert orch.main(["suggestions", "show", sid[:12]]) == 2
    out = capsys.readouterr().out
    assert "命中 2 条" in out and other in out


def test_suggestions_show_requires_id():
    with pytest.raises(SystemExit) as exc:
        orch.main(["suggestions", "show"])
    assert exc.value.code == 2
